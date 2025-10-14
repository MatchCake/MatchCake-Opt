#%%
import os
import time
from pathlib import Path
from typing import Optional, Any
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml
import torch
from ax import RangeParameterConfig
from matchcake import NonInteractingFermionicDevice
from matchcake.operations import SptmAngleEmbedding, SptmfRxRx, SptmFHH

from matchcake_opt.datasets import *
from matchcake_opt.modules.classification_model import ClassificationModel
from matchcake_opt.tr_pipeline.automl_pipeline import AutoMLPipeline
from matchcake_opt.tr_pipeline.lightning_pipeline import LightningPipeline
#%%
class NIFDL(ClassificationModel):
    MODEL_NAME = "NIFDL"
    DEFAULT_N_QUBITS = 16
    DEFAULT_LEARNING_RATE = 2e-4
    DEFAULT_N_LAYERS = 6

    HP_CONFIGS = [
        RangeParameterConfig(
            name="learning_rate",
            parameter_type="float",
            bounds=(1e-5, 0.1),
        ),
        RangeParameterConfig(
            name="n_qubits",
            parameter_type="int",
            bounds=(4, 32),
            step_size=2,
        ),
        RangeParameterConfig(
            name="n_layers",
            parameter_type="int",
            bounds=(1, 10),
        ),
    ]

    def __init__(
            self,
            input_shape: Optional[tuple[int, ...]],
            output_shape: Optional[tuple[int, ...]],
            learning_rate: float = DEFAULT_LEARNING_RATE,
            n_qubits: int = DEFAULT_N_QUBITS,
            n_layers: int = DEFAULT_N_LAYERS,
            **kwargs,
    ):
        super().__init__(input_shape=input_shape, output_shape=output_shape, learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters("learning_rate", "n_qubits", "n_layers")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_encoders = 8
        self.R_DTYPE = torch.float32
        self.C_DTYPE = torch.cfloat
        self.q_device = NonInteractingFermionicDevice(
            wires=self.n_qubits, r_dtype=self.R_DTYPE, c_dtype=self.C_DTYPE, show_progress=False
        )
        self.q_node = qml.QNode(self.circuit, self.q_device, interface="torch", diff_method="backprop")
        self._weight_shapes = {"weights": (self.n_layers, (self.n_qubits - 1) * 2)}
        self.flatten = torch.nn.Flatten()
        self.encoders = torch.nn.ModuleList(
            [
                qml.qnn.TorchLayer(self.q_node, self._weight_shapes)
                for _ in range(self.n_encoders)
            ]
        )
        self.readout = torch.nn.LazyLinear(self.output_size)
        self._build()

    def _build(self):
        dummy_input = torch.randn((3, *self.input_shape)).to(device=self.device)
        with torch.no_grad():
            self(dummy_input)
        return self

    def circuit(self, inputs, weights):
        SptmAngleEmbedding(inputs, wires=range(self.n_qubits))
        for i in range(self.n_layers):
          for j in range(self.n_qubits - 1):
            SptmfRxRx(weights[i, j*2 : j*2+2], wires=[j, j+1])
            SptmFHH(wires=[j, j+1])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, x) -> Any:
        x = self.flatten(x).to(self.device)
        x_split = torch.split(x, self.n_qubits, dim=1)
        x_out = [layer(x_chunk) for layer, x_chunk in zip(self.encoders, x_split)]
        x = torch.cat(x_out, dim=1).to(self.device)
        x = self.readout(x)
        return x

    @property
    def input_size(self):
        return int(np.prod(self.input_shape))

    @property
    def output_size(self):
        return int(np.prod(self.output_shape))
#%%
# Dataset
dataset_name = "Digits2D"
fold_id = 0
batch_size = 32
random_state = 0
num_workers = 0

# Model
model_cls = NIFDL

# Pipeline
job_output_folder = Path(os.getcwd()) / "data" / "automl" / dataset_name / model_cls.MODEL_NAME
checkpoint_folder = Path(job_output_folder) / "checkpoints"
pipeline_args = dict(
    max_epochs=128,  # increase at least to 256
    max_time="00:00:05:00",  # DD:HH:MM:SS, increase at least to "00:01:00:00"
)
#%%
datamodule = DataModule.from_dataset_name(
    dataset_name,
    fold_id=fold_id,
    batch_size=batch_size,
    random_state=random_state,
    num_workers=num_workers,
)
automl_pipeline = AutoMLPipeline(
    model_cls=model_cls,
    datamodule=datamodule,
    checkpoint_folder=checkpoint_folder,
    automl_iterations=5,  # increase at least to 32
    inner_max_epochs=12,  # increase at least to 128
    inner_max_time="00:00:02:00",  # increase at least to "00:00:10:00"
    automl_overwrite_fit=True,
    **pipeline_args
)
#%%
# lightning_pipeline = LightningPipeline(
#     model_cls=model_cls,
#     datamodule=datamodule,
#     checkpoint_folder=checkpoint_folder,
#     max_epochs=10,
#     max_time="00:00:03:00",  # DD:HH:MM:SS
#     overwrite_fit=True,
#     verbose=True,
#     **dict(
#         n_qubits=27,
#         n_layers=6,
#         learning_rate=2e-4,
#     ),
# )
# metrics = lightning_pipeline.run()
# print("⚡" * 20, "\nValidation Metrics:\n", metrics, "\n", "⚡" * 20)
# exit()
#%%
start_time = time.perf_counter()
automl_pipeline.run()
end_time = time.perf_counter()
print(f"Time taken: {end_time - start_time:.4f} seconds")
#%%
print(f"Best Hyperparameters:\n{json.dumps(automl_pipeline.get_best_params(), indent=2, default=str)}")
#%%
lt_pipeline, metrics = automl_pipeline.run_best_pipeline()
print("⚡" * 20, "\nValidation Metrics:\n", metrics, "\n", "⚡" * 20)
#%%
test_metrics = lt_pipeline.run_test()
print("⚡" * 20, "\nTest Metrics:\n", test_metrics, "\n", "⚡" * 20)
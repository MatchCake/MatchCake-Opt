import argparse
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from matchcake_opt.datamodules.datamodule import DataModule
from matchcake_opt.modules.classification_model import ClassificationModel
from matchcake_opt.tr_pipeline.lightning_pipeline import LightningPipeline


class TestLightningPipeline:
    @pytest.fixture(autouse=True, scope="class")
    def checkpoint_folder(self):
        folder = Path(".tmp") / "checkpoints" / "lightning"
        yield folder
        shutil.rmtree(folder, ignore_errors=True)

    @pytest.fixture
    def datamodule_instance(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr("matchcake_opt.datasets.digits2d.load_digits", mock)
        mock.return_value = (np.zeros((10, 8 * 8)), np.zeros((10,), dtype=int))
        datamodule = DataModule.from_dataset_name("digits2d", 0)
        return datamodule

    @pytest.fixture
    def model_cls(self, monkeypatch):
        monkeypatch.setattr("matchcake_opt.modules.base_model.BaseModel.configure_metrics", MagicMock())
        model = ClassificationModel((8, 8), (10,))
        model.parameters = MagicMock()
        param = torch.nn.Parameter(torch.Tensor([1]))
        model.parameters.return_value = [param]
        model.forward = lambda x: param * torch.ones((x.shape[0], *model.output_shape))
        model_cls = MagicMock()
        model_cls.return_value = model
        model_cls.MODEL_NAME = model.MODEL_NAME
        return model_cls

    @pytest.fixture
    def pipeline_instance(self, model_cls, datamodule_instance, checkpoint_folder):
        pipeline = LightningPipeline(
            model_cls,
            datamodule_instance,
            checkpoint_folder=checkpoint_folder,
            max_epochs=2,
            accelerator="cpu",
        )
        return pipeline

    def test_add_specific_args(self):
        parser = LightningPipeline.add_specific_args()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_run_and_run_test(self, pipeline_instance):
        metrics = pipeline_instance.run()
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "train_loss" in metrics
        assert "test_loss" not in metrics
        test_metrics = pipeline_instance.run_test()
        assert isinstance(test_metrics, dict)
        assert "test_loss" in test_metrics

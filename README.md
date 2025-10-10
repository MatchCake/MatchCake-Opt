# MatchCake-Opt

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/MatchCake-Opt.svg?style=social)](https://github.com/MatchCake/MatchCake-Opt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/MatchCake-Opt?style=social)](https://github.com/MatchCake/MatchCake-Opt/network/members)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![downloads](https://img.shields.io/pypi/dm/MatchCake-Opt)](https://pypi.org/project/MatchCake-Opt)
[![PyPI version](https://img.shields.io/pypi/v/MatchCake-Opt)](https://pypi.org/project/MatchCake-Opt)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

![Tests Workflow](https://github.com/MatchCake/MatchCake-Opt/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/MatchCake-Opt/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/MatchCake-Opt/actions/workflows/docs.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


# Description

This package is plugin of [MatchCake](https://github.com/MatchCake/MatchCake). The main goal of this package is
to implement optimization pipelines and torch-modules that are used in MatchCake's projects.

## Installation

### For users

With `python` and `pip` installed,
```bash
pip install git+https://github.com/MatchCake/MatchCake-Opt
```

With `poetry` installed,
```bash
poetry add "git+https://github.com/MatchCake/MatchCake-Opt"
```


### For developers

With `python` and `pip` installed, run the following commands to install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry
python -m poetry install
```

With `poetry` installed, run the following commands to install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
poetry install
```

## Quick Usage Exemple

```python
from typing import Optional

import numpy as np
import torch
from ax import ChoiceParameterConfig, RangeParameterConfig

from matchcake_opt.datasets import *
from matchcake_opt.modules.classification_model import ClassificationModel
from matchcake_opt.tr_pipeline.automl_pipeline import AutoMLPipeline


class LinearNN(ClassificationModel):
    MODEL_NAME = "LinearNN"
    HP_CONFIGS = [
        RangeParameterConfig(
            name="learning_rate",
            parameter_type="float",
            bounds=(1e-5, 0.1),
        ),
        RangeParameterConfig(
            name="n_neurons",
            parameter_type="int",
            bounds=(4, 16),
        ),
    ]

    def __init__(
            self,
            input_shape: Optional[tuple[int, ...]],
            output_shape: Optional[tuple[int, ...]],
            learning_rate: float = 2e-4,
            n_neurons: int = 128,
            **kwargs,
    ):
        super().__init__(input_shape=input_shape, output_shape=output_shape, learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters("learning_rate", "n_neurons")
        self.nn = torch.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(n_neurons), 
            torch.nn.ReLU(), 
            torch.nn.LazyLinear(self.output_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    
    @property
    def output_size(self):
        return int(np.prod(self.output_shape))

datamodule = DataModule.from_dataset_name("Digits2D")
automl_pipeline = AutoMLPipeline(model_cls=LinearNN, datamodule=datamodule)
automl_pipeline.run()
lt_pipeline, metrics = automl_pipeline.run_best_pipeline()
print("⚡" * 20, "\nValidation Metrics:\n", metrics, "\n", "⚡" * 20)
test_metrics = lt_pipeline.run_test()
print("⚡" * 20, "\nTest Metrics:\n", test_metrics, "\n", "⚡" * 20)
```


## References


## License


## Acknowledgements


## Citation
```
```

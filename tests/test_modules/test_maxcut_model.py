from unittest.mock import MagicMock

import pytest
import torch
from torchmetrics import MetricCollection

from matchcake_opt import DataModule
from matchcake_opt.datasets.maxcut_dataset import MaxcutDataset
from matchcake_opt.modules.maxcut_model import MaxcutModel


class TestMaxcutModel:
    @pytest.fixture
    def model_instance(self):
        model = MaxcutModel((4,), (1,))
        model.parameters = MagicMock()
        param = torch.nn.Parameter(torch.Tensor([1]))
        model.parameters.return_value = [param]
        model.forward = MagicMock()
        model.forward.return_value = param * torch.ones(model.output_shape)
        model.sample = MagicMock()
        model.sample.return_value = torch.ones(model.input_shape)
        return model

    @pytest.fixture
    def sample_batch(self):
        dataset = MaxcutDataset(4, "regular", d=3)
        return dataset[0]

    def test_configure_metrics(self, model_instance):
        assert isinstance(model_instance.configure_metrics(), MetricCollection)

    def test_training_step(self, model_instance, sample_batch):
        model_instance.training_step(sample_batch, 0)
        model_instance.forward.assert_called_once()

    def test_validation_step(self, model_instance, sample_batch):
        model_instance.validation_step(sample_batch, 0)
        model_instance.forward.assert_called_once()

    def test_test_step(self, model_instance, sample_batch):
        model_instance.test_step(sample_batch, 0)
        model_instance.forward.assert_called_once()

    def test_predict(self, model_instance, sample_batch):
        model_instance.predict(sample_batch)
        model_instance.sample.assert_called_once()

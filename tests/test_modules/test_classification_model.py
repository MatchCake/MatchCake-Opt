from unittest.mock import MagicMock

import pytest
import torch
from torchmetrics import MetricCollection

from matchcake_opt.modules.classification_model import ClassificationModel


class TestClassificationModel:
    @pytest.fixture
    def model_instance(self, monkeypatch, sample_batch):
        monkeypatch.setattr("matchcake_opt.modules.base_model.BaseModel.configure_metrics", MagicMock())
        model = ClassificationModel((8,), (2,))
        model.parameters = MagicMock()
        param = torch.nn.Parameter(torch.Tensor([1]))
        model.parameters.return_value = [param]
        model.forward = MagicMock()
        model.forward.return_value = param * torch.ones((sample_batch[0].shape[0], *model.output_shape))
        return model

    @pytest.fixture
    def sample_batch(self):
        x = torch.zeros(3, 8)
        y = torch.zeros(3).long()
        return x, y

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
        model_instance.predict(sample_batch[0])
        model_instance.forward.assert_called_once()

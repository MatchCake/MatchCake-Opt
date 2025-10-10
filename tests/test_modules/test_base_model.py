import argparse
from unittest.mock import MagicMock

import pytest
import torch

from matchcake_opt.modules.base_model import BaseModel


class TestBaseModel:
    @pytest.fixture
    def model_instance(self, monkeypatch, sample_batch):
        monkeypatch.setattr("matchcake_opt.modules.base_model.BaseModel.configure_metrics", MagicMock())
        model = BaseModel((8,), (1,))
        model.train_metrics = MagicMock()
        model.val_metrics = MagicMock()
        model.test_metrics = MagicMock()
        model.parameters = MagicMock()
        param = torch.nn.Parameter(torch.Tensor([1]))
        model.parameters.return_value = [param]
        model.forward = MagicMock()
        model.forward.return_value = param * torch.ones((sample_batch[0].shape[0], *model.output_shape))
        return model

    @pytest.fixture
    def sample_batch(self):
        x = torch.zeros(3, 8)
        y = torch.zeros(3)
        return x, y

    def test_add_model_specific_args(self):
        parser = BaseModel.add_model_specific_args()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_configure_optimizers(self, model_instance):
        assert isinstance(model_instance.configure_optimizers(), torch.optim.Optimizer)

    def test_on_train_epoch_start(self, model_instance):
        model_instance.on_train_epoch_start()
        model_instance.train_metrics.reset.assert_called_once()

    def test_on_train_epoch_end(self, model_instance):
        model_instance.on_train_epoch_end()
        model_instance.train_metrics.compute.assert_called_once()

    def test_on_validation_epoch_start(self, model_instance):
        model_instance.on_validation_epoch_start()
        model_instance.val_metrics.reset.assert_called_once()

    def test_on_validation_epoch_end(self, model_instance):
        model_instance.on_validation_epoch_end()
        model_instance.val_metrics.compute.assert_called_once()

    def test_on_test_epoch_start(self, model_instance):
        model_instance.on_test_epoch_start()
        model_instance.test_metrics.reset.assert_called_once()

    def test_on_test_epoch_end(self, model_instance):
        model_instance.on_test_epoch_end()
        model_instance.test_metrics.compute.assert_called_once()

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

    def test_optimizer(self, model_instance):
        assert model_instance.optimizer == model_instance.DEFAULT_OPTIMIZER

    def test_learning_rate(self, model_instance):
        assert model_instance.learning_rate == model_instance.DEFAULT_LEARNING_RATE

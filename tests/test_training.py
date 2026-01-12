import pytest
import os.path
import importlib
import types
import torch
from src.mnist_project.train import train

@pytest.mark.skipif(
    not os.path.exists("src/mnist_project/data/"),
    reason="Data files not found",
)


def test_training_logs_metrics(monkeypatch):
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx): return torch.randn(1, 28, 28), torch.tensor(0)

    monkeypatch.setattr(train, "corrupt_mnist", lambda *_: (DummyDataset(), DummyDataset()))

    logged = []
    monkeypatch.setattr(
        train,
        "wandb",
        types.SimpleNamespace(
            init=lambda **_: None,
            log=lambda d: logged.append(d),
            Image=lambda *_: None,
            finish=lambda: None,
        ),
    )

    train.train(lr=1e-3, batch_size=1, epochs=1)

    assert logged
    assert "train_loss" in logged[0]



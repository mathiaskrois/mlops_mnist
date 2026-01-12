from wandb.integration.torch.wandb_torch import torch
from src.mnist_project.model import MyAwesomeModel
import pytest
@pytest.mark.parametrize("batch_size", [32, 128])

def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 32, 128)
    y = model(x)
    assert y.shape == (batch_size, 10)

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))

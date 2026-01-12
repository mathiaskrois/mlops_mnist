import torch
import typer
from pathlib import Path

from model import MyAwesomeModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_checkpoint: str, test_images_path: str, test_target_path: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"{model_checkpoint=}")
    print(f"{test_images_path=}")
    print(f"{test_target_path=}")

    # Always load checkpoints onto CPU (works even if saved from MPS/CUDA)
    state_dict = torch.load(model_checkpoint, map_location="cpu")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    test_images = torch.load(test_images_path, map_location="cpu")
    test_target = torch.load(test_target_path, map_location="cpu")

    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    correct, total = 0, 0
    with torch.inference_mode():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).sum().item()
            total += target.size(0)

    print(f"Test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    typer.run(evaluate)

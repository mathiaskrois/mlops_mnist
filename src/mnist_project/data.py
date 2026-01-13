import torch
import typer
from pathlib import Path
import matplotlib.pyplot as plt  # only needed for plotting
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, processed / "train_images.pt")
    torch.save(train_target, processed / "train_target.pt")
    torch.save(test_images, processed / "test_images.pt")
    torch.save(test_target, processed / "test_target.pt")


def corrupt_mnist(processed_dir: str = "data/processed") -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    p = Path(processed_dir)

    train_images = torch.load(p / "train_images.pt", map_location="cpu")
    train_target = torch.load(p / "train_target.pt", map_location="cpu")
    test_images = torch.load(p / "test_images.pt", map_location="cpu")
    test_target = torch.load(p / "test_target.pt", map_location="cpu")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    typer.run(preprocess_data)
    train_set, test_set = corrupt_mnist()
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])

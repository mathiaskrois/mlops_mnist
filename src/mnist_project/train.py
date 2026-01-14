import matplotlib.pyplot as plt
import torch
import typer
import wandb
from data import corrupt_mnist
from model import MyAwesomeModel
from sklearn.metrics import RocCurveDisplay
from torchvision.utils import make_grid

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        preds, targets = [], []

        for i, (img, target) in enumerate(train_dataloader):
            img = img.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

            wandb.log(
                {
                    "epoch": epoch,
                    "iter": i,
                    "train_loss": loss.item(),
                    "train_accuracy": accuracy,
                }
            )

            # Store predictions on CPU ONLY
            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                # -------- Image logging (CPU ONLY) --------
                img_cpu = img[:5].detach().cpu()

                grid = make_grid(img_cpu, nrow=5, normalize=True)
                wandb.log(
                    {
                        "input_grid": wandb.Image(
                            grid, caption=f"Epoch {epoch} – inputs"
                        )
                    }
                )

                # -------- Gradient histogram (CPU ONLY) --------
                grads = torch.cat(
                    [
                        p.grad.detach().cpu().flatten()
                        for p in model.parameters()
                        if p.grad is not None
                    ],
                    dim=0,
                )

                wandb.log({"gradients": wandb.Histogram(grads.numpy())})

                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")

        # -------- ROC curves (CPU tensors only) --------
        preds_epoch = torch.cat(preds, dim=0)      # [N, 10]
        targets_epoch = torch.cat(targets, dim=0)  # [N]

        plt.figure(figsize=(8, 6))
        for class_id in range(10):
            one_hot = (targets_epoch == class_id).int()

            RocCurveDisplay.from_predictions(
                one_hot.numpy(),
                preds_epoch[:, class_id].numpy(),
                name=f"class {class_id}",
                plot_chance_level=(class_id == 2),
            )

        plt.title(f"ROC curves (epoch {epoch})")
        plt.tight_layout()

        wandb.log({"roc": wandb.Image(plt.gcf())})
        plt.close()

    wandb.finish()


if __name__ == "__main__":
    typer.run(train)

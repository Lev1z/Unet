import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from skimage.metrics import structural_similarity
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def np_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def draw_scene(seed: int, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    bg = np.stack(
        [
            0.18 + 0.18 * grid_x,
            0.20 + 0.14 * grid_y,
            0.24 + 0.10 * (1 - grid_x),
        ],
        axis=-1,
    )
    bg += rng.normal(0, 0.025, bg.shape).astype(np.float32)
    bg = np.clip(bg, 0, 1)

    image = Image.fromarray((bg * 255).astype(np.uint8), mode="RGB")
    mask = Image.new("L", (size, size), 0)
    image_draw = ImageDraw.Draw(image, "RGBA")
    mask_draw = ImageDraw.Draw(mask)

    for _ in range(rng.integers(2, 6)):
        shape_type = rng.choice(["ellipse", "rect", "triangle"])
        w = int(rng.integers(size // 7, size // 3))
        h = int(rng.integers(size // 7, size // 3))
        left = int(rng.integers(1, max(2, size - w - 1)))
        top = int(rng.integers(1, max(2, size - h - 1)))
        box = [left, top, left + w, top + h]
        color = tuple(int(v) for v in rng.integers(70, 245, size=3)) + (230,)

        if shape_type == "ellipse":
            image_draw.ellipse(box, fill=color)
            mask_draw.ellipse(box, fill=255)
        elif shape_type == "rect":
            image_draw.rounded_rectangle(box, radius=int(rng.integers(0, 5)), fill=color)
            mask_draw.rounded_rectangle(box, radius=int(rng.integers(0, 5)), fill=255)
        else:
            points = [
                (left + w // 2, top),
                (left, top + h),
                (left + w, top + h),
            ]
            image_draw.polygon(points, fill=color)
            mask_draw.polygon(points, fill=255)

    image_np = np.asarray(image).astype(np.float32) / 255.0
    mask_np = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]
    return image_np, mask_np


def degrade_image(clean: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100_000)
    size = clean.shape[0]
    img = Image.fromarray((clean * 255).astype(np.uint8), mode="RGB")
    down = int(rng.integers(max(16, size // 3), max(17, size // 2 + 1)))
    img = img.resize((down, down), Image.Resampling.BICUBIC).resize((size, size), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.6, 1.5))))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr += rng.normal(0, float(rng.uniform(0.025, 0.06)), arr.shape).astype(np.float32)
    return np.clip(arr, 0, 1)


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, count: int, size: int, seed_offset: int):
        self.samples = []
        for idx in range(count):
            image, mask = draw_scene(seed_offset + idx, size=size)
            self.samples.append((np_to_tensor(image), torch.from_numpy(mask.transpose(2, 0, 1)).float()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class SyntheticRestorationDataset(Dataset):
    def __init__(self, count: int, size: int, seed_offset: int):
        self.samples = []
        for idx in range(count):
            clean, _ = draw_scene(seed_offset + idx, size=size)
            degraded = degrade_image(clean, seed_offset + idx)
            self.samples.append((np_to_tensor(degraded), np_to_tensor(clean)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, out_channels: int, base: int = 12):
        super().__init__()
        self.enc1 = DoubleConv(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class FCNSmall(nn.Module):
    def __init__(self, out_channels: int, base: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            DoubleConv(3, base),
            nn.MaxPool2d(2),
            DoubleConv(base, base * 2),
            nn.MaxPool2d(2),
            DoubleConv(base * 2, base * 4),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainRestorationCNN(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualUNetRestoration(nn.Module):
    outputs_image = True

    def __init__(self, base: int = 16, residual_scale: float = 0.5):
        super().__init__()
        self.core = UNetSmall(out_channels=3, base=base)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.core(x)) * self.residual_scale
        return (x + residual).clamp(0, 1)


def predict_restoration(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if getattr(model, "outputs_image", False):
        return outputs.clamp(0, 1)
    return torch.sigmoid(outputs).clamp(0, 1)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    task: str,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in tqdm(loader, desc=f"{task} epoch {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if task == "segmentation":
                probs = torch.sigmoid(outputs)
                intersection = (probs * targets).sum(dim=(1, 2, 3))
                dice = (2 * intersection + 1e-6) / (
                    probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
                )
                loss = 0.5 * bce_loss(outputs, targets) + (1 - dice.mean())
            else:
                preds = outputs.clamp(0, 1) if getattr(model, "outputs_image", False) else torch.sigmoid(outputs)
                loss = mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                if task == "segmentation":
                    probs = torch.sigmoid(outputs)
                    intersection = (probs * targets).sum(dim=(1, 2, 3))
                    dice = (2 * intersection + 1e-6) / (
                        probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
                    )
                    loss = 0.5 * bce_loss(outputs, targets) + (1 - dice.mean())
                else:
                    preds = outputs.clamp(0, 1) if getattr(model, "outputs_image", False) else torch.sigmoid(outputs)
                    loss = mse_loss(preds, targets)
                val_loss += float(loss.item()) * inputs.size(0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(loader.dataset),
                "val_loss": val_loss / len(val_loader.dataset),
            }
        )
    return history


def evaluate_segmentation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ious, dices, pixel_accs = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            probs = torch.sigmoid(model(inputs))
            preds = (probs > 0.5).float()
            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
            pred_sum = preds.sum(dim=(1, 2, 3))
            target_sum = targets.sum(dim=(1, 2, 3))
            ious.extend(((intersection + 1e-6) / (union + 1e-6)).cpu().numpy().tolist())
            dices.extend(((2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)).cpu().numpy().tolist())
            pixel_accs.extend((preds == targets).float().mean(dim=(1, 2, 3)).cpu().numpy().tolist())
    return {
        "mean_iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "pixel_accuracy": float(np.mean(pixel_accs)),
    }


def evaluate_restoration(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mses, psnrs, ssims = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = predict_restoration(model, inputs)
            for pred, target in zip(preds.cpu(), targets.cpu()):
                pred_np = pred.numpy().transpose(1, 2, 0)
                target_np = target.numpy().transpose(1, 2, 0)
                mse = float(np.mean((pred_np - target_np) ** 2))
                mses.append(mse)
                psnrs.append(float(20 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))))
                ssims.append(
                    float(
                        structural_similarity(
                            target_np,
                            pred_np,
                            data_range=1.0,
                            channel_axis=2,
                        )
                    )
                )
    return {
        "mse": float(np.mean(mses)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def evaluate_restoration_input(loader: DataLoader) -> dict[str, float]:
    mses, psnrs, ssims = [], [], []
    for inputs, targets in loader:
        for pred, target in zip(inputs, targets):
            pred_np = pred.numpy().transpose(1, 2, 0)
            target_np = target.numpy().transpose(1, 2, 0)
            mse = float(np.mean((pred_np - target_np) ** 2))
            mses.append(mse)
            psnrs.append(float(20 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))))
            ssims.append(
                float(
                    structural_similarity(
                        target_np,
                        pred_np,
                        data_range=1.0,
                        channel_axis=2,
                    )
                )
            )
    return {
        "mse": float(np.mean(mses)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    return arr.transpose(1, 2, 0)


def save_segmentation_examples(
    models: dict[str, nn.Module],
    dataset: Dataset,
    device: torch.device,
    out_path: Path,
    count: int = 4,
) -> None:
    fig, axes = plt.subplots(count, 4, figsize=(10, 2.6 * count))
    for row in range(count):
        image, mask = dataset[row]
        with torch.no_grad():
            preds = {
                name: (torch.sigmoid(model(image.unsqueeze(0).to(device))) > 0.5).float().squeeze(0).cpu()
                for name, model in models.items()
            }
        axes[row, 0].imshow(tensor_to_image(image))
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(mask.squeeze(0), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(preds["U-Net"].squeeze(0), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("U-Net")
        axes[row, 3].imshow(preds["FCN"].squeeze(0), cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("FCN")
        for col in range(4):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_restoration_examples(
    models: dict[str, nn.Module],
    dataset: Dataset,
    device: torch.device,
    out_path: Path,
    count: int = 4,
) -> None:
    fig, axes = plt.subplots(count, 4, figsize=(10, 2.6 * count))
    for row in range(count):
        degraded, clean = dataset[row]
        with torch.no_grad():
            preds = {
                name: predict_restoration(model, degraded.unsqueeze(0).to(device)).squeeze(0).cpu()
                for name, model in models.items()
            }
        axes[row, 0].imshow(tensor_to_image(degraded))
        axes[row, 0].set_title("Degraded")
        axes[row, 1].imshow(tensor_to_image(clean))
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(tensor_to_image(preds["Residual U-Net"]))
        axes[row, 2].set_title("Residual U-Net")
        axes[row, 3].imshow(tensor_to_image(preds["Plain CNN"]))
        axes[row, 3].set_title("Plain CNN")
        for col in range(4):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_history_plot(histories: dict[str, list[dict[str, float]]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        ax.plot(epochs, val_loss, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="U-Net segmentation and restoration experiments")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_train = SyntheticSegmentationDataset(args.train_count, args.size, 0)
    seg_val = SyntheticSegmentationDataset(args.val_count, args.size, 10_000)
    restore_train = SyntheticRestorationDataset(args.train_count, args.size, 20_000)
    restore_val = SyntheticRestorationDataset(args.val_count, args.size, 30_000)

    seg_train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True)
    seg_val_loader = DataLoader(seg_val, batch_size=args.batch_size)
    restore_train_loader = DataLoader(restore_train, batch_size=args.batch_size, shuffle=True)
    restore_val_loader = DataLoader(restore_val, batch_size=args.batch_size)

    results = {
        "config": vars(args) | {"output_dir": str(args.output_dir), "device": str(device)},
        "segmentation": {},
        "restoration": {},
    }

    seg_models = {
        "U-Net": UNetSmall(out_channels=1),
        "FCN": FCNSmall(out_channels=1, base=8),
    }
    seg_histories = {}
    for name, model in seg_models.items():
        history = train_model(model, seg_train_loader, seg_val_loader, args.epochs, 1e-3, "segmentation", device)
        seg_histories[name] = history
        results["segmentation"][name] = evaluate_segmentation(model, seg_val_loader, device) | {
            "parameters": count_parameters(model),
            "history": history,
        }

    restore_models = {
        "Residual U-Net": ResidualUNetRestoration(),
        "Plain CNN": PlainRestorationCNN(),
    }
    results["restoration"]["Degraded input"] = evaluate_restoration_input(restore_val_loader)
    restore_histories = {}
    for name, model in restore_models.items():
        history = train_model(model, restore_train_loader, restore_val_loader, args.epochs, 1e-3, "restoration", device)
        restore_histories[name] = history
        results["restoration"][name] = evaluate_restoration(model, restore_val_loader, device) | {
            "parameters": count_parameters(model),
            "history": history,
        }

    save_segmentation_examples(seg_models, seg_val, device, figures_dir / "segmentation_examples.png")
    save_restoration_examples(restore_models, restore_val, device, figures_dir / "restoration_examples.png")
    save_history_plot(seg_histories, figures_dir / "segmentation_loss.png", "Segmentation Validation Loss")
    save_history_plot(restore_histories, figures_dir / "restoration_loss.png", "Restoration Validation Loss")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

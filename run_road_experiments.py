import argparse
import copy
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


CLASS_NAMES = ["background", "road", "vehicle", "obstacle"]
CLASS_COLORS = np.array(
    [
        [40, 110, 180],
        [90, 90, 90],
        [220, 60, 50],
        [245, 180, 40],
    ],
    dtype=np.uint8,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def np_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def draw_road_scene(seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), (55, 135, 200))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image, "RGBA")
    mask_draw = ImageDraw.Draw(mask)

    horizon = int(size * rng.uniform(0.38, 0.48))
    sky_color = tuple(int(v) for v in rng.integers([45, 105, 165], [95, 165, 225]))
    draw.rectangle([0, 0, size, horizon], fill=sky_color + (255,))

    # Background buildings/sidewalk regions.
    for _ in range(int(rng.integers(3, 7))):
        w = int(rng.integers(size // 9, size // 4))
        h = int(rng.integers(size // 7, size // 3))
        x0 = int(rng.integers(0, max(1, size - w)))
        y0 = int(rng.integers(max(1, horizon - h // 2), max(2, horizon + size // 10)))
        color = tuple(int(v) for v in rng.integers(70, 150, size=3))
        draw.rectangle([x0, y0, x0 + w, min(size, y0 + h)], fill=color + (210,))

    road_top_left = int(size * rng.uniform(0.35, 0.45))
    road_top_right = int(size * rng.uniform(0.55, 0.65))
    road_poly = [(road_top_left, horizon), (road_top_right, horizon), (size, size), (0, size)]
    road_color = tuple(int(v) for v in rng.integers([55, 55, 55], [115, 115, 115]))
    draw.polygon(road_poly, fill=road_color + (255,))
    mask_draw.polygon(road_poly, fill=1)

    # Lane markings.
    lane_center = size // 2 + int(rng.integers(-size // 18, size // 18 + 1))
    for i in range(6):
        y0 = int(horizon + (size - horizon) * (i / 6))
        y1 = int(horizon + (size - horizon) * ((i + 0.45) / 6))
        width = max(1, int(size * (0.01 + 0.015 * i / 6)))
        draw.line([(lane_center, y0), (lane_center, y1)], fill=(245, 230, 170, 200), width=width)

    # Vehicles.
    for _ in range(int(rng.integers(2, 5))):
        y = int(rng.integers(horizon + size // 12, size - size // 8))
        scale = (y - horizon) / max(1, size - horizon)
        w = int(size * rng.uniform(0.08, 0.18) * (0.55 + scale))
        h = int(w * rng.uniform(0.55, 0.85))
        x = int(rng.integers(max(1, size // 8), max(size // 8 + 1, size - w - size // 8)))
        box = [x, y, x + w, min(size - 1, y + h)]
        color = tuple(int(v) for v in rng.integers([120, 30, 30], [245, 220, 220]))
        draw.rounded_rectangle(box, radius=max(1, w // 8), fill=color + (245,))
        mask_draw.rounded_rectangle(box, radius=max(1, w // 8), fill=2)
        windshield = [x + w // 5, y + h // 8, x + 4 * w // 5, y + h // 2]
        draw.rectangle(windshield, fill=(40, 90, 130, 180))
        for wx in (x + w // 5, x + 4 * w // 5):
            draw.ellipse([wx - w // 10, y + h - h // 5, wx + w // 10, y + h], fill=(25, 25, 25, 255))

    # Obstacles: cones, barriers, boxes.
    for _ in range(int(rng.integers(2, 5))):
        y = int(rng.integers(horizon + size // 8, size - size // 10))
        scale = (y - horizon) / max(1, size - horizon)
        w = int(size * rng.uniform(0.035, 0.075) * (0.7 + scale))
        h = int(w * rng.uniform(1.0, 1.6))
        x = int(rng.integers(size // 12, max(size // 12 + 1, size - w - size // 12)))
        if rng.random() < 0.6:
            pts = [(x + w // 2, y), (x, y + h), (x + w, y + h)]
            draw.polygon(pts, fill=(245, 150, 25, 245))
            mask_draw.polygon(pts, fill=3)
            draw.line([(x + w // 4, y + h // 2), (x + 3 * w // 4, y + h // 2)], fill=(250, 250, 250, 230), width=1)
        else:
            box = [x, y, x + w * 2, y + h]
            draw.rectangle(box, fill=(230, 170, 40, 240))
            mask_draw.rectangle(box, fill=3)

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr += rng.normal(0, 0.025, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    return arr, np.asarray(mask).astype(np.int64)


def degrade_image(clean: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100_000)
    size = clean.shape[0]
    img = Image.fromarray((clean * 255).astype(np.uint8), mode="RGB")
    down = int(rng.integers(max(24, size // 2), max(25, int(size * 0.72))))
    img = img.resize((down, down), Image.Resampling.BICUBIC).resize((size, size), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.45, 1.1))))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr += rng.normal(0, float(rng.uniform(0.01, 0.028)), arr.shape).astype(np.float32)
    return np.clip(arr, 0, 1)


class RoadSegmentationDataset(Dataset):
    def __init__(self, count: int, size: int, seed_offset: int):
        self.samples = []
        for idx in range(count):
            image, mask = draw_road_scene(seed_offset + idx, size)
            self.samples.append((np_to_tensor(image), torch.from_numpy(mask).long()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class RoadRestorationDataset(Dataset):
    def __init__(self, count: int, size: int, seed_offset: int, img_dir: Path):
        self.samples = []
        sources = load_img_sources(img_dir, size)
        self.source_type = "img" if sources else "generated_road_placeholder"
        if not sources:
            sources = [draw_road_scene(90_000 + idx, size)[0] for idx in range(3)]
        for idx in range(count):
            clean = sources[idx % len(sources)].copy()
            degraded = degrade_image(clean, seed_offset + idx)
            self.samples.append((np_to_tensor(degraded), np_to_tensor(clean)))
        self.source_count = len(sources)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def load_img_sources(img_dir: Path, size: int) -> list[np.ndarray]:
    if not img_dir.exists():
        return []
    images = []
    for path in sorted(img_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        images.append(np.asarray(image).astype(np.float32) / 255.0)
    return images[:3]


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
    def __init__(self, out_channels: int, base: int = 16):
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
    def __init__(self, out_channels: int, base: int = 10):
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


class RestorationUNet(nn.Module):
    outputs_image = True

    def __init__(self, base: int = 16, residual_scale: float = 0.35):
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_segmentation(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"segmentation epoch {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * inputs.size(0)
        val_total = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_total += float(criterion(model(inputs), targets).item()) * inputs.size(0)
        val_loss = val_total / len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset), "val_loss": val_loss})
    model.load_state_dict(best_state)
    return history


def train_restoration(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"restoration epoch {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = predict_restoration(model, inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * inputs.size(0)
        val_total = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_total += float(criterion(predict_restoration(model, inputs), targets).item()) * inputs.size(0)
        val_loss = val_total / len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset), "val_loss": val_loss})
    model.load_state_dict(best_state)
    return history


def evaluate_segmentation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    intersections = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    unions = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    pixel_correct = 0
    pixel_total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            targets_np = targets.numpy()
            pixel_correct += int((preds == targets_np).sum())
            pixel_total += int(targets_np.size)
            for cls in range(len(CLASS_NAMES)):
                pred_cls = preds == cls
                target_cls = targets_np == cls
                intersections[cls] += np.logical_and(pred_cls, target_cls).sum()
                unions[cls] += np.logical_or(pred_cls, target_cls).sum()
    ious = (intersections + 1e-6) / (unions + 1e-6)
    return {
        "mean_iou_all": float(np.mean(ious)),
        "mean_iou_foreground": float(np.mean(ious[1:])),
        "pixel_accuracy": float(pixel_correct / pixel_total),
        "class_iou": {name: float(ious[idx]) for idx, name in enumerate(CLASS_NAMES)},
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
                ssims.append(float(structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)))
    return {"mse": float(np.mean(mses)), "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def evaluate_restoration_input(loader: DataLoader) -> dict[str, float]:
    mses, psnrs, ssims = [], [], []
    for inputs, targets in loader:
        for pred, target in zip(inputs, targets):
            pred_np = pred.numpy().transpose(1, 2, 0)
            target_np = target.numpy().transpose(1, 2, 0)
            mse = float(np.mean((pred_np - target_np) ** 2))
            mses.append(mse)
            psnrs.append(float(20 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))))
            ssims.append(float(structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)))
    return {"mse": float(np.mean(mses)), "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


def save_segmentation_examples(models: dict[str, nn.Module], dataset: Dataset, device: torch.device, out_path: Path, count: int = 4) -> None:
    fig, axes = plt.subplots(count, 4, figsize=(10, 2.6 * count))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    for row, idx in enumerate(indices):
        image, mask = dataset[int(idx)]
        with torch.no_grad():
            preds = {name: model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy() for name, model in models.items()}
        axes[row, 0].imshow(tensor_to_image(image))
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(mask_to_rgb(mask.numpy()))
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(mask_to_rgb(preds["U-Net"]))
        axes[row, 2].set_title("U-Net")
        axes[row, 3].imshow(mask_to_rgb(preds["FCN"]))
        axes[row, 3].set_title("FCN")
        for col in range(4):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_restoration_examples(models: dict[str, nn.Module], dataset: Dataset, device: torch.device, out_path: Path, count: int = 3) -> None:
    fig, axes = plt.subplots(count, 4, figsize=(10, 2.8 * count))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    for row, idx in enumerate(indices):
        degraded, clean = dataset[int(idx)]
        with torch.no_grad():
            preds = {name: predict_restoration(model, degraded.unsqueeze(0).to(device)).squeeze(0).cpu() for name, model in models.items()}
        axes[row, 0].imshow(tensor_to_image(degraded))
        axes[row, 0].set_title("Degraded")
        axes[row, 1].imshow(tensor_to_image(clean))
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(tensor_to_image(preds["Restoration U-Net"]))
        axes[row, 2].set_title("Restoration U-Net")
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
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Road-scene segmentation and restoration experiments")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--train-count", type=int, default=768)
    parser.add_argument("--val-count", type=int, default=192)
    parser.add_argument("--restore-train-count", type=int, default=96)
    parser.add_argument("--restore-val-count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-dir", type=Path, default=Path("img"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_road"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_train = RoadSegmentationDataset(args.train_count, args.size, 0)
    seg_val = RoadSegmentationDataset(args.val_count, args.size, 10_000)
    restore_train = RoadRestorationDataset(args.restore_train_count, args.size, 20_000, args.img_dir)
    restore_val = RoadRestorationDataset(args.restore_val_count, args.size, 30_000, args.img_dir)

    seg_train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True)
    seg_val_loader = DataLoader(seg_val, batch_size=args.batch_size)
    restore_train_loader = DataLoader(restore_train, batch_size=args.batch_size, shuffle=True)
    restore_val_loader = DataLoader(restore_val, batch_size=args.batch_size)

    results: dict[str, object] = {
        "config": vars(args) | {
            "img_dir": str(args.img_dir),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "restoration_source_images": restore_train.source_count,
            "restoration_source_type": restore_train.source_type,
            "classes": CLASS_NAMES,
        },
        "segmentation": {},
        "restoration": {"Degraded input": evaluate_restoration_input(restore_val_loader)},
    }

    seg_models = {"U-Net": UNetSmall(out_channels=len(CLASS_NAMES)), "FCN": FCNSmall(out_channels=len(CLASS_NAMES))}
    seg_histories = {}
    for name, model in seg_models.items():
        history = train_segmentation(model, seg_train_loader, seg_val_loader, args.epochs, device)
        seg_histories[name] = history
        results["segmentation"][name] = evaluate_segmentation(model, seg_val_loader, device) | {
            "parameters": count_parameters(model),
            "history": history,
        }

    restore_models = {"Restoration U-Net": RestorationUNet(), "Plain CNN": PlainRestorationCNN()}
    restore_histories = {}
    for name, model in restore_models.items():
        history = train_restoration(model, restore_train_loader, restore_val_loader, max(30, args.epochs), device)
        restore_histories[name] = history
        results["restoration"][name] = evaluate_restoration(model, restore_val_loader, device) | {
            "parameters": count_parameters(model),
            "history": history,
        }

    save_segmentation_examples(seg_models, seg_val, device, figures_dir / "segmentation_examples.png")
    save_restoration_examples(restore_models, restore_val, device, figures_dir / "restoration_examples.png")
    save_history_plot(seg_histories, figures_dir / "segmentation_loss.png", "Road Segmentation Validation Loss")
    save_history_plot(restore_histories, figures_dir / "restoration_loss.png", "Restoration Validation Loss")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

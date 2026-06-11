import argparse
import copy
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
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

CAMVID_URLS = {
    "camvid_tiny": "https://s3.amazonaws.com/fast-ai-sample/camvid_tiny.tgz",
    "camvid": "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz",
}

CAMVID_ROAD_CLASSES = {"Road", "RoadShoulder", "Sidewalk", "LaneMkgsDriv", "LaneMkgsNonDriv"}
CAMVID_VEHICLE_CLASSES = {"Car", "SUVPickupTruck", "Truck_Bus", "Train", "MotorcycleScooter"}
CAMVID_OBSTACLE_CLASSES = {
    "Animal",
    "Bicyclist",
    "CartLuggagePram",
    "Child",
    "Column_Pole",
    "Fence",
    "OtherMoving",
    "ParkingBlock",
    "Pedestrian",
    "SignSymbol",
    "TrafficCone",
    "TrafficLight",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def np_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Refusing to extract unsafe path: {member.name}")
        tar.extractall(destination)


def find_camvid_root(search_root: Path) -> Path:
    candidates = []
    for path in search_root.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "images").is_dir() and (path / "labels").is_dir() and (path / "codes.txt").is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not find CamVid images/labels/codes.txt under {search_root}")
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def prepare_camvid_dataset(dataset_name: str, data_dir: Path, download: bool) -> Path:
    dataset_dir = data_dir / dataset_name
    try:
        return find_camvid_root(dataset_dir)
    except FileNotFoundError:
        pass
    if not download:
        raise FileNotFoundError(f"{dataset_name} is not available under {dataset_dir}; rerun with --download")

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / f"{dataset_name}.tgz"
    if not archive_path.exists():
        print(f"Downloading {dataset_name} from {CAMVID_URLS[dataset_name]}")
        urllib.request.urlretrieve(CAMVID_URLS[dataset_name], archive_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    safe_extract_tar(archive_path, dataset_dir)
    return find_camvid_root(dataset_dir)


def load_camvid_mapping(codes_path: Path) -> np.ndarray:
    codes = [line.strip() for line in codes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = np.zeros(len(codes), dtype=np.uint8)
    for idx, name in enumerate(codes):
        if name in CAMVID_ROAD_CLASSES:
            mapping[idx] = 1
        elif name in CAMVID_VEHICLE_CLASSES:
            mapping[idx] = 2
        elif name in CAMVID_OBSTACLE_CLASSES:
            mapping[idx] = 3
    return mapping


def pair_camvid_files(root: Path) -> list[tuple[Path, Path]]:
    image_paths = sorted((root / "images").glob("*.png"))
    pairs = []
    for image_path in image_paths:
        label_path = root / "labels" / f"{image_path.stem}_P.png"
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"No CamVid image/label pairs found under {root}")
    return pairs


def select_pairs(pairs: list[tuple[Path, Path]], count: int, split: str, seed: int) -> list[tuple[Path, Path]]:
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    val_count = max(1, min(len(shuffled) // 5, 128))
    if split == "val":
        base = shuffled[:val_count]
    else:
        base = shuffled[val_count:] or shuffled
    if count <= len(base):
        return base[:count]
    return [base[idx % len(base)] for idx in range(count)]


class CamVidSegmentationDataset(Dataset):
    def __init__(self, root: Path, count: int, size: int, split: str, seed: int):
        self.samples = []
        self.root = root
        mapping = load_camvid_mapping(root / "codes.txt")
        pairs = select_pairs(pair_camvid_files(root), count, split, seed)
        for image_path, label_path in pairs:
            image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            label = Image.open(label_path).resize((size, size), Image.Resampling.NEAREST)
            label_np = np.asarray(label, dtype=np.int64)
            label_np = np.where(label_np < len(mapping), mapping[label_np], 0).astype(np.int64)
            image_np = np.asarray(image).astype(np.float32) / 255.0
            self.samples.append((np_to_tensor(image_np), torch.from_numpy(label_np).long()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def load_camvid_sources(root: Path, size: int, limit: int = 3) -> list[np.ndarray]:
    sources = []
    for image_path, _ in pair_camvid_files(root)[:limit]:
        image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        sources.append(np.asarray(image).astype(np.float32) / 255.0)
    return sources


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


class VisualRestorationDataset(Dataset):
    def __init__(self, size: int, img_dir: Path, fallback_sources: list[np.ndarray] | None = None):
        self.samples = []
        sources = load_img_sources(img_dir, size)
        self.source_type = "img" if sources else "camvid_images"
        if not sources and fallback_sources:
            sources = fallback_sources
        if not sources:
            self.source_type = "generated_road_placeholder"
            sources = [draw_road_scene(90_000 + idx, size)[0] for idx in range(3)]
        for source in sources[:3]:
            enhanced = enhance_for_visual_restoration(source)
            self.samples.append((source, enhanced))
        self.source_count = len(sources)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.samples[idx]


def enhance_for_visual_restoration(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray((image * 255).astype(np.uint8), mode="RGB")
    scale = 3 if max(pil.size) < 180 else 2
    pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(pil)
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    arr = np.asarray(gray).astype(np.float32)
    light_mask = arr > 168
    mid_mask = (arr > 118) & (arr <= 168)
    arr[light_mask] = 255
    arr[mid_mask] = arr[mid_mask] + (255 - arr[mid_mask]) * 0.38
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    clean = Image.fromarray(arr, mode="L")
    clean = clean.filter(ImageFilter.MaxFilter(size=3))
    clean = clean.filter(ImageFilter.MinFilter(size=3))
    clean = clean.filter(ImageFilter.ModeFilter(size=3))
    clean = clean.filter(ImageFilter.MedianFilter(size=3))
    clean = Image.fromarray(remove_small_dark_components(np.asarray(clean), min_area=95, threshold=190), mode="L")
    clean = clean.filter(ImageFilter.UnsharpMask(radius=0.9, percent=125, threshold=5))
    clean = ImageEnhance.Contrast(clean).enhance(1.08)
    return np.asarray(clean.convert("RGB")).astype(np.float32) / 255.0


def remove_small_dark_components(image: np.ndarray, min_area: int, threshold: int = 95) -> np.ndarray:
    dark = image < threshold
    visited = np.zeros(dark.shape, dtype=bool)
    cleaned = image.copy()
    height, width = dark.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not dark[y, x]:
                continue
            stack = [(y, x)]
            component = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and dark[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in component]
            xs = [point[1] for point in component]
            span_y = max(ys) - min(ys) + 1
            span_x = max(xs) - min(xs) + 1
            if len(component) < min_area or (span_y <= 22 and span_x <= 22):
                for cy, cx in component:
                    cleaned[cy, cx] = 255
    return cleaned


def load_img_sources(img_dir: Path, size: int) -> list[np.ndarray]:
    if not img_dir.exists():
        return []
    images = []
    for path in sorted(img_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image = Image.open(path).convert("RGB")
        if max(image.size) < size:
            scale = int(np.ceil(size / max(image.size)))
            image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def compute_class_weights(dataset: Dataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for _, mask in dataset:
        counts += np.bincount(mask.numpy().ravel(), minlength=len(CLASS_NAMES))
    frequencies = counts / max(1.0, counts.sum())
    weights = 1.0 / np.sqrt(frequencies + 1e-6)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.35, 5.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_segmentation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    class_weights: torch.Tensor,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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


def save_visual_restoration_examples(dataset: VisualRestorationDataset, out_path: Path, count: int = 3) -> None:
    count = min(count, len(dataset))
    fig, axes = plt.subplots(count, 2, figsize=(7, 3.0 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    for row, idx in enumerate(indices):
        original, enhanced = dataset[int(idx)]
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Original")
        axes[row, 1].imshow(enhanced)
        axes[row, 1].set_title("Enhanced result")
        for col in range(2):
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
    parser = argparse.ArgumentParser(description="CamVid road-scene segmentation and visual image enhancement experiments")
    parser.add_argument("--seg-dataset", choices=["camvid_tiny", "camvid", "synthetic"], default="camvid_tiny")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-dir", type=Path, default=Path("img"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_camvid"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    camvid_root = None
    restoration_fallback_sources = None
    if args.seg_dataset == "synthetic":
        seg_train = RoadSegmentationDataset(args.train_count, args.size, 0)
        seg_val = RoadSegmentationDataset(args.val_count, args.size, 10_000)
        segmentation_source_type = "generated_road_scenes"
    else:
        try:
            camvid_root = prepare_camvid_dataset(args.seg_dataset, args.data_dir, args.download)
            seg_train = CamVidSegmentationDataset(camvid_root, args.train_count, args.size, "train", args.seed)
            seg_val = CamVidSegmentationDataset(camvid_root, args.val_count, args.size, "val", args.seed)
            restoration_fallback_sources = load_camvid_sources(camvid_root, args.size)
            segmentation_source_type = str(camvid_root)
        except Exception as exc:
            print(f"CamVid loading failed ({exc}); falling back to generated road scenes.")
            seg_train = RoadSegmentationDataset(args.train_count, args.size, 0)
            seg_val = RoadSegmentationDataset(args.val_count, args.size, 10_000)
            segmentation_source_type = "generated_road_scenes_fallback"

    visual_restore = VisualRestorationDataset(args.size, args.img_dir, restoration_fallback_sources)

    seg_train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True)
    seg_val_loader = DataLoader(seg_val, batch_size=args.batch_size)
    class_weights = compute_class_weights(seg_train, device)

    results: dict[str, object] = {
        "config": vars(args) | {
            "data_dir": str(args.data_dir),
            "img_dir": str(args.img_dir),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "segmentation_dataset": args.seg_dataset,
            "segmentation_source_type": segmentation_source_type,
            "camvid_root": str(camvid_root) if camvid_root else None,
            "restoration_mode": "visual_only_no_reference",
            "restoration_source_images": visual_restore.source_count,
            "restoration_source_type": visual_restore.source_type,
            "classes": CLASS_NAMES,
            "segmentation_class_weights": {name: float(class_weights[idx].cpu()) for idx, name in enumerate(CLASS_NAMES)},
        },
        "segmentation": {},
        "restoration": {
            "mode": "visual_only_no_reference",
            "note": "The img samples have no high-resolution ground truth, so no artificial degradation or full-reference metrics are used.",
        },
    }

    seg_models = {"U-Net": UNetSmall(out_channels=len(CLASS_NAMES)), "FCN": FCNSmall(out_channels=len(CLASS_NAMES))}
    seg_histories = {}
    for name, model in seg_models.items():
        history = train_segmentation(model, seg_train_loader, seg_val_loader, args.epochs, device, class_weights)
        seg_histories[name] = history
        results["segmentation"][name] = evaluate_segmentation(model, seg_val_loader, device) | {
            "parameters": count_parameters(model),
            "history": history,
        }

    save_segmentation_examples(seg_models, seg_val, device, figures_dir / "segmentation_examples.png")
    save_visual_restoration_examples(visual_restore, figures_dir / "restoration_examples.png")
    save_history_plot(seg_histories, figures_dir / "segmentation_loss.png", "Road Segmentation Validation Loss")
    stale_restoration_loss = figures_dir / "restoration_loss.png"
    if stale_restoration_loss.exists():
        stale_restoration_loss.unlink()

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

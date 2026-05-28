import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.int64)


def compute_iou(pred: np.ndarray, gt: np.ndarray, class_id: int):
    pred_c = pred == class_id
    gt_c = gt == class_id
    union = np.logical_or(pred_c, gt_c).sum()
    if union == 0:
        return None
    inter = np.logical_and(pred_c, gt_c).sum()
    return inter / union


def evaluate_split(pred_dir: Path, gt_dir: Path, classes):
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction PNGs found in {pred_dir}")

    intersections = {class_id: 0 for class_id in classes}
    unions = {class_id: 0 for class_id in classes}
    used = 0

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT mask for {pred_path.name}: {gt_path}")

        pred = read_mask(pred_path)
        gt = read_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {pred_path.name}: pred {pred.shape}, gt {gt.shape}")

        used += 1
        for class_id in classes:
            pred_c = pred == class_id
            gt_c = gt == class_id
            intersections[class_id] += np.logical_and(pred_c, gt_c).sum()
            unions[class_id] += np.logical_or(pred_c, gt_c).sum()

    per_class = {}
    for class_id in classes:
        per_class[class_id] = None if unions[class_id] == 0 else intersections[class_id] / unions[class_id]

    valid = [iou for iou in per_class.values() if iou is not None]
    miou = float(np.mean(valid)) if valid else float("nan")
    return used, miou, per_class


def parse_classes(value: str):
    classes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            classes.extend(range(int(start), int(end) + 1))
        else:
            classes.append(int(part))
    return classes


def main():
    parser = argparse.ArgumentParser(description="Evaluate mIoU for rendered object masks.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--splits", default="train,test", help="Comma-separated split names.")
    parser.add_argument("--classes", default="1-8", help="Classes to include, e.g. 1-8 or 0,1,2.")
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    root = Path(args.model_path)

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        render_dir = root / split / f"ours_{args.iteration}"
        pred_dir = render_dir / "objects_pred"
        gt_dir = render_dir / "gt_objects"
        used, miou, per_class = evaluate_split(pred_dir, gt_dir, classes)

        pcs = ", ".join(
            f"{class_id}:{'nan' if iou is None else f'{iou:.4f}'}"
            for class_id, iou in per_class.items()
        )
        print(f"{split}: files={used} mIoU={miou:.4f} classes=[{pcs}]")


if __name__ == "__main__":
    main()

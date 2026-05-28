import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = torch.outer(gauss, gauss)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11):
    channels = img1.shape[1]
    window = gaussian_window(window_size, channels=channels, device=img1.device)
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0)


def evaluate_split(render_dir, gt_dir):
    names = sorted(p.name for p in render_dir.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No PNG files found in {render_dir}")

    psnrs = []
    ssims = []
    mses = []
    maes = []

    for name in names:
        render = load_rgb(render_dir / name)
        gt = load_rgb(gt_dir / name)
        if render.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {name}: {render.shape} vs {gt.shape}")

        mse = torch.mean((render - gt) ** 2)
        mae = torch.mean(torch.abs(render - gt))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-12)))

        mses.append(mse.item())
        maes.append(mae.item())
        psnrs.append(psnr.item())
        ssims.append(ssim(render, gt).item())

    return {
        "files": len(names),
        "PSNR": sum(psnrs) / len(psnrs),
        "SSIM": sum(ssims) / len(ssims),
        "MSE": sum(mses) / len(mses),
        "MAE": sum(maes) / len(maes),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RGB metrics for rendered images.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.model_path)
    results = {}
    for split in args.splits:
        split_dir = root / split / f"ours_{args.iteration}"
        results[split] = evaluate_split(split_dir / "renders", split_dir / "gt")

    for split, metrics in results.items():
        values = " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
        )
        print(f"{split}: {values}")

    if args.output:
        output = Path(args.output)
        output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

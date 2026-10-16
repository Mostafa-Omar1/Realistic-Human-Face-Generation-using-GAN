"""Present the results: loss curves, progress over epochs, final generated faces.

Outputs (in outputs/):
    loss_curve.png         Generator vs Discriminator loss + D(x)/D(G(z)) per epoch
    progress.png           Same fixed noise rendered at several epochs (compare over time)
    training.gif           Animation of the fixed-noise samples across all epochs
    final_faces.png        A fresh batch of 64 faces from the trained Generator
    interpolation.png      Smooth walks between random latent vectors
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import save_image

from models import LATENT_DIM, Generator

OUT = "outputs"


def plot_losses(path):
    with open(path) as f:
        log = json.load(f)
    epochs, steps = log["epochs"], log["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    s, ld, lg = zip(*steps)
    axes[0].plot(s, ld, label="Discriminator", alpha=0.8)
    axes[0].plot(s, lg, label="Generator", alpha=0.8)
    axes[0].set(title="GAN losses during training", xlabel="iteration", ylabel="BCE loss")
    axes[0].legend()

    e = [h["epoch"] for h in epochs]
    axes[1].plot(e, [h["d_real"] for h in epochs], "o-", label="D(x)  real")
    axes[1].plot(e, [h["d_fake"] for h in epochs], "o-", label="D(G(z))  fake")
    axes[1].axhline(0.5, color="gray", ls="--", lw=1)
    axes[1].set(title="Discriminator confidence per epoch", xlabel="epoch", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_curve.png"), dpi=120)
    plt.close(fig)


def progress_figure(sample_files):
    picks = sorted({0, len(sample_files) // 4, len(sample_files) // 2,
                    3 * len(sample_files) // 4, len(sample_files) - 1})
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.4))
    for ax, i in zip(axes, picks):
        img = Image.open(sample_files[i])
        w = img.width // 2  # top-left 4x4 faces keep each panel readable
        ax.imshow(img.crop((0, 0, w, w)))
        ax.set_title(f"Epoch {i + 1}")
        ax.axis("off")
    fig.suptitle("Same noise vectors, generated at different epochs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "progress.png"), dpi=120)
    plt.close(fig)

    frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in sample_files]
    frames[0].save(os.path.join(OUT, "training.gif"), save_all=True,
                   append_images=frames[1:] + [frames[-1]] * 5, duration=300, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join("checkpoints", "generator.pt"))
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    if os.path.exists(os.path.join(OUT, "history.json")):
        plot_losses(os.path.join(OUT, "history.json"))
    sample_files = sorted(glob.glob(os.path.join(OUT, "samples", "epoch_*.png")))
    if sample_files:
        progress_figure(sample_files)

    G = Generator().to(device)
    G.load_state_dict(torch.load(args.weights, map_location=device))
    G.eval()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        faces = G(torch.randn(args.num, LATENT_DIM, 1, 1, device=device))
        save_image(faces, os.path.join(OUT, "final_faces.png"), nrow=8,
                   normalize=True, value_range=(-1, 1))

        # Linear interpolation between pairs of latent vectors
        rows, steps = 6, 10
        z0 = torch.randn(rows, LATENT_DIM, 1, 1, device=device)
        z1 = torch.randn(rows, LATENT_DIM, 1, 1, device=device)
        t = torch.linspace(0, 1, steps, device=device).view(1, steps, 1, 1, 1)
        z = (z0.unsqueeze(1) * (1 - t) + z1.unsqueeze(1) * t).view(-1, LATENT_DIM, 1, 1)
        save_image(G(z), os.path.join(OUT, "interpolation.png"), nrow=steps,
                   normalize=True, value_range=(-1, 1))
    print(f"Results written to {OUT}/")


if __name__ == "__main__":
    main()

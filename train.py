"""Train the DCGAN on preprocessed CelebA faces with adversarial learning.

Every epoch it logs Generator/Discriminator losses and D(x) / D(G(z)) scores,
saves a sample grid from a FIXED noise batch (so progress is comparable over
time) and writes a checkpoint that can be resumed with --resume.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from models import LATENT_DIM, Discriminator, Generator, weights_init

OUT = "outputs"
CKPT = "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "faces_64.npy"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)

    # Whole dataset as uint8 NHWC; kept on the GPU when possible (~800MB for 67k faces)
    data = torch.from_numpy(np.load(args.data)).permute(0, 3, 1, 2).contiguous()
    data = data.to(device)
    n = data.shape[0]
    print(f"Loaded {n} faces of size {tuple(data.shape[1:])} on {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    history = []
    step_log = []  # (global_step, loss_d, loss_g) every 50 steps for a detailed loss curve
    start_epoch = 0
    ckpt_path = os.path.join(CKPT, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        fixed_noise = ck["fixed_noise"]
        history = ck["history"]
        step_log = ck.get("step_log", [])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    steps_per_epoch = n // args.batch_size
    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        sums = dict(loss_d=0.0, loss_g=0.0, d_real=0.0, d_fake=0.0)

        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            real = data[idx].float().div_(127.5).sub_(1.0)  # normalize to [-1, 1]
            flip = torch.rand(real.size(0), device=device) < 0.5
            real[flip] = real[flip].flip(3)  # random horizontal flip augmentation
            b = real.size(0)

            # ---- Discriminator: maximize log D(x) + log(1 - D(G(z))) ----
            z = torch.randn(b, LATENT_DIM, 1, 1, device=device)
            fake = G(z)
            out_real = D(real)
            out_fake = D(fake.detach())
            # One-sided label smoothing (real = 0.9) stabilizes training
            loss_d = criterion(out_real, torch.full_like(out_real, 0.9)) + \
                criterion(out_fake, torch.zeros_like(out_fake))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- Generator: maximize log D(G(z)) (non-saturating loss) ----
            out_fake = D(fake)
            loss_g = criterion(out_fake, torch.ones_like(out_fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            sums["loss_d"] += loss_d.item()
            sums["loss_g"] += loss_g.item()
            sums["d_real"] += torch.sigmoid(out_real).mean().item()
            sums["d_fake"] += torch.sigmoid(out_fake).mean().item()

            if i % 50 == 0:
                step_log.append((epoch * steps_per_epoch + i, loss_d.item(), loss_g.item()))
            if i % 100 == 0:
                print(f"[{epoch + 1}/{args.epochs}] step {i}/{steps_per_epoch} "
                      f"loss_D {loss_d.item():.3f} loss_G {loss_g.item():.3f}")

        stats = {k: v / steps_per_epoch for k, v in sums.items()}
        stats["epoch"] = epoch + 1
        stats["time_s"] = round(time.time() - t0, 1)
        history.append(stats)
        print(f"Epoch {epoch + 1}: loss_D {stats['loss_d']:.3f} loss_G {stats['loss_g']:.3f} "
              f"D(x) {stats['d_real']:.3f} D(G(z)) {stats['d_fake']:.3f} ({stats['time_s']}s)")

        # Save generated samples from fixed noise to track quality over time
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, os.path.join(OUT, "samples", f"epoch_{epoch + 1:03d}.png"),
                   nrow=8, normalize=True, value_range=(-1, 1))

        ck = dict(G=G.state_dict(), D=D.state_dict(), opt_g=opt_g.state_dict(),
                  opt_d=opt_d.state_dict(), fixed_noise=fixed_noise, history=history,
                  step_log=step_log, epoch=epoch)
        torch.save(ck, ckpt_path)
        torch.save(G.state_dict(), os.path.join(CKPT, "generator.pt"))
        with open(os.path.join(OUT, "history.json"), "w") as f:
            json.dump({"epochs": history, "steps": step_log}, f)


if __name__ == "__main__":
    main()

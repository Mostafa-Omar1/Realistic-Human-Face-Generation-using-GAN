# Training Results

**Setup:** DCGAN, 64x64 faces, 67,533 CelebA images (1 shard), batch 128, Adam 2e-4 / β1 0.5,
25 epochs (13,175 iterations), GTX 1660 SUPER, ~65–90 s per epoch (~35 min total).

| Epoch | Loss D | Loss G | D(x) | D(G(z)) |
|---|---|---|---|---|
| 1  | 0.836 | 6.169 | 0.751 | 0.024 |
| 5  | 0.792 | 3.284 | 0.719 | 0.084 |
| 10 | 0.815 | 2.679 | 0.696 | 0.125 |
| 15 | 0.847 | 2.786 | 0.693 | 0.119 |
| 20 | 0.723 | 3.024 | 0.737 | 0.098 |
| 25 | 0.690 | 3.256 | 0.762 | 0.087 |

## Observations
- **Epoch 1:** blurry, grid-patterned blobs with only a rough face layout.
- **Epochs 5–7:** clear faces with hair, eyes, mouth and backgrounds.
- **Epochs 13–25:** sharper details, more consistent skin tones, and accessories such as glasses.
  The same noise vector keeps the same identity and pose, so G learned a stable mapping.
- No mode collapse: the generated faces differ in gender, hair, pose, skin tone and background,
  and latent interpolation (`outputs/interpolation.png`) moves smoothly between faces.
- Losses oscillate without diverging, which is normal for adversarial training. The Discriminator
  stays ahead (D(G(z)) ≈ 0.1), while G keeps improving visually.
- Limitations: some faces still show asymmetric eyes or smeared backgrounds. Improving this would
  need more data (all 3 shards), more epochs, or a stronger architecture (e.g. WGAN-GP or StyleGAN).

## Output files
`outputs/real_faces.png`, `outputs/final_faces.png`, `outputs/progress.png`,
`outputs/training.gif`, `outputs/loss_curve.png`, `outputs/interpolation.png`,
`outputs/samples/epoch_001..025.png`, `checkpoints/generator.pt`

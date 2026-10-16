"""DCGAN Generator and Discriminator for 64x64 RGB face images."""
import torch.nn as nn

LATENT_DIM = 100


class Generator(nn.Module):
    """Maps a latent noise vector z (LATENT_DIM x 1 x 1) to a 3x64x64 image in [-1, 1]."""

    def __init__(self, latent_dim=LATENT_DIM, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # z -> (base*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
            # -> (base*4) x 8 x 8
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            # -> (base*2) x 16 x 16
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            # -> base x 32 x 32
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            # -> 3 x 64 x 64
            nn.ConvTranspose2d(base, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Classifies a 3x64x64 image as real or fake; outputs a raw logit."""

    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # 3 x 64 x 64 -> base x 32 x 32
            nn.Conv2d(3, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (base*2) x 16 x 16
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (base*4) x 8 x 8
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (base*8) x 4 x 4
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> 1 x 1 x 1
            nn.Conv2d(base * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    """DCGAN init: conv weights ~ N(0, 0.02), batchnorm gamma ~ N(1, 0.02)."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

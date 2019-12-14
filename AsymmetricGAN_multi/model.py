import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator_xy(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator_xy, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

# Generator 3: the same as Generator_xy
# class Generator_yx(nn.Module):
#     """Generator network."""
#     def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
#         super(Generator_yx, self).__init__()
#
#         layers = []
#         layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         # Down-sampling layers.
#         curr_dim = conv_dim
#         for i in range(2):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2
#
#         # Bottleneck layers.
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
#
#         # Up-sampling layers.
#         for i in range(2):
#             layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim // 2
#
#         layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.Tanh())
#         self.main = nn.Sequential(*layers)
#
#     def forward(self, x, c):
#         # Replicate spatially and concatenate domain information.
#         # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
#         # This is because instance normalization ignores the shifting (or bias) effect.
#         c = c.view(c.size(0), c.size(1), 1, 1)
#         c = c.repeat(1, 1, x.size(2), x.size(3))
#         x = torch.cat([x, c], dim=1)
#         return self.main(x)

# Generator 2
class Generator_yx(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=2):
        super(Generator_yx, self).__init__()

        print('Generator yx with 2 residual layers')

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

# Generator 1
# class Generator_yx(nn.Module):
#     """Generator. CNN Architecture."""
#     def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
#         super(Generator_yx, self).__init__()
#
#         layers = []
#
#         layers.append(nn.Conv2d(3+c_dim, 9, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.InstanceNorm2d(9, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         layers.append(nn.Conv2d(9, 8, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.InstanceNorm2d(8, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         layers.append(nn.Conv2d(8, 7, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.InstanceNorm2d(7, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         layers.append(nn.Conv2d(7, 6, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.InstanceNorm2d(6, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         layers.append(nn.Conv2d(6, 5, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.InstanceNorm2d(5, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         layers.append(nn.Conv2d(5, 4, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.InstanceNorm2d(4, affine=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         layers.append(nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.Tanh())
#         self.main = nn.Sequential(*layers)
#
#     def forward(self, x, c):
#         # replicate spatially and concatenate domain information
#         c = c.view(c.size(0), c.size(1), 1, 1)
#         c = c.repeat(1, 1, x.size(2), x.size(3))
#         x = torch.cat([x, c], dim=1)
#         return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

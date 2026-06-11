import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Define Basic Blocks ----
class ConvBlock(nn.Module):
    r"""
    Defines a ConvBlock layer, consisting in two 3x3 convolutions with a stride of 1, each followed by a BatchNorm2d layer
    and a ReLU activation function.
    """

    def __init__(self, ch_in: int, ch_out: int, n_layers_per_block: int = 2) -> None:
        super(ConvBlock, self).__init__()

        self.conv = (
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        for _ in range(n_layers_per_block - 1):
            self.conv += (
                nn.Conv2d(
                    ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True
                ),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    r"""
    Defines a ResidualConvBlock layer, consisting in two 3x3 convolutions with a stride of 1, each followed by a BatchNorm2d layer
    and a ReLU activation function. After the convolution is applied, a skip connection between input and output is added.
    """

    def __init__(self, ch_in: int, ch_out: int, n_layers_per_block: int = 2) -> None:
        super(ResidualConvBlock, self).__init__()

        self.conv = (
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        for _ in range(n_layers_per_block - 1):
            self.conv += (
                nn.Conv2d(
                    ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True
                ),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Sequential(*self.conv)

        self.conv2 = nn.Conv2d(
            ch_out + ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        x = self.conv2(torch.cat((x, h), dim=1))
        return x


class DownBlock(nn.Module):
    """Standard downsampling block with Conv + ReLU + MaxPool"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_layers_per_block: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.conv = ConvBlock(ch_in, ch_out, n_layers_per_block)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x  # Return downsampled + skip connection


class ResDownBlock(nn.Module):
    """Residual downsampling block with Conv + Skip connection"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_layers_per_block: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.conv = ResidualConvBlock(ch_in, ch_out, n_layers_per_block)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x  # Return downsampled + skip connection


class UpBlock(nn.Module):
    """Standard upsampling block with ConvTranspose"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_layers_per_block: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * ch_in, ch_out, n_layers_per_block)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate with skip connection
        return self.conv(x)


class ResUpBlock(nn.Module):
    """Residual upsampling block with ConvTranspose + Skip"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_layers_per_block: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(2 * ch_in, ch_out, n_layers_per_block)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate with skip connection
        return self.conv(x)


class AttentionDownBlock(nn.Module):
    """Downsampling block with Multi-Head Self-Attention and Squeeze-Excitation"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_layers_per_block: int = 2,
        n_heads: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.conv = ResidualConvBlock(ch_in, ch_out, n_layers_per_block)

        # Squeeze-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
            nn.Sigmoid(),
        )

        # Multi-Head Self-Attention
        self.n_heads = n_heads
        self.mhsa = nn.MultiheadAttention(
            embed_dim=ch_out, num_heads=n_heads, batch_first=True
        )

        # Downsampling with pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)

        # Apply Squeeze-Excitation
        se_weight = self.se(x)
        x = x * se_weight

        # Reshape for Multi-Head Attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(
            1, 2
        )  # Shape: (batch, seq_len, channels)
        x_attn, _ = self.mhsa(x_flat, x_flat, x_flat)  # Self-attention
        x = x_attn.transpose(1, 2).view(b, c, h, w)  # Reshape back

        return self.pool(x), x  # Downsampled output + skip connection


class AttentionUpBlock(nn.Module):
    """Upsampling block with Attention-Guided Fusion using Multi-Head Self-Attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_layers_per_block: int = 2,
        n_heads: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2)

        # Multi-Head Self-Attention for feature refinement
        self.n_heads = n_heads
        self.mhsa = nn.MultiheadAttention(
            embed_dim=ch_in, num_heads=n_heads, batch_first=True
        )

        # Convolution layers
        self.conv = ResidualConvBlock(ch_in * 2, ch_out, n_layers_per_block)

    def forward(self, x, skip_connection):
        x = self.upconv(x)

        # Reshape for Multi-Head Self-Attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(
            1, 2
        )  # Shape: (batch, seq_len, channels)
        x_attn, _ = self.mhsa(x_flat, x_flat, x_flat)  # Self-attention
        x = x_attn.transpose(1, 2).view(b, c, h, w)  # Reshape back

        # Concatenate with skip connection and apply convolutions
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

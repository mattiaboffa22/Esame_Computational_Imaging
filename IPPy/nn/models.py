import torch
import torch.nn as nn

from ._blocks import *


blocks_dict = {
    "DownBlock": DownBlock,
    "ResDownBlock": ResDownBlock,
    "AttentionDownBlock": AttentionDownBlock,
    "UpBlock": UpBlock,
    "ResUpBlock": ResUpBlock,
    "AttentionUpBlock": AttentionUpBlock,
}


class UNet(nn.Module):
    r"""
    Initialize a UNet model, mapping a torch tensor with shape (N, input_ch, nx, ny) to a torch tensor
    with shape (N, output_ch, nx, ny).

    :param int input_ch: number of channels in input tensor.
    :param int output_ch: number of channels in output tensor.
    :param list[int] middle_ch: a list containing the number of channels in each convolution level. len(middle_ch) is the number
                                of downsampling levels in the resulting model. The input dimensions nx and ny both MUST be divisible
                                by 2 ** len(middle_ch).
    :param str final_activation: Can be either None, "relu" or "sigmoid". Activation function for the final layer.
    """

    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 1,
        middle_ch: tuple[int] = [64, 128, 256, 512, 1024],
        n_layers_per_block: int = 2,
        down_layers: tuple[str] = (
            "ResDownBlock",
            "ResDownBlock",
            "ResDownBlock",
            "ResDownBlock",
        ),
        up_layers: tuple[str] = (
            "ResUpBlock",
            "ResUpBlock",
            "ResUpBlock",
            "ResUpBlock",
        ),
        n_heads: int | None = None,
        final_activation: str | None = None,
    ) -> None:
        super(UNet, self).__init__()

        # Get all the properties as internal variables
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.middle_ch = middle_ch
        self.n_layers_per_block = n_layers_per_block
        self.down_layers = down_layers
        self.up_layers = up_layers
        self.n_heads = n_heads
        self.final_activation = final_activation

        # Define blocks
        self.preprocess = ConvBlock(ch_in, middle_ch[0])

        # ---- ENCODER ----
        self.encoder_layers = nn.ModuleList(
            [
                blocks_dict[down_layer](
                    middle_ch[i],
                    middle_ch[i + 1],
                    n_layers_per_block,
                    n_heads,
                )
                for i, down_layer in enumerate(self.down_layers)
            ]
        )

        # ---- BOTTLENECK ----
        self.bottleneck = ResidualConvBlock(
            middle_ch[-1], middle_ch[-1], n_layers_per_block
        )

        # ---- DECODER ----
        self.decoder_layers = nn.ModuleList(
            [
                blocks_dict[up_layer](
                    middle_ch[::-1][i],
                    middle_ch[::-1][i + 1],
                    n_layers_per_block,
                    n_heads,
                )
                for i, up_layer in enumerate(self.up_layers)
            ]
        )

        # ---- POSTPROCESS ----
        self.postprocess = nn.Conv2d(
            middle_ch[0], ch_out, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Preprocess
        h = self.preprocess(x)

        # Downpath
        for l in range(len(self.encoder_layers)):
            h, skip = self.encoder_layers[l](h)
            skip_connections.append(skip)

        # Bottleneck
        h = self.bottleneck(h)

        # Uppath
        for l in range(len(self.decoder_layers)):
            skip = skip_connections.pop()  # Retrieve last skip connection
            h = self.decoder_layers[l](h, skip)

        if self.final_activation is not None:
            if self.final_activation.lower() == "sigmoid":
                return nn.Sigmoid()(self.postprocess(h))
            elif self.final_activation.lower() == "relu":
                return nn.ReLU()(self.postprocess(h))
        return self.postprocess(h)

from torch import nn
from torch.nn import init


def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02):
    """
    Initialize the weights of a network, based on the layer type.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def get_config(model):
    r"""
    Return the configuration dictionary from the provided model, in the shape of
    a dictionary.
    """
    out_cfg = {
        "ch_in": model.ch_in,
        "ch_out": model.ch_out,
        "middle_ch": model.middle_ch,
        "n_layers_per_block": model.n_layers_per_block,
        "down_layers": model.down_layers,
        "up_layers": model.up_layers,
        "n_heads": model.n_heads,
        "final_activation": model.final_activation,
    }
    return out_cfg

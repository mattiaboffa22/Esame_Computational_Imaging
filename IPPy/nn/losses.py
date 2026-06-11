import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
except ImportError:
    models = None


class MixedLoss(nn.Module):
    def __init__(
        self,
        loss_vec: tuple,
        weight_parameters: tuple[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss_vec = loss_vec
        self.weight_parameters = weight_parameters

    def forward(self, x, y):
        out = 0
        for reg_param, loss_fn in zip(self.weight_parameters, self.loss_vec):
            out = out + reg_param * loss_fn(x, y)
        return out


class FourierLoss(nn.Module):
    def forward(self, x, y):
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_y = torch.fft.fft2(y, norm="ortho")
        return F.mse_loss(torch.abs(fft_x), torch.abs(fft_y))


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if models is None:
            raise ImportError(
                "torchvision is required to use PerceptualLoss, but it is not installed."
            )
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[
            :16
        ]  # Use early layers for texture
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.vgg = vgg.eval()

    def forward(self, x, y):
        # If x and y are grey-scale, repeat along channel dimension
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        return F.mse_loss(self.vgg(x), self.vgg(y))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def forward(self, x, y):
        mu_x = F.avg_pool2d(x, self.window_size)
        mu_y = F.avg_pool2d(y, self.window_size)

        sigma_x = F.avg_pool2d(x**2, self.window_size) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, self.window_size) - mu_y**2
        sigma_xy = F.avg_pool2d(x * y, self.window_size) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)
        )

        return 1 - ssim.mean()

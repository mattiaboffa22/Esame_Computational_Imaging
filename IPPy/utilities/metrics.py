import functools
import math

import torch
from skimage.metrics import structural_similarity as ssim


####################################################################
# Custom decorator
####################################################################
def average_on_batch(func):
    @functools.wraps(func)
    def wrapper(tensor1, tensor2, *args, **kwargs):
        # Ensure the inputs are tensors
        if not isinstance(tensor1, torch.Tensor) or not isinstance(
            tensor2, torch.Tensor
        ):
            raise TypeError("Input must be a PyTorch tensor")

        # Check the shape of the tensor
        assert tensor1.shape == tensor2.shape
        if tensor1.ndimension() == 4:
            N, c, h, w = tensor1.shape
            if N == 1:
                # If N == 1, directly apply the function and return the result
                return func(tensor1, tensor2, *args, **kwargs)
            else:
                # If N > 1, apply the function to each sample and average the results
                results = [
                    torch.tensor(
                        func(tensor1[i : i + 1], tensor2[i : i + 1], *args, **kwargs)
                    )
                    for i in range(N)
                ]
                return torch.mean(torch.stack(results), dim=0).item()
        else:
            raise ValueError("Input tensor must have shape (N, c, h, w)")

    return wrapper


####################################################################
# Metrics
####################################################################
@average_on_batch
def RE(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    r"""
    Compute relative error between two input tensors with shape (1, c, h, w).
    """
    return torch.norm(x_pred.flatten() - x_true.flatten(), 2) / torch.norm(
        x_true.flatten(), 2
    )


@average_on_batch
def SSIM(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    r"""
    Compute the SSIM between two input tensors x_pred and x_true. Both are assumed to be in the range [0, 1].
    """
    return ssim(
        x_pred[0, 0].detach().cpu().numpy(),
        x_true[0, 0].detach().cpu().numpy(),
        data_range=1,
    )


@average_on_batch
def PSNR(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    r"""
    Compute the PSNR between two input tensors x_pred and x_true. Both are assumed to be in the range [0, 1].
    """
    mse = torch.mean(torch.square(x_pred.flatten() - x_true.flatten()))
    if mse == 0:
        return 100
    return -20 * math.log10(math.sqrt(mse))


@average_on_batch
def RMSE(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    r"""
    Compute the Root Mean Squared Error (RMSE) between the two input tensors x_pred and x_true.
    """
    return torch.sqrt(torch.mean(torch.square(x_pred.flatten() - x_true.flatten())))

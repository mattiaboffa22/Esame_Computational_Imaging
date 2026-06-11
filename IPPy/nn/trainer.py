import time

import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset

from ..utilities import _utilities, metrics
from . import _utilities as nn_utils
from . import models
import json


def train(
    model: nn.Module,
    training_data: Dataset,
    optimizer: torch.optim.Optimizer,
    loss_fn=nn.MSELoss(),
    n_epochs: int = 50,
    batch_size: int = 4,
    save_each: int | None = None,
    weights_path: str | None = None,
    device: str = "cpu",
) -> None:
    r"""
    Train a given pytorch model on an input training set. Note that if save_each is defined not to be None, then a weigths_path
    has to be given as input as well. Otherwise, this function does not save the resulting model weights, and the user should
    save them by himself.

    :param nn.Module model: The model to be trained.
    :param training_data: A pytorch training dataset. Has to be initialized by the function TrainDataset from utilities.dataloader.
    :param loss_fn: A pytorch loss function.
    :param int n_epochs: The number of epochs of the training process.
    :param int batch_size: Number of samples in each batch.
    :param int save_each: If given, saves a model checkpoint every X epochs, where X is the value of save_each.
    :param str weights_path: If save_each is given, represents the path on which saving the weights of the model.
    :param str device: The device on which the operations are performed.
    """

    ### Initialize training
    # Define dataloader
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # Verbose
    print(f"Training NN model for {n_epochs} epochs and batch size of {batch_size}.")

    # Cycle over the epochs
    loss_history = []
    ssim_history = []
    for epoch in range(n_epochs):

        # Cycle over the batches
        epoch_loss = 0.0
        ssim_loss = 0.0
        start_time = time.time()
        for t, data in enumerate(train_loader):
            x, y = data

            # Send x and y to gpu
            x = x.to(device)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            # update loss
            epoch_loss += loss.item()
            ssim_loss += metrics.SSIM(y_pred.cpu().detach(), y.cpu().detach())

            # Measure time
            formatted_time = _utilities.formatted_time(start_time)

            # Verbose
            print(
                f"({formatted_time}) Epoch ({epoch+1} / {n_epochs}) -> Loss = {epoch_loss / (t + 1):0.4f}, "
                + f"SSIM = {ssim_loss / (t + 1):0.4f}.",
                end="\r",
            )

        # Update the history
        avg_loss = epoch_loss / (t + 1)
        avg_ssim = ssim_loss / (t + 1)
        loss_history.append(avg_loss)
        ssim_history.append(avg_ssim)

        # Save the weights of the model
        if save_each is not None and (epoch + 1) % save_each == 0:
            torch.save(
                model.state_dict(),
                weights_path,
            )
    print()
    return {"loss": loss_history, "ssim": ssim_history}


def save(model: nn.Module, weights_path: str):
    r"""
    Save a trained model provided as input in the required path.
    """
    # Create saving folder if does not exists
    _utilities.create_path_if_not_exists(weights_path)

    # Get configuration from model
    model_config = nn_utils.get_config(model)

    # Save model configuration as json
    with open(f"{weights_path}/config.json", "w") as fp:
        json.dump(model_config, fp, indent=2)

    # Save model weights
    torch.save(
        model.state_dict(),
        f"{weights_path}/weights.pth",
    )


def load(weights_path: str):
    r"""
    Load a trained model from the given path.
    """
    # Load configuration
    with open(f"{weights_path}/config.json") as fp:
        model_config = json.load(fp)

    # Get model by configuration
    model = models.UNet(**model_config)

    # Load model weights
    model.load_state_dict(torch.load(f"{weights_path}/weights.pth", weights_only=True))
    return model

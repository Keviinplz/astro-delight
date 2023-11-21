import os
import tempfile
from typing import TypedDict, cast

import torch
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader

from astro_delight.models.cnn.model import DelightCnn, DelightCnnParameters
from astro_delight.training.dataset import (
    DelightDataset,
    DelightDatasetOptions,
    DelightDatasetType,
)


class HyperParameters(TypedDict):
    lr: float
    batch_size: int | float
    nconv1: int | float
    nconv2: int | float
    nconv3: int | float
    ndense: int | float
    dropout: float


def _get_value_from_parameter(parameter: int | float, base: int = 2) -> int:
    return int(base**parameter) if isinstance(parameter, float) else parameter


def get_delight_cnn_parameters(
    params: HyperParameters, options: DelightDatasetOptions
) -> DelightCnnParameters:
    return {
        "nconv1": _get_value_from_parameter(params["nconv1"]),
        "nconv2": _get_value_from_parameter(params["nconv2"]),
        "nconv3": _get_value_from_parameter(params["nconv3"]),
        "ndense": _get_value_from_parameter(params["ndense"]),
        "levels": options.n_levels,
        "dropout": _get_value_from_parameter(params["dropout"]),
        "rot": options.rot,
        "flip": options.flip,
    }


def train_delight_cnn_model(params: HyperParameters, options: DelightDatasetOptions):
    batch_size = _get_value_from_parameter(params["batch_size"])
    lr = _get_value_from_parameter(params["lr"], base=10)

    device = "cpu" if torch.cuda.is_available() is False else "cuda"
    net = DelightCnn(get_delight_cnn_parameters(params, options))
    net.train(True)
    net.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    checkpoint = cast(Checkpoint | None, train.get_checkpoint())  # type: ignore
    start_epoch = 0

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))  # type: ignore
            start_epoch = int(checkpoint_dict["epoch"]) + 1
            net.load_state_dict(checkpoint_dict["net_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    train_dataset = DelightDataset(options=options, datatype=DelightDatasetType.TRAIN)
    val_dataset = DelightDataset(
        options=options, datatype=DelightDatasetType.VALIDATION
    )
    train_dl = DataLoader(train_dataset, batch_size=batch_size)
    val_dl = DataLoader(val_dataset, batch_size=batch_size)

    # Training

    for epoch in range(start_epoch, 50):
        running_loss = 0.0
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += float(loss.item())

            if i % batch_size == batch_size - 1:
                train_loss = running_loss / batch_size
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / (epoch + 1))
                )
                running_loss = 0.0

        # Validation

        val_loss = 0.0
        with torch.no_grad():
            for vinputs, vlabels in val_dl:
                vinputs, vlabels = (
                    vinputs.to(device),
                    vlabels.to(device),
                )

                voutputs = net(vinputs)

                vloss = criterion(voutputs, vlabels)
                val_loss += vloss.cpu().numpy()

        # Metrics
        metrics = {"val_loss": val_loss / len(val_dl), "train_loss": train_loss}
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(  # type: ignore
                {
                    "epoch": epoch,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))  # type: ignore

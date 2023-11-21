import os
import tempfile
from typing import TypedDict, cast

import torch
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader

from astro_delight.models.cnn.model import (
    DelightCnn,
    DelightCnnParameters,
)
from astro_delight.training.dataset import (
    DelightDataset,
    DelightDatasetOptions,
    DelightDatasetType,
)


class HyperParametersOptions(TypedDict):
    lr: float
    batch_size: int | float
    epochs: int


class TrainDelightCnnModelParameters(TypedDict):
    dataset: DelightDatasetOptions
    model: DelightCnnParameters
    hyperparams: HyperParametersOptions
    device: str


def train_delight_cnn_model(params: TrainDelightCnnModelParameters):
    net = DelightCnn(params["model"])
    net.train(True)
    net.to(params["device"])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=10 ** params["hyperparams"]["lr"])
    checkpoint = cast(Checkpoint | None, train.get_checkpoint())  # type: ignore
    start_epoch = 0

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))  # type: ignore
            start_epoch = int(checkpoint_dict["epoch"]) + 1
            net.load_state_dict(checkpoint_dict["net_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    train_dataset = DelightDataset(
        options=params["dataset"], datatype=DelightDatasetType.TRAIN
    )
    val_dataset = DelightDataset(
        options=params["dataset"], datatype=DelightDatasetType.VALIDATION
    )

    batch_size = (
        int(2 ** params["hyperparams"]["batch_size"])
        if isinstance(params["hyperparams"]["batch_size"], float)
        else params["hyperparams"]["batch_size"]
    )
    train_dl = DataLoader(train_dataset, batch_size=batch_size)
    val_dl = DataLoader(val_dataset, batch_size=batch_size)

    # Training

    running_loss = 0.0
    train_loss = 0.0
    for epoch in range(start_epoch, params["hyperparams"]["epochs"]):
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(params["device"]), labels.to(params["device"])

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += float(loss.item())

            if (
                i % params["hyperparams"]["batch_size"]
                == params["hyperparams"]["batch_size"] - 1
            ):  # print every 2000 mini-batches
                train_loss = running_loss / params["hyperparams"]["batch_size"]
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
                    vinputs.to(params["device"]),
                    vlabels.to(params["device"]),
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

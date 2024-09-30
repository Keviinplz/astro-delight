import os
import tempfile
from typing import Literal, TypedDict, cast

import torch
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils.logging import logger
from utils.stoppers import Stopper

from .dataset import (
    DelightDataset,
    DelightDatasetOptions,
    DelightDatasetType,
)
from .model import DelightCnn, DelightCnnParameters


class HyperParameters(TypedDict):
    lr: float
    batch_size: int | float
    nconv1: int | float
    nconv2: int | float
    nconv3: int | float
    ndense: int | float
    dropout: float
    epochs: int


def _get_value_from_parameter(parameter: int | float, base: int = 2) -> int:
    return int(base**parameter) if isinstance(parameter, float) else parameter


def get_delight_cnn_parameters(
    params: HyperParameters, options: DelightDatasetOptions
) -> DelightCnnParameters:
    return DelightCnnParameters(
        nconv1=_get_value_from_parameter(params["nconv1"]),
        nconv2=_get_value_from_parameter(params["nconv2"]),
        nconv3=_get_value_from_parameter(params["nconv3"]),
        ndense=_get_value_from_parameter(params["ndense"]),
        levels=options.n_levels,
        dropout=params["dropout"],
        rot=options.rot,
        flip=options.flip,
    )


def _train_one_epoch(
    *,
    device: str,
    train_dl: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    optimizer: torch.optim.Adam,  # type: ignore
    model: DelightCnn,
    criterion: torch.nn.Module,
):
    running_loss = 0.0
    last_loss = 0.0
    data: tuple[torch.Tensor, torch.Tensor]
    outputs: torch.Tensor
    loss: torch.Tensor

    model.train()
    for i, data in enumerate(train_dl):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()  # type: ignore

        optimizer.step()  # type: ignore

        running_loss += loss.item()

        if (i % batch_size) == (batch_size - 1):
            last_loss = running_loss / batch_size  # loss per batch
            running_loss = 0.0

    return last_loss


def _validate_train(
    *,
    device: str,
    val_dl: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: DelightCnn,
    criterion: torch.nn.Module,
):
    running_loss = 0.0
    data: tuple[torch.Tensor, torch.Tensor]
    outputs: torch.Tensor
    loss: torch.Tensor

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(val_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(val_dl)


def _train(
    *,
    start_epoch: int,
    num_epochs: int,
    batch_size: int,
    device: str,
    train_dl: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_dl: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Adam,  # type: ignore
    model: DelightCnn,
    criterion: torch.nn.Module,
    is_ray: bool,
    stopper: Stopper | None = None,
    writer: SummaryWriter | None = None,
):
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        train_loss = _train_one_epoch(
            device=device,
            train_dl=train_dl,
            batch_size=batch_size,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
        )

        val_loss = _validate_train(
            device=device, val_dl=val_dl, model=model, criterion=criterion
        )

        logger.info(
            f"[EPOCH {epoch+1}] train loss = {train_loss} | val_loss = {val_loss}"
        )

        metrics = {"val_loss": val_loss, "train_loss": train_loss}

        if is_ray is False:
            if writer:
                writer.add_scalars("[MSE Loss]: Train / Validation", metrics, epoch)  # type: ignore
        else:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(  # type: ignore
                    {
                        "epoch": epoch,
                        "net_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(  # type: ignore
                    metrics=metrics,
                    checkpoint=Checkpoint.from_directory(tempdir),  # type: ignore
                )
        if stopper and stopper.early_stop(validation_loss=val_loss):
            logger.info(f"Stopped due Early Stop condition, last epoch: {epoch}")
            break

    if writer:
        writer.close()


def train_delight_cnn_model(
    params: HyperParameters,
    options: DelightDatasetOptions,
    stopper: Stopper | None = None,
    environment: Literal["cpu"] | Literal["cuda"] | None = None,
    writer: SummaryWriter | None = None,
    production: bool = False,
) -> DelightCnn:
    if not environment:
        device = "cpu" if torch.cuda.is_available() is False else "cuda"
    else:
        device = environment
    batch_size = _get_value_from_parameter(params["batch_size"])
    net_options = get_delight_cnn_parameters(params, options)
    net = DelightCnn(net_options)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params["lr"], weight_decay=1e-4)  # type: ignore
    checkpoint = cast(Checkpoint | None, train.get_checkpoint())  # type: ignore
    start_epoch = 0

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))  # type: ignore
            start_epoch = int(checkpoint_dict["epoch"]) + 1
            net.load_state_dict(checkpoint_dict["net_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    train_dtype = DelightDatasetType.P_TRAIN if production else DelightDatasetType.TRAIN
    val_dtype = (
        DelightDatasetType.P_VAL if production else DelightDatasetType.VALIDATION
    )

    train_dataset = DelightDataset(options=options, datatype=train_dtype)
    val_dataset = DelightDataset(options=options, datatype=val_dtype)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        "Starting: epochs=%s,batch_size=%s,lr=%s,nconv1=%s,nconv2=%s,nconv3=%s,ndense=%s,dropout=%s"
        % (
            params["epochs"],
            batch_size,
            params["lr"],
            net_options.nconv1,
            net_options.nconv2,
            net_options.nconv3,
            net_options.ndense,
            net_options.dropout,
        )
    )

    _train(
        start_epoch=start_epoch,
        num_epochs=params["epochs"],
        batch_size=batch_size,
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        model=net,
        criterion=criterion,
        is_ray=checkpoint is not None,
        stopper=stopper,
        writer=writer,
    )

    return net

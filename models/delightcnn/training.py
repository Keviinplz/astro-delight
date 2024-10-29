import logging
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import ray
import torch
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.delightcnn.dataset import DelightDataset, DelightDatasetOptions
from models.delightcnn.schemas import DelightCnnParameters
from utils.stoppers import Stopper

from .model import DelightCnn

logging.getLogger(__name__).addHandler(logging.NullHandler())


class HyperParameters(TypedDict):
    nconv1: int | float
    nconv2: int | float
    nconv3: int | float
    ndense: int | float
    dropout: float
    batch_size: int | float


@dataclass
class TrainingOptions:
    criterion: torch.nn.Module
    dataset_options: DelightDatasetOptions
    optimizer: partial[torch.optim.Optimizer]  # type: ignore
    train_dataset: DelightDataset
    val_dataset: DelightDataset
    epochs: int
    device: torch.device


@dataclass
class _ParsedHyperParameters:
    nconv1: int
    nconv2: int
    nconv3: int
    ndense: int
    dropout: float
    batch_size: int


def _get_value_from_parameter(parameter: int | float, base: int = 2) -> int:
    return int(base**parameter) if isinstance(parameter, float) else parameter


def parse_ray_param_space(params: HyperParameters) -> _ParsedHyperParameters:
    return _ParsedHyperParameters(
        nconv1=_get_value_from_parameter(params["nconv1"]),
        nconv2=_get_value_from_parameter(params["nconv2"]),
        nconv3=_get_value_from_parameter(params["nconv3"]),
        ndense=_get_value_from_parameter(params["ndense"]),
        dropout=params["dropout"],
        batch_size=_get_value_from_parameter(params["batch_size"]),
    )


def train_one_epoch(
    *,
    device: torch.device,
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    optimizer: torch.optim.Optimizer,  # type: ignore
    model: DelightCnn,
    criterion: torch.nn.Module,
):
    running_loss = 0.0
    last_loss = 0.0
    data: tuple[torch.Tensor, torch.Tensor]
    outputs: torch.Tensor
    loss: torch.Tensor

    model.train()
    for i, data in enumerate(train_dataloader):
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


def validate_train(
    *,
    device: torch.device,
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: DelightCnn,
    criterion: torch.nn.Module,
):
    running_loss = 0.0
    data: tuple[torch.Tensor, torch.Tensor]
    outputs: torch.Tensor
    loss: torch.Tensor

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(val_dataloader)


def execute_train_model(
    model_parameters: DelightCnnParameters,
    criterion: torch.nn.Module,
    optimizer: partial[torch.optim.Optimizer],  # type: ignore
    train_dataset: DelightDataset,
    val_dataset: DelightDataset,
    epochs: int = 50,
    batch_size: int = 40,
    device: torch.device = torch.device("cuda"),
    stopper: Stopper | None = None,
    writter: SummaryWriter | None = None,
):
    model = DelightCnn(options=model_parameters)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer_function = optimizer(params=model.parameters())
    model.to(device)
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            device=device,
            train_dataloader=train_dataloader,
            batch_size=batch_size,
            optimizer=optimizer_function,
            model=model,
            criterion=criterion,
        )

        val_loss = validate_train(
            device=device,
            val_dataloader=val_dataloader,
            model=model,
            criterion=criterion,
        )

        logging.info(
            f"[EPOCH {epoch+1}] train loss = {train_loss} | val_loss = {val_loss}"
        )
        metrics = {"val_loss": val_loss, "train_loss": train_loss}

        if writter is not None:
            writter.add_scalars("[MSE Loss]: Train / Validation", metrics, epoch)  # type: ignore

        if ray.is_initialized():  # type: ignore
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(  # type: ignore
                    {
                        "epoch": epoch,
                        "net_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer_function.state_dict(),
                    },
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(  # type: ignore
                    metrics=metrics,
                    checkpoint=Checkpoint.from_directory(tempdir),  # type: ignore
                )

        if stopper and stopper.early_stop(validation_loss=val_loss):
            logging.info(f"Stopped due Early Stop condition, last epoch: {epoch}")
            break

    return model


def ray_wrapper_training_function(
    param_space: HyperParameters, training_options: TrainingOptions
):
    parsed_parameters = parse_ray_param_space(param_space)
    dataset_options = training_options.dataset_options

    parameters = DelightCnnParameters(
        channels=dataset_options.channels,
        levels=dataset_options.levels,
        rot=dataset_options.rot,
        flip=dataset_options.flip,
        nconv1=parsed_parameters.nconv1,
        nconv2=parsed_parameters.nconv2,
        nconv3=parsed_parameters.nconv3,
        ndense=parsed_parameters.ndense,
        dropout=parsed_parameters.dropout,
    )
    return execute_train_model(
        model_parameters=parameters,
        criterion=training_options.criterion,
        optimizer=training_options.optimizer,
        train_dataset=training_options.train_dataset,
        val_dataset=training_options.val_dataset,
        epochs=training_options.epochs,
        batch_size=parsed_parameters.batch_size,
        device=training_options.device,
    )

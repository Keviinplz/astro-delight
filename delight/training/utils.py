from datetime import datetime
from typing import cast

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def params_to_string(params: dict[str, object]) -> str:
    name = ""
    for key, value in params.items():
        name += f"{key}_{value}-"
    return name[:-1]


def train(
    epochs: int,
    device: str,
    model: torch.nn.Module,
    train_dl: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_dl: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    batch_size: int,
    tb_writer: SummaryWriter,
):
    best_vloss = 1_000_000.0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pbar = tqdm(range(epochs), leave=False, position=0)
    for epoch in pbar:
        pbar.set_description("Running epoch %s" % (epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        model.to(device)
        avg_loss = train_one_epoch(
            dataset=train_dl,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epoch=epoch,
            tb_writer=tb_writer,
            device=device,
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            vdata: tuple[torch.Tensor, torch.Tensor]
            for vdata in val_dl:
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)

                vloss: float = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / len(val_dl)
        pbar.set_description(
            "LOSS train %s valid %s" % (avg_loss, avg_vloss), refresh=False
        )

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars(  # type:ignore
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch + 1,
        )
        tb_writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "states/model_{}_{}".format(ts, epoch)
            torch.save(model.state_dict(), model_path)  # type:ignore

    return best_vloss


def train_one_epoch(
    *,
    dataset: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch_size: int,
    epoch: int,
    tb_writer: SummaryWriter,
    device: str = "cuda",
):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    pbar = tqdm(dataset, leave=False, position=1)
    for i, data in enumerate(pbar):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += cast(float, loss.item())
        if i % batch_size == batch_size - 1:
            last_loss = running_loss / batch_size  # loss per batch
            pbar.set_description("batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch * len(dataset) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)  # type:ignore
            running_loss = 0.0

    return last_loss

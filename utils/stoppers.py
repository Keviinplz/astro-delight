import logging
from typing import Protocol

logging.getLogger(__name__).addHandler(logging.NullHandler())


class Stopper(Protocol):
    def early_stop(self, validation_loss: float) -> bool:
        raise NotImplementedError


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            logging.info(
                f"Validation loss has been improved from {self.min_validation_loss} -> {validation_loss}"
            )
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            logging.info(
                f"Validation loss is not improving. Best val loss={self.min_validation_loss}"
            )
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

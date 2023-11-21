import os
from typing import Callable, TypeVar, TypedDict

from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.schedulers import AsyncHyperBandScheduler

TrainingParameters = TypeVar("TrainingParameters", bound=TypedDict)

def search_hyperparameters(train_fn: Callable[[TrainingParameters], float], config: TrainingParameters):
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        max_t=400,
        grace_period=20,
        reduction_factor=3,
        brackets=1,
    )

    num_training_iterations = 75

    reporter = JupyterNotebookReporter(overwrite=False, max_report_frequency=60)

    return tune.run(  # type: ignore
        train_fn,
        name="exp27",
        metric="val_loss",
        mode="min",
        scheduler=scheduler,
        local_dir=os.path.join(os.getcwd(), "raytune"),
        sync_config=tune.SyncConfig(),
        stop={"training_iteration": num_training_iterations},
        num_samples=150,
        resources_per_trial={"cpu": 5, "gpu": 1},
        verbose=3,
        progress_reporter=reporter,
        checkpoint_score_attr="val_loss",
        config=config, # type: ignore
    )



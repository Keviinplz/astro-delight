from typing import Callable

import numpy as np
import numpy.typing as npt
from sklearn.utils import resample  # type: ignore

StatisticFunction = Callable[[npt.NDArray[np.float32]], float]


def bootstrap_statistic(
    data: npt.NDArray[np.float32],
    statistic: StatisticFunction,
    n_iterations: int = 1000,
) -> float:
    stats = np.zeros(n_iterations)
    for i in range(n_iterations):
        sample: npt.NDArray[np.float32] = resample(data)  # type: ignore
        stats[i] = statistic(sample)
    return np.std(stats).item()


def __assert_expected_shape(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]
) -> None:
    has_shape_2 = len(y_true.shape) == len(y_pred.shape) == 2
    are_points = y_true.shape[1] == y_pred.shape[1] == 2
    assert (
        has_shape_2 and are_points
    ), f"Expected vectors of dim (N, 2): y_true={y_true.shape} y_pred={y_pred.shape}"


def delta_mean(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], pixscale: float
) -> npt.NDArray[np.float32]:
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))


def rmse(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], pixscale: float
) -> tuple[float, float]:
    __assert_expected_shape(y_true, y_pred)

    sum_distance_squared: npt.NDArray[np.float32] = np.sum(
        (y_true - y_pred) ** 2, axis=1
    )

    value = np.sqrt(np.mean(sum_distance_squared)) * pixscale  # type: ignore
    assert isinstance(value, float), f"Expected float result: {value}"
    return value, bootstrap_statistic(
        sum_distance_squared, lambda x: np.sqrt(np.mean(x)) * pixscale
    )


def mean_deviation(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], pixscale: float
) -> tuple[float, float]:
    __assert_expected_shape(y_true, y_pred)

    deviation: npt.NDArray[np.float32] = np.linalg.norm(y_true - y_pred, axis=1)  # type: ignore
    return np.mean(deviation).item() * pixscale, bootstrap_statistic(
        deviation,
        lambda x: np.mean(x) * pixscale,  # type: ignore
    )


def median_deviation(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], pixscale: float
) -> tuple[float, float]:
    __assert_expected_shape(y_true, y_pred)

    deviation: npt.NDArray[np.float32] = np.linalg.norm(y_true - y_pred, axis=1)  # type: ignore
    return np.median(deviation).item() * pixscale, bootstrap_statistic(
        deviation,
        lambda x: np.median(x) * pixscale,  # type: ignore
    )


def __get_mode_use_numpy(deviation: npt.NDArray[np.float32]) -> float:
    counts, bin_edges = np.histogram(deviation, bins=np.linspace(0, 10, 200))
    mode_bin_index = np.argmax(counts)
    mode = (bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2

    return mode


def mode_deviation(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], pixscale: float
) -> tuple[float, float]:
    __assert_expected_shape(y_true, y_pred)

    deviation: npt.NDArray[np.float32] = (
        np.linalg.norm(y_true - y_pred, axis=1) * pixscale
    )  # type: ignore
    mode = __get_mode_use_numpy(deviation)

    return mode, bootstrap_statistic(
        deviation,
        __get_mode_use_numpy,
    )

from dataclasses import dataclass
from enum import StrEnum


class PanstarrsChannel(StrEnum):
    G = "g"
    I = "i"  # noqa: E741
    R = "r"
    Y = "y"
    Z = "z"


@dataclass
class PanstarrsImageMetadata:
    name: str
    ra: float  # right ascension
    dec: float  # declination


@dataclass
class PanstarrsDownloadResult:
    success: bool
    message: str
    path: str | None

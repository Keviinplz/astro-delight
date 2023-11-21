from typing import TypedDict
from typing_extensions import Unpack

from astro_delight.models.cnn import DelightCnn
from astro_delight.training.utils import train

class RayCnnTrainerParameters(TypedDict):
    dropout: float
    log10lr: float
    log2nconv1: float
    log2nconv2: float
    log2nconv3: float
    log2ndense: float
    log2batch_size: float
    
def ray_cnn_trainer(**kwargs: Unpack[RayCnnTrainerParameters]):

    nconv1 = int(2**kwargs["log2nconv1"])
    nconv2 = int(2**kwargs["log2nconv2"])
    nconv3 = int(2**kwargs["log2nconv3"])
    ndense = int(2**kwargs["log2ndense"])
    
    model = DelightCnn(
        nconv1=nconv1,
        nconv2=nconv2,
        nconv3=nconv3,
        ndense=ndense,
        levels=5,
        dropout=kwargs["dropout"],
        rot=True,
        flip=True
    )

    
import os
import asyncio
from enum import Enum
from typing import cast
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
from astropy import wcs  # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
from astropy.io import fits  # type: ignore
from astroquery.hips2fits import hips2fits  # type: ignore

class PanstarrsChannel(str, Enum):
    G = 'g'
    I = 'i'
    R = 'r'
    Y = 'y'
    Z = 'z'

@dataclass
class PanstarrsImageMetadata:
    name: str
    ra: float  # right ascension
    dec: float # declination

@dataclass
class DownloadResult:
    success: bool
    message: str
    path: str | None

def build_wcs_header_from_image_metadata(meta: PanstarrsImageMetadata) -> dict[str, str | int | float]:
    return {
        'NAXIS': 2,
        'NAXIS1': 480,       
        'NAXIS2': 480,       
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CDELT1': 6.94444461259988E-05,
        'CDELT2': 6.94444461259988E-05,
        'CRPIX1': 240.0,
        'CRPIX2': 240.0,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CRVAL1': meta.ra,
        'CRVAL2': meta.dec,
        'PC1_1': -1.,
        'PC1_2': 0.,
        'PC2_1': 0.,
        'PC2_2': 1.
    }

def download_data_by_channel_and_metadata(
    channel: PanstarrsChannel, 
    meta: PanstarrsImageMetadata, 
    path_to_save: str
) -> DownloadResult:
    filename = "%s/stack_%s_%s_ra%.6f_dec%.6f_arcsec120.fits" % (path_to_save, meta.name, channel.value, meta.ra, meta.dec)
    
    if os.path.exists(filename):
        return DownloadResult(success=True, message="[OK] File already exists", path=filename)
    
    wcs_header = build_wcs_header_from_image_metadata(meta)
    w = wcs.WCS(header=wcs_header)
    hips = f'CDS/P/PanSTARRS/DR1/{channel.value}'

    retries = 30
    error = None
    while retries > 0:
        try:
            result = cast(fits.hdu.HDUList, hips2fits.query_with_wcs(  # type: ignore
                hips=hips,
                wcs=w,
                get_query_payload=False,
                format='fits')
            )
            result.writeto(filename, overwrite=True)  # type: ignore
            break
        except Exception as e:
            retries -= 1
            error = e
    
    if error is not None:
        return DownloadResult(success=False, message=f"[ERROR] {str(error)}", path=None)
        
    return DownloadResult(success=True, message="[OK] File downloaded", path=filename)


def extract_sky_coords_from_row(coords: SkyCoord) -> tuple[float, float]:
    return coords.ra.degree, coords.dec.degree  # type: ignore

async def main():
    # Test if works
    loop = asyncio.get_running_loop()
    tasks: list[asyncio.Future[DownloadResult]] = []

    with ProcessPoolExecutor() as executor:
        DATADIR_SOURCE = os.path.join(os.getcwd(), "data")
        path = os.path.join(DATADIR_SOURCE, "coords_all_data_nlevels5_maskFalse_objectsTrue.pkl")
        print(f"Getting images from {path}")

        with open(path, 'rb') as f:
            coords: pd.DataFrame = pd.read_pickle(f)
        channels = [PanstarrsChannel(c) for c in ['g', 'i', 'r', 'y', 'z']]
        data = coords["sn_coords"].apply(extract_sky_coords_from_row) # type: ignore

        total_images = len(data) * len(channels)  # type: ignore
        print(f"Preparing download tasks... above {total_images} images to download")  # type: ignore

        name: str
        sky_coords: tuple[float, float]
        for name, sky_coords in tqdm(data.items(), desc="Creating tasks"):  # type: ignore
            for channel in channels:
                tasks.append(
                    loop.run_in_executor(
                        executor, 
                        download_data_by_channel_and_metadata,
                        channel,
                        PanstarrsImageMetadata(name, *sky_coords),  # type: ignore
                        DATADIR_SOURCE
                    )
                ) 


        results: list[DownloadResult] = cast(list[DownloadResult], await tqdm_async.gather(*tasks, desc="Downloading"))  # type: ignore

        successes = list(filter(lambda result: result.success, results))
        failed = list(filter(lambda result: result.success is False, results))

        print(f"Finished {len(successes)}/{total_images} images downloaded, {len(failed)} failed")

if __name__ == '__main__':
    asyncio.run(main())
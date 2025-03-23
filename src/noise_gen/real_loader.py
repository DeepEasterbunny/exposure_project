import os
from pathlib import Path
import tempfile

import dask.array as da
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

import hyperspy.api as hs
import kikuchipy as kp
import h5py
from omegaconf import OmegaConf
import typer

def get_data(data_path:str):
    ds = [file.name for file in data_path.glob("*.hdf5")]
    if len(ds) == 0:
        print(f'No master patterns found in {data_path}')
    return ds


def main(cfg_path:str = 'configs/config.yaml'):
    cfg = OmegaConf.load(cfg_path)

    data_path = cfg['paths']['real']
    file_list = get_data(data_path=Path(data_path))
    # print(len(file_list))
    for file in file_list:
        f = h5py.File(data_path + file, 'r')
        # print(f.keys())
        # print(f['Scan 1']['EBSD'].keys())
        pattern = f['Scan 1']['EBSD']['Data']['RawPatterns'][3212,:,:]
        # ['DD', 'MAD', 'MADPhase', 'NIndexedBands', 'PCX', 'PCY', 'PHI', 'Phase', 'RadonBandCount', 'RadonQuality', 'RawPatterns', 'X BEAM', 'X SAMPLE', 'Y BEAM', 'Y SAMPLE', 'Z SAMPLE', 'phi1', 'phi2']
        print(f['Scan 1']['EBSD']['Data'].keys())
        print(f"Pattern center X: {f['Scan 1']['EBSD']['Data']['PCX'][2]}")
        print(f"Pattern center X mean: {np.mean(f['Scan 1']['EBSD']['Data']['PCX'][:])}")
        print(f"Pattern center X variance: {np.var(f['Scan 1']['EBSD']['Data']['PCX'][:])}")
        print(f"Sample tilt: {f['Scan 1']['EBSD']['Header']['SampleTilt'][()]}")
        print(f"Pattern height: {f['Scan 1']['EBSD']['Header']['PatternHeight'][()]}")
        print(f"Patternwidth: {f['Scan 1']['EBSD']['Header']['PatternWidth'][()]}")
        print(f"Camera Tilt: {f['Scan 1']['EBSD']['Header']['CameraTilt'][()]}")

        pattern_scaled = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))

        plt.figure()
        plt.imshow(pattern_scaled, cmap='gray', vmin=0, vmax=1)
        plt.savefig(data_path + 'fig1.png')

if __name__ == '__main__':
    typer.run(main)

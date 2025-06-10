import os
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import hyperspy.api as hs
import kikuchipy as kp


from diffpy.structure import Atom, Lattice, Structure
from diffsims.crystallography import ReciprocalLatticeVector
from kikuchipy.detectors import EBSDDetector

from orix import io, plot, quaternion, vector, sampling
from orix.crystal_map import Phase
from orix.quaternion import Rotation

import h5py
import os
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import KikuchiDataset
import typer

# SET TO FALSE LATER
PROGRESS_BARS = True
CREATE_DATASET = True

def main(cfg_path:str = 'configs/config_ebsd.yaml'):
    print(f"Using config path: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # Rotation settings
    use_eulers = cfg['experiments']['use_eulers']   
    if use_eulers:
        print("Loading rotations from euler file")
    else:
        print(f"Creating random rotations with orix.quaternions.Rotation")


    mp_path = Path(cfg['paths']['mp'])
    mp_names = get_mps(mp_path)
    print(mp_names)
    
    num_mps = len(mp_names)
    
    for i, master_pattern in enumerate(mp_names):
        file_name = str(master_pattern).split(sep = '/')[-1]
        file_name = file_name.split(sep = '.')[0]
        dict_name = cfg['paths']['dicts'] + file_name + '.h5'
        file_name = cfg['paths']['torch_data'] + file_name + '.pt'

        print(f"Loading master pattern {i+1} of {num_mps}")

        mp = kp.load(mp_path / master_pattern)
        mp_lp = mp.as_lambert(show_progressbar = PROGRESS_BARS)

        data_path = cfg['paths']['real'] + 'TWIP_full_map_01.hdf5'
        #################
        # Code from Yuxan
        #################
        with h5py.File(data_path, 'r') as f:
            DD = f["Scan 1"]['EBSD']['Data']["DD"][()] # dictionary of data
            dx = f["Scan 1"]['EBSD']['Header']['XSTEP'][()] # x-stepsize [um]
            dy = f["Scan 1"]['EBSD']['Header']['YSTEP'][()] # y-stepsize [um]
            Nx = f["Scan 1"]['EBSD']['Header']['NCOLS'][()] # scanned map width [pixels]
            Ny = f["Scan 1"]['EBSD']['Header']['NROWS'][()] # scanned map height [pixels]
            Qx = f["Scan 1"]['EBSD']['Header']['PatternHeight'][()] # pattern height [pixels]
            Qy = f["Scan 1"]['EBSD']['Header']['PatternWidth'][()] # pattern width [pixels]
            sample_tilt = f["Scan 1"]['EBSD']['Header']['SampleTilt'][()] # sample stage tilt [deg]
            camera_tilt = f["Scan 1"]['EBSD']['Header']['CameraTilt'][()] # detector tilt [deg]
            px_x = f["Scan 1"]['EBSD']['Header']['SEPixelSizeX'][()] # x-pixelsize [mm]
            px_y = f["Scan 1"]['EBSD']['Header']['SEPixelSizeY'][()] # y-pixelsize [mm]
            camera_height = f["Scan 1"]['EBSD']['Header']['DetectorFullHeightMicrons'][()] # detector height [um]
            camera_width = f["Scan 1"]['EBSD']['Header']['DetectorFullWidthMicrons'][()] # detector width [um]
            bin = f["Scan 1"]['EBSD']['Header']['MapStepFactor'][()] # binning factor
            DD = f["Scan 1"]["EBSD"]["Data"]["DD"][()] # detector distance [% of camera height]
            PCX = f["Scan 1"]["EBSD"]["Data"]["PCX"][()] # pattern center x [% of pattern width]
            PCY = f["Scan 1"]["EBSD"]["Data"]["PCY"][()] # pattern center y [% of pattern height]
            PC = (PCX.mean(), PCY.mean(), DD.mean()) # pattern center [Bruker convention]
            real_patterns = f['Scan 1']['EBSD']['Data']['RawPatterns'][:,:,:]

        detector = kp.detectors.EBSDDetector(
            shape=(Qx, Qy), pc=PC,
            sample_tilt=sample_tilt, 
            tilt=camera_tilt,
            convention="bruker"
        )

        print(f"Creating dictionary with {detector = }")

        detector_values = {"shape": (Qx, Qy),
                        "pc": [PCX.mean(), PCY.mean(), DD.mean()],
                        "sample_tilt": sample_tilt,
                        "tilt": camera_tilt}
        

        rots  = sampling.get_sample_fundamental(
            method="cubochoric", resolution=1.5, point_group=mp_lp.phase.point_group
            )
        print(rots.shape)
        print("Getting patterns")
        s = mp_lp.get_patterns(rotations = rots, detector = detector, show_progressbar=PROGRESS_BARS, compute = True, dtype_out=np.float32)
        print(s.detector)
        s.save(dict_name)

        
def get_mps(mp_path:str):
    mps = [file.name for file in mp_path.glob("*.h5")]
    if len(mps) == 0:
        print(f'No master patterns found in {mp_path}')
    return mps

def get_values_from_config(cfg):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    generated_values = {}

    for key, value in cfg_dict.items():
        if isinstance(value, list) and len(value) == 2: 
            generated_values[key] = np.random.uniform(value[0], value[1])
            if key == 'sample_tilt':
                generated_values[key] = np.round(generated_values[key], decimals = 1)
            else:
                generated_values[key] = np.round(generated_values[key], decimals = 3)
        else:  
            generated_values[key] = value
    
    return generated_values


def createDetector(detector_values):
    """"
    reference https://kikuchipy.org/en/latest/reference/generated/kikuchipy.detectors.EBSDDetector.html
    
    shape: output shape, in (rows x columns)
    tilt: detector tilt in degrees
    sample_tilt: tilt of sample
    pc: Centers of the projection in fractions of total size, you can input a list of lists here.

    """

    shape = (detector_values['height'], detector_values['width'])
    tilt = 0.0
    sample_tilt = detector_values['sample_tilt']
    pc = [detector_values['pcx'], detector_values['pcy'], detector_values['pcz']]    
    
    return EBSDDetector(shape = shape, tilt = tilt, sample_tilt = sample_tilt, pc = pc)

def rotationFromEuler(anlgefile):
    with open(anlgefile, 'r') as f:
        lines = f.readlines()
    num_lines = int(lines[1].strip())
    data = np.loadtxt(lines[2:2 + num_lines], delimiter=",") 
    
    return Rotation.from_euler(euler = data, degrees = True)

def patternsAsPNG(mp, rots:Rotation, det:EBSDDetector, filename:str = 'out.png'):
    s = mp.get_patterns(rotations = rots, detector = det, show_progressbar=False)
    _ = hs.plot.plot_images(
        s, axes_decor=None, label=None, colorbar=False, tight_layout=True
    )
    plt.savefig('figs/' + filename)
    return s




if __name__ == '__main__':
    typer.run(main)

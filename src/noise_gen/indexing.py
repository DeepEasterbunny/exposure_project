#%% Import Package
from pathlib import Path
import orix
import pyebsdindex
import numpy as np
import kikuchipy as kp
from orix import io, sampling
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Orientation, Rotation
import typer
import h5py
import torch
from dataset import KikuchiDataset
import matplotlib.pyplot as plt

def Hybrid_Kikuchi_Indexing(
    s: kp.signals.EBSD, 
    mp: kp.signals.EBSDMasterPattern, 
    indexer: pyebsdindex._ebsd_index_single.EBSDIndexer,
    pc_refinement=False, downsample_factor=5,
    orientation_resolution=1.5,
    orientation_refinement=True,
    result_saving=True):
    """
    This function performs Kikuchi indexing on the TKD data set:
        1. Hough indexing is performed on the background corrected data set.
        2. Dictionary indexing is performed on unindexed points.

    s: EBSD signal (ideally background corrected, and with detector calibration)
    mp: Master pattern
    """
    pl = PhaseList(mp.phase)
    # - - - - - - - - - - - - - - - - - - - - - - - -
    # Hough indexing
    xmap_hi = s.hough_indexing(pl, indexer, verbose=2)

    if pc_refinement:
        xmap_hi, det = s.refine_orientation_projection_center(
            xmap_hi, s.detector, mp, energy=30, 
            trust_region=[5, 5, 5, 0.1, 0.1, 0.1])

    # - - - - - - - - - - - - - - - - - - - - - - - -
    # Dictionary indexing
    R_sample = sampling.get_sample_fundamental(
        resolution=orientation_resolution, 
        point_group=mp.phase.point_group)
    O_sample = Orientation(R_sample, symmetry=mp.phase.point_group)

    # Downsample the signal to speed up the indexing
    s_dummy = s.deepcopy() # this is necessary because the inplace=False argument does not work
    if downsample_factor > 1:
        s_downsampled = s_dummy.downsample(downsample_factor, inplace=False)
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Data set has been downsampled by a factor of {downsample_factor}.")
    else:
        s_downsampled = s_dummy.deepcopy()
    # s_downsampled.detector.shape = s_downsampled.detector.shape[::-1]
    det = s_downsampled.detector.deepcopy()
    print("- - - - - - - - - - - - - - - - - - - - - - - -")
    print(f"Before dictionary indexing, the detector shape is {det.shape}")
    # det.shape = s_downsampled.axes_manager.signal_shape[::-1]
    s_dict = mp.get_patterns(O_sample, det, energy=30)

    xmap_di = s_downsampled.dictionary_indexing(s_dict)
    
    if orientation_refinement:
        xmap_di = s_downsampled.refine_orientation(
            xmap=xmap_di, detector=det,
            master_pattern=mp, energy=30,
            trust_region=[2, 2, 2],
        )
    print("- - - - - - - - - - - - - - - - - - - - - - - -")
    print(f"After dictionary indexing, the detector shape is {det.shape}")
    
    if result_saving:
        io.save("xmap_hi.h5", xmap_hi, overwrite=True)
        io.save("xmap_di.h5", xmap_di, overwrite=True)

    return xmap_hi, xmap_di


def Hough_Indexing(
    s: kp.signals.EBSD, 
    mp: kp.signals.EBSDMasterPattern,
    indexer: pyebsdindex._ebsd_index_single.EBSDIndexer,
    pc_refinement=False, result_saving=True, result_saving_name="xmap_hi.h5"):
    """
    This function performs Hough Indexing on the TKD data set
    """
    pl = PhaseList(mp.phase)
    # Hough indexing
    xmap_hi = s.hough_indexing(pl, indexer, verbose=2)

    if pc_refinement:
        xmap_hi, det = s.refine_orientation_projection_center(
            xmap_hi, s.detector, mp, energy=30, 
            trust_region=[5, 5, 5, 0.1, 0.1, 0.1])

    if result_saving:
        io.save("xmap_hi.h5", xmap_hi, overwrite=True)

    return xmap_hi


def plot_indexed_results_over_signals(
    s: kp.signals.EBSD,
    xmap: orix.crystal_map.crystal_map.CrystalMap,
    sim: kp.simulations.KikuchiPatternSimulator,
    color: str = "r",
    linewidth: float = 0.5,
    remove_previous_markers=False,
):
    """
    Note that this does not work for signals that have been calibrated. To work around,
    one can recreate a signal with the same detector and detector shape, but without calibration.
    s_dummy = kp.signals.EBSD(s.data)
    and plot the markers on s_dummy instead.
    """
    if remove_previous_markers:
        del s.metadata.Markers
    s_dummy = kp.signals.EBSD(s.data)
    s_dummy.detector = s.detector
    sim_xmap = sim.on_detector(s_dummy.detector, xmap.orientations)
    markers = sim_xmap.as_markers(lines_kwargs={"linewidth": linewidth, "color": color})
    s_dummy.add_marker(markers, plot_marker=False, permanent=True)
    s_dummy.plot()

def main():
    # f = h5py.File('/work3/s203768/Real/TWIP_full_map_01.hdf5', 'r')
    # patterns = f['Scan 1']['EBSD']['Data']['RawPatterns'][:,:,:]
    # s = kp.signals.EBSD(patterns)
    # sims = torch.load('/work3/s203768/EMSoftData/torch/Ni-master-30kV-sig-0-thickness-453.pt', weights_only=False)  # Load the .pt file
    # print(sims.data[].shape)

    dict = kp.load('/work3/s203768/EMSoftData/dicts/Ni-master-30kV-sig-0-thickness-300.h5')
    f = h5py.File('/work3/s203768/Real/TWIP_full_map_01.hdf5', 'r')
    patterns = f['Scan 1']['EBSD']['Data']['RawPatterns'][:,:,:]
    print(f['Scan 1']['EBSD']['Data']['PHI'])
    print(patterns.shape)
    data = kp.signals.EBSD(patterns)
    # sims = KikuchiDataset(**sims)  # Create a dataset object
    xmap = data.dictionary_indexing(dictionary = dict, keep_n = 1)
    data.plot()
    rots = dict.xmap.rotations[xmap.simulation_indices[:]]
    eulers = rots.to_euler(degrees = True)
    print(eulers.shape)
    # print(eulers)
    print(f"Number of unique rotations found: {len(np.unique(eulers))}")
    num_lines = eulers.shape[0]
    header = f"eu\n{num_lines}\n"
    data_lines = ""
    for line in range(num_lines):
        if line % 1000 == 0:
            # idx = xmap.simulation_indices[line][0]
            # print(idx)
            im = patterns[line,:,:]
            plt.figure()
            plt.imshow(im, cmap='Grays')
            plt.savefig(f'/work3/s203768/EMSoftData/figs/image_{line}.png')
            plt.close()

        line = eulers[line, :, :]
        data_lines += str(line[0,0]) + ',' + str(line[0,1]) + ',' + str(line[0,2]) + "\n"
        
    output_text = header + data_lines

    with open("/work3/s203768/EMSoftData/angle/dict_angles.txt", "w") as f:
        f.write(output_text)

    print("/work3/s203768/EMSoftData/angle/dict_angles.txt")


if __name__ == '__main__':
    typer.run(main)
import kikuchipy as kp
import torch
from torch.nn import MSELoss

import sys
import os
from contextlib import redirect_stdout
def get_pattern_rotation(pattern, dictionary = '/work3/s203768/EMSoftData/dicts/Dict_Ni_in_use.h5'):
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
    
            dict = kp.load(dictionary)
            kp_pattern = kp.signals.EBSD(pattern)
            xmap = kp_pattern.dictionary_indexing(dictionary=dict, keep_n=1)

            rots = dict.xmap.rotations[xmap.simulation_indices[:]]

    return rots
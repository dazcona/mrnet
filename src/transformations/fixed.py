import torch
import numpy as np
import config

def get_middle_slices(array):

    if not config.FIXED_NUMBER: return array

    assert config.NUM_FIXED_SLICES % 2 == 0, 'NUM_FIXED_SLICES must be divisible by 2'

    # Middle index
    mid_index = len(array) // 2

    if config.NUM_FIXED_SLICES == 1:

        array = array[mid_index : mid_index + 1] # middle slice only
        array = np.stack((array,) * 1, axis=1)

    else:

        # half num slices
        HALF_NUM_SLICES = config.NUM_FIXED_SLICES // 2

        # slice the array
        array = array[mid_index - HALF_NUM_SLICES : mid_index + HALF_NUM_SLICES]

    array = torch.FloatTensor(array)

    return array
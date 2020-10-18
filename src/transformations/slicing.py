import numpy as np

def slice_images(array, cut='vertical'):

    if cut == 'horizontal':

        # 1. Horizontal slicing
        # print('[SLICING] Performing horizontal slicing')

        array_rotated = {}
        for i in range(len(array)):
            slice_array = array[i]
            for j in range(len(slice_array)):
                array_rotated.setdefault(j, [])
                array_rotated[j].append(
                    slice_array[j]
                )
        
        array = np.array( [ np.stack(array_rotated[key], axis=1)  for key in array_rotated.keys() ] )

    elif cut == 'diagonal':

        # 2. Diagonal slicing
        raise ValueError('[SLICING] Diagonal slicing not supported yet! Not performing any transformation')

    elif cut != 'vertical': # 'vertical' slicing is the default

        raise ValueError('{} is not supported as a way to cut the MRI slices'.format(cut))

    return array


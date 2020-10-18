import numpy as np
import config


def create_originals(number_original_images, rate_interpolation):
    """ Create array of original images with the rate of interplation """

    return { i: rate_interpolation for i in range(number_original_images)}


def create_weights(number_original_images, number_interpolated_images):
    """ Create weights for each new interpolated images """

    weights = {} 
    for i in range(number_interpolated_images):
        weights.setdefault(i, {})
        for j in range(number_original_images):
            weights[i][j] = 0
    return weights


def create_interpolated_images(array):
    """ Create interpolated images """

    if not config.INTERPOLATION: return array

    # print('[INTERPOLATION] Applying interpolation')

    # Original images
    number_original_images = len(array)

    # Rate of interpolation
    rate_interpolation = float(config.NUM_INTERPOLATED_SLICES / number_original_images)

    # Create array of original images with the rate of interplation
    originals = create_originals(number_original_images, rate_interpolation)

    # Create weights for each new interpolated images
    w = create_weights(number_original_images, config.NUM_INTERPOLATED_SLICES)

    # Populate weights
    for i in range(config.NUM_INTERPOLATED_SLICES):

        # Reset weight to distribute
        weight_to_distribute = 1

        # find the original index that still has to share weight
        j = 0
        while originals[j] == 0:
            j += 1

        # distribute the weight
        while weight_to_distribute > 0 and j < len(originals) :

            # max is 1 to distribute
            weight = min(weight_to_distribute, originals[j])

            # add it to the dictionary
            w[i][j] = weight
            originals[j] -= weight
            weight_to_distribute -= weight

            # done with this original image?
            if originals[j] == 0: j += 1

    # Use weights to create the interpolated images
    interpolated = []
    for key, values in w.items():
        inter_image = sum([ array[k] * v for k, v in values.items() if v > 0 ])
        interpolated.append(inter_image)

    return np.array(interpolated)

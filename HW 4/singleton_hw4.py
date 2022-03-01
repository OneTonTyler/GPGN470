# GPGN470 Applications of Remote Sensing
# H4: Landsat
# Due: 02 March 2022
# Tyler Singleton

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os


def extract_raw(fname, current_dir='raws/'):
    """Return a numpy array from raws

    Keyword arguments:
        fname -- filename
        current_dir -- working directory for raw files (default: .../raws/)
    """
    try:
        with open(current_dir + fname, 'rb') as file:
            return np.fromfile(file, count=-1, dtype='uint8').reshape(1500, 1500)
    except OSError:
        print('Could not find filename: ', os.getcwd() + '\\' + fname)
        print('Check file is in correct directory')


def hist_equalize(img, n_bins=256):
    """Returns an enhanced image using histogram equalization

    Keyword arguments:
        img -- image array
        n_bins -- number of bins (default: 256)
    """
    # Store image values
    img_shape = np.shape(img)
    img.ravel()

    # Create a histogram of the image as a probability density function
    density, bin_edges = np.histogram(img, bins=n_bins, range=(0, n_bins), density=True)

    # Set a range of color values proportional to the density probability
    color_value = np.cumsum(density) * 255

    # Map image color values from histogram
    enhanced_img = np.interp(img, bin_edges[:-1], color_value).reshape(img_shape)
    enhanced_img[enhanced_img > 255] = 255
    return enhanced_img


def hist_linear(img):
    """Returns an enhanced image using a linear contrast stretch

    Keyword arguments:
        img -- image array
    """
    # Find min and max values
    min_val = np.amin(img)
    max_val = np.amax(img)

    return ((img - min_val)/(max_val - min_val)) * 255


# Compose an array of all bands
bands = np.array([extract_raw(fname=f'band{i+1}c.raw') for i in range(7)])

# Testing
test_fig = hist_equalize(bands[5])

# plt.figure()
# plt.imshow(test_fig, cmap='gray')
# plt.show()

plt.figure()
plt.hist(test_fig.flatten(), 64)
plt.xlim(0,256)
plt.show()




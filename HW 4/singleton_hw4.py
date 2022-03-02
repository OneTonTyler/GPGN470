# GPGN470 Applications of Remote Sensing
# H4: Landsat
# Due: 02 March 2022
# Tyler Singleton

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os


def extract_raw(fname, current_dir='raws/'):
    """Return a 2D numpy array from raws

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
    """Returns an enhanced image using a linear contrast stretch"""
    # Find min and max values
    min_val = np.amin(img)
    max_val = np.amax(img)

    return ((img - min_val)/(max_val - min_val)) * 255


# Compose an array of all bands
bands = np.array([extract_raw(fname=f'band{i+1}c.raw') for i in range(7)])

# --- Question 2 ---

# --- Part I ---
fig, ax = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
fig.suptitle('Thermal Infrared Images from Landsat over San Diego', fontsize=16, y=0.92)

for i, func in zip(range(3), [lambda x: x, hist_linear, hist_equalize]):
    data = func(bands[5])
    im = ax[i].imshow(data, cmap='gray', clim=(0, 255))

# Setting Titles
ax[0].set_title('Original Data (0-255)')
ax[1].set_title('Linear Contrast Stretch')
ax[2].set_title('Histogram Equalization')

ax_cbar = fig.add_axes([0.25, 0.08, 0.50, 0.05])
plt.colorbar(im, orientation='horizontal', cax=ax_cbar)

plt.savefig('Enhanced_Images')
plt.show()

# --- Part II ---
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey='all')
fig.suptitle('Thermal Infrared Images Histograms from Landsat over San Diego', fontsize=16)

for i, func in zip(range(3), [lambda x: x, hist_linear, hist_equalize]):
    data = func(bands[5]).ravel()
    im = ax[i].hist(data, bins=64)

    ax[i].set_xlim(0, 255)

# Setting Titles
ax[0].set_title('Original Data (0-255)')
ax[1].set_title('Linear Contrast Stretch')
ax[2].set_title('Histogram Equalization')

ax[0].set_ylabel('Counts')
ax[1].set_xlabel('Color Value')

plt.tight_layout()
plt.savefig('Enhanced_Image_Histograms')
plt.show()


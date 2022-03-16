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

    return ((img - min_val) / (max_val - min_val)) * 255


# Band  Color           Min     Center
# --------------------------------------
# 0     Blue-Green      0.45    0.485
# 1     Green           0.52    0.56
# 2     Red             0.63    0.66
# 3     Near-IR         0.76    0.83
# 4     Mid-IR          1.55    1.65
# 5     Thermal-IR      10.40   11.45
# 6     Mid-IR          2.08    2.255

# Compose an array of all bands
bands = np.array([extract_raw(fname=f'band{i + 1}c.raw') for i in range(7)])

# --- Question 2 ---

# --- Part I ---
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True, constrained_layout=True)
fig.suptitle('Thermal Infrared Images from Landsat over San Diego', fontsize=16, y=0.92)

for i, func in zip(range(3), [lambda x: x, hist_linear, hist_equalize]):
    data = func(bands[5])
    im = ax[i].imshow(data, cmap='gray', clim=(0, 255))

# Setting Titles
ax[0].set_title('Original Data (0-255)')
ax[1].set_title('Linear Contrast Stretch')
ax[2].set_title('Histogram Equalization')

# Setting Labels
ax[0].set_ylabel('Row Index')
ax[0].set_xlabel('Column Index')
ax[1].set_xlabel('Column Index')
ax[2].set_xlabel('Column Index')

ax_cbar = fig.add_axes([0.25, 0.05, 0.50, 0.05])
plt.colorbar(im, orientation='horizontal', cax=ax_cbar)

plt.savefig('Enhanced_Images')
plt.show()

# --- Part II ---
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey='all')
fig.suptitle('Thermal Infrared Images Histograms from Landsat over San Diego', fontsize=16)

for i, func in zip(range(3), [lambda x: x, hist_linear, hist_equalize]):
    data = func(bands[5]).ravel()
    ax[i].hist(data, bins=64)
    ax[i].set_xlim(0, 255)

# Setting Titles
ax[0].set_title('Original Data (0-255)')
ax[1].set_title('Linear Contrast Stretch')
ax[2].set_title('Histogram Equalization')

ax[0].set_ylabel('Counts')
ax[0].set_xlabel('Color Value')
ax[1].set_xlabel('Color Value')
ax[2].set_xlabel('Color Value')

plt.tight_layout()
plt.savefig('Enhanced_Image_Histograms')
plt.show()

# --- Question 3 ---
rgb = np.stack((bands[2], bands[1], bands[0]))

# Plot rgb images
# Pre-stack enhancement
fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharey='all', sharex='all')
fig.suptitle('RGB Images from Landsat over San Diego', fontsize=16, y=0.98)

for i, func in zip(range(3), [lambda x: x, hist_linear, hist_equalize]):
    data_rgb = np.stack(list(map(func, rgb)), axis=2).astype(int)
    ax[0, i].imshow(data_rgb)

# Post-stack enhancement
for i, func in zip(range(3), [lambda x: x, hist_linear, hist_equalize]):
    data_rgb = func(np.stack(rgb[:], axis=2)).astype(int)
    ax[1, i].imshow(data_rgb)

# Setting Titles
ax[0, 0].set_title('Original Data (0-255)')
ax[0, 1].set_title('Linear Contrast Stretch')
ax[0, 2].set_title('Histogram Equalization')

# Setting Labels
ax[0, 0].set_ylabel('Row Index (Pre-Stack)')
ax[1, 0].set_ylabel('Row Index (Post-Stack)')

ax[1, 0].set_xlabel('Column Index')
ax[1, 1].set_xlabel('Column Index')
ax[1, 2].set_xlabel('Column Index')

plt.tight_layout()
plt.savefig('RGB_Enhanced_Images')
plt.show()

# --- Question 4 ---
# Normalized Difference Vegetation Index
# NDVI = (Near IR - Red) / (Near IR + Red)
red = bands[2].ravel().astype(float)
near_ir = bands[3].ravel().astype(float)
NDVI = np.array([(n - r) / (n + r) for n, r in zip(near_ir, red)]).reshape(1500, 1500)

# Plot Figure
plt.figure(figsize=(10, 10), constrained_layout=True)
plt.imshow(NDVI)

plt.title('Normalized Difference Vegetation Index over San Diego from Landsat', fontsize=16, y=1.02)
plt.colorbar(fraction=0.046, pad=0.04)

plt.ylabel('Row Index')
plt.xlabel('Column Index')

plt.savefig('NDVI')
plt.show()


# --- Question 5 ---
def smoothing_kernel(img, wid=9):
    """Smoothing filter"""
    smooth_filter = np.ones((wid, wid), dtype=float)*1/wid**2
    return scipy.signal.convolve2d(img, smooth_filter, mode='valid')


def sharping_kernel(img):
    """Sharpening filter"""
    identity = np.identity(3)
    lap_filter = [[1/9, 1/9, 1/9], [1/9, -20, 1/9], [1/9, 1/9, 1/9]]
    return scipy.signal.convolve2d(img, identity - lap_filter, mode='valid')


def illuminate_kernel(img):
    """Sobel filter"""
    x = [[47, 0, -47],
         [162, 0, -162],
         [47, 0, -47]]

    y = [[47, 162, 47],
         [0, 0, 0],
         [-47, -162, -47]]

    img_x = scipy.signal.convolve2d(img, x, mode='valid')
    img_y = scipy.signal.convolve2d(img, y, mode='valid')

    sobel_filter = (img_x**2 + img_y**2) ** 0.5
    return sobel_filter


fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
plt.suptitle('Filtered Images over San Diego from Landsat - Band 1', fontsize=16)

for i, func in zip(range(2), [lambda x: x, smoothing_kernel]):
    ax[0, i].imshow(func(bands[0].astype(float)), cmap='gray')

for i, func in zip(range(2), [sharping_kernel, illuminate_kernel]):
    ax[1, i].imshow(func(bands[0].astype(float)), cmap='gray')

# Setting Titles
ax[0, 0].set_title('Original Data (0-255)')
ax[0, 1].set_title('Smoothed Filter')
ax[1, 0].set_title('Sharpened Filter')
ax[1, 1].set_title('Sobel Filter')

# Setting Labels
ax[0, 0].set_ylabel('Row Index')
ax[1, 0].set_ylabel('Row Index')

ax[1, 0].set_xlabel('Column Index')
ax[1, 1].set_xlabel('Column Index')

plt.tight_layout()
plt.savefig('Filtered_Images')
plt.show()
# GPGN470 Applications of Remote Sensing
# HW 5: Classification
# Due: 16 March 2022
# Tyler Singleton

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def extract_raw(fname, current_dir='modis/'):
    """Return a 2D numpy array from raws

    Keyword arguments:
        fname       -- filename
        current_dir -- working directory for raw files (default: .../modis/)
    """
    try:
        with open(current_dir + fname, 'r') as file:
            return np.genfromtxt(file, dtype=int).ravel().reshape(400, 400)
    except OSError:
        print('Could not find filename: ', os.getcwd() + '\\' + fname)
        print('Check file is in correct directory')


def plot_subplots(row, column, data, **kwargs):
    """Automatically generate subplots from matplotlib

    Keyword arguments:
        row     -- row index
        column  -- column index
        data    -- data to be plotted
    """

    # Create figure object with subplots
    figure, axes = plt.subplots(row, column, **kwargs)

    # Plot figures on their respective subplot
    n = 0
    try:
        for x in range(row):
            for y in range(column):
                axes[x, y].imshow(data[n], cmap='gray', clim=(0, 255))
                n += 1
    except IndexError:
        pass

    # Remove empty subplots
    for axis in axes.flat:
        if not bool(axis.has_data()):
            figure.delaxes(axis)

    return figure, axes


# --- Question 1 ---
# --- Part I ---
# Loading bands into an array
band_files = [extract_raw(fname=f'modis{i + 1}.dat') for i in range(7)]

# Plotting Spectral Bands
kwarg = {'figsize': (15, 15), 'sharey': True, 'sharex': True}
fig, ax = plot_subplots(3, 3, band_files, **kwarg)
fig.suptitle('MODIS Spectral Bands \n', fontsize=16)

# Subplot Titles
ax[0, 0].set_title('Band 1 - Red')
ax[0, 1].set_title('Band 2 - Near-IR')
ax[0, 2].set_title('Band 3 - Blue')
ax[1, 0].set_title('Band 4 - Green')
ax[1, 1].set_title('Band 5 - Mid-IR')
ax[1, 2].set_title('Band 6 - Mid-IR')
ax[2, 0].set_title('Band 7 - Mid-IR')

# Axis Labels
ax[0, 0].set_ylabel('Row Index')
ax[1, 0].set_ylabel('Row Index')
ax[2, 0].set_ylabel('Row Index')

ax[2, 0].set_xlabel('Column Index')

# Color Bar
ax_cbar = fig.add_axes([0.375, 0.275, 0.605, 0.05])
norm = mpl.colors.Normalize(vmin=0, vmax=255)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='gray'), orientation='horizontal', cax=ax_cbar, label="Color Value")

plt.tight_layout()
plt.savefig('MODIS-GrayScale')
plt.show()

# --- Part II ---
# Load data
rgb = np.stack((band_files[0], band_files[3], band_files[2]), axis=2)
plt.figure(figsize=(10, 9))
plt.imshow(rgb)

# Title and labels
plt.title('MODIS RGB Composite Image - Band 1, 3, 4 \n')
plt.ylabel('Row Index')
plt.xlabel('Column Index')

plt.tight_layout()
plt.savefig('MODIS-RGB')
plt.show()


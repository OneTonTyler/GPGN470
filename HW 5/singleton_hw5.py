# GPGN470 Applications of Remote Sensing
# HW 5: Classification
# Due: 16 March 2022
# Tyler Singleton

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import Polygon

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import mahotas


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
fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(rgb)

# Title and labels
plt.suptitle('MODIS RGB Composite Image - Band 1, 3, 4 \n')
ax.set_ylabel('Row Index')
ax.set_xlabel('Column Index')

# Adding grid lines
ax.grid(color='g', linestyle='dashed', which='both')
ax.tick_params(labelright=True, labeltop=True)

# Adding boundaries for training
# Create a Polygons
cloud = np.array([[300, 300], [395, 300], [350, 250], [325, 250], [325, 200], [300, 200], [300, 300]])
cloud_poly = Polygon(cloud, linewidth=1, edgecolor='r', facecolor='none')

disintegration = np.array([[175, 215], [175, 190], [220, 210], [220, 235], [175, 225]])
dis_poly = Polygon(disintegration, linewidth=1, edgecolor='r', facecolor='none')

melt_pond = np.array([[280, 50], [290, 50], [290, 45], [280, 45], [280, 50]])
melt_poly = Polygon(melt_pond, linewidth=1, edgecolor='r', facecolor='none')

sea_ice = np.array([[125, 320], [125, 325], [110, 325], [110, 340], [125, 340], [155, 325], [155, 290],
                    [125, 275], [125, 320]])
sea_ice_poly = Polygon(sea_ice, linewidth=1, edgecolor='r', facecolor='none')

ocean = np.array([[200, 10], [175, 10], [175, 50], [200, 50], [200, 10]])
ocean_poly = Polygon(ocean, linewidth=1, edgecolor='r', facecolor='none')

ice = np.array([[300, 50], [300, 100], [325, 100], [325, 150], [300, 150], [250, 250], [250, 100], [300, 50]])
ice_poly = Polygon(ice, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(cloud_poly)
ax.add_patch(dis_poly)
ax.add_patch(melt_poly)
ax.add_patch(sea_ice_poly)
ax.add_patch(ocean_poly)
ax.add_patch(ice_poly)

plt.tight_layout()
plt.savefig('MODIS-RGB')
plt.show()


# --- Question 2 ---
def render(poly, group_number):
    """Return polygon as grid of points inside polygon."""
    xs, ys = zip(*poly)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    new_poly = [(int(x - minx), int(y - miny)) for (x, y) in poly]

    X = maxx - minx + 1
    Y = maxy - miny + 1

    grid = np.zeros((X, Y), dtype=np.int8)
    mahotas.polygon.fill_polygon(new_poly, grid)

    coords = [(x + minx, y + miny, group_number + 1) for (x, y) in zip(*np.nonzero(grid))]
    return coords


# Create an array of coordinates paired with the group number
# [x, y, group_number]
tpix = [render(group, index) for group, index in zip([cloud, disintegration, melt_pond, sea_ice, ocean, ice], range(6))]
group_names = ['cloud', 'disintegration', 'melt pond', 'sea ice', 'ocean', 'ice']

# Create 7 column matrix
# Extract band values from all 7 bands

# tpix  -> Group 1 (Cloud)
#       -> Group 2 (Disintegration)
#       -> ...
#       -> Group 6 (Ice)    -> group_member (Target for classification)
#                           -> [x, y, group_number]
#
# target -> [1, 1, 1, 2, 2, ..., 6, 6] (size of 16746 -> total_pix of group 1 + group 2 + ... group 6)
target = np.array([])
for group in tpix:
    for group_member in group:
        target = np.concatenate((target, [group_member[2]]))

# band_files    -> Band 1
#               -> Band 2
#               -> ...
#               -> Band 7   -> 400 X 400 (Spectral Value)
#
# train -> Band 1 -> Band 1[coords] = Spectral Value
spectral_value = np.array([])
for band in band_files:
    for group in tpix:
        for group_member in group:
            spectral_value = np.append(spectral_value, band[group_member[0:2]])
spectral_value = spectral_value.reshape(16746, 7)

# Target
# [1, 1, 1, 2, 2, 3, 4, ..., 6, 6] These are the groups
#
# Training
# [55, 56, 73, 34, ... 55] Band 1
# [75, 96, 83, 37, ... 25] Band 2
# [15, 26, 93, 34, ... 50] Band 3
# ...
# [18, 99, 93, 45, ... 32] Band 7
#

# --- Question 3 ---
spectral_value = np.float_(spectral_value)

# --- Question 4 ---
# Initializing algorithm with training data
# X -> 16746, 7
# y -> 16746
# Returns X_new (n_samples, n_features)
LDA = LinearDiscriminantAnalysis()
LDA.fit_transform(spectral_value, target)

# Reshaping data
# Reshaping is causing an issue, but I do not have the time to fix it
bands = np.array(band_files, dtype=float)
X = np.array([band.ravel() for band in bands]).T # (160000, 7)

# Prediction / Classification
group_pred = LDA.predict(X) # (160000)
group_pred = group_pred.reshape(400, 400)

# Setting up colors
colors = np.array(
    [
        [255,255,255],  # Clouds
        [153, 0 , 0 ],  # Disintegration
        [ 0 ,153,153],  # Melt Ponds
        [255,255, 0 ],  # Sea Ice
        [ 0 , 0 , 0 ],  # Ocean
        [ 0 ,255, 0 ]]) # Ice
# Python needs 0-1 and not 0-255
colors=colors/255.0

# Make a colormap from the list of colors
classmap = plt.matplotlib.colors.ListedColormap(colors, 'Classification', 6)

plt.figure(figsize=(10,10))
plt.imshow(group_pred, cmap=classmap, interpolation=None)
plt.show()

# --- Testing ---
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_fit = rf.fit(spectral_value, target)
rf_pred = rf.predict(X).reshape(400, 400)

plt.figure(figsize=(10,10))
plt.imshow(rf_pred, cmap=classmap, interpolation='none')
plt.show()
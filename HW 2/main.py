import math
import matplotlib.pyplot as plt
import scipy.constants as physical_const
import numpy as np

# Defining Physical Constants
A = physical_const.Wien  # 2.898 x 10^-3 K m
h = physical_const.Planck  # 6.626e^-34    J / Hz
k = physical_const.Boltzmann  # 1.381e^-23    J / K
c = physical_const.speed_of_light  # 3.00 x 10^8   m / s


def spectral_radiance(wavelength, T):
    """Returns the spectral radiance (l) in terms of wavelength"""
    try:
        return ((2 * h * c * c) / (wavelength ** 5)) / (math.exp((h * c) / (wavelength * k * T)) - 1)
    except OverflowError:
        return 0


def wiens_law(T):
    """Returns the wavelength which gives the maximum spectral radiance """
    return A / T


# Defining axis limits
temps = [290, 1250, 1800, 2600, 5000, 6000]
wavelengths = np.geomspace((10 ** -8), (10 ** -2), 200)
radiance = list()
for T in temps:
    radiance.append(np.array([spectral_radiance(x, T) for x in wavelengths]))

# Max spectral radiance
max_wavelengths = np.array([wiens_law(T) for T in temps]) * (10 ** 6)

# Plots

# Setting up the color map
color_scale = np.linspace(max_wavelengths[0], max_wavelengths[-1], len(max_wavelengths))
colors = plt.cm.turbo(color_scale / np.max(color_scale))
sm = plt.cm.ScalarMappable(cmap=plt.cm.turbo, norm=plt.Normalize(vmin=color_scale[-1], vmax=color_scale[0]))

# Legend
# Hard coded... Don't judge, the wavelength was calculated above >.<
legend = ['Earth \n290 K \n9.99 $\mu m$',
          'Lava \n1250 K \n2.32 $\mu m$',
          'Candlelight \n1800 K \n1.61 $\mu m$',
          'Incandescent Bulb \n2600 K \n1.11 $\mu m$',
          'Daytime LED Bulb \n5000 K \n0.58 $\mu m$',
          'Sun \n6000 K \n0.48 $\mu m$']

# Plotting figures
fig, ax = plt.subplots()

for i in range(len(temps)):
    plt.loglog(wavelengths * 10**6, radiance[i], color=colors[i])
plt.ylim(10, 10 ** 15)

# Color bar Settings
cbar = fig.colorbar(sm, ticks=color_scale)
cbar.ax.set_yticklabels(legend, wrap=True)
cbar.ax.invert_yaxis()

# Figure labels
plt.xlabel('Wavelength ($\mu m$)')
plt.ylabel('Spectral Radiance ($Wm^{-2}sr^{-1}m^{-1}$)')
plt.title('Blackbody Radiation Curve')

plt.grid(which='major', axis='x')
plt.tight_layout()
plt.savefig('Blackbody_Curve')
plt.show()
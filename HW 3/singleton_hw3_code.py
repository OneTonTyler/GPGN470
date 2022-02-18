# GPGN470 Applications of Remote Sensing
# H3: Orbits, Distortion, Apertures
# Due: 18 February 2022
# Tyler Singleton

import numpy as np
import matplotlib.pyplot as plt
import math

from astropy.constants import R_earth, GM_earth

# Define Constants
pi = math.pi
R = R_earth.value
GM = GM_earth.value
Le = 2 * pi / 86164.1          # Earth's Rotational Frequency

# Topex Orbital Parameters
i_topex = 66.04 * pi / 180     # Inclination of the Orbital Plane
q_topex = 9.9156 / 127         # Angular velocity ratio between rotations

# Other Satellite Parameters


# Functions
def cos(angle): return math.cos(angle)


def sin(angle): return math.sin(angle)


def orbital_frequency(a):
    """Calculates a satellite's orbital frequency at a given altitude

    w = (GM/r^3)^(1/2)
    1/T = w/2pi

    Inputs:
        a = orbital altitude
    Outputs
        orbital frequency in rads / seconds
    """
    # Convent km to m
    r = a * (10**3)
    return (math.sqrt((GM / ((r + R) ** 3))) ** -1) * 2 * pi


def ground_tracks(a, incl=i_topex, q=q_topex):
    """Returns an array of x, y, z coordinates given the full orbital period

    x = cos(t)cos(v) - cos(i)sin(t)sin(v)
    y = cos(t)sin(v) + cos(i)sin(t)cos(v)
    z = sin(i)sin(t)

    Inputs:
        a = altitude
    Outputs:
        [longitude, latitude, tracking period]
    """

    # Parameter Calculations
    dt = np.arange(-1740, 20000, 10)             # Number of measurements
    w_s = (orbital_frequency(a) ** -1) * 2 * pi  # Satellite's orbital frequency

    # Angle Calculations
    # Latitude
    theta_t = np.array([math.asin(sin(w_s*t)*sin(incl)) for t in dt])

    # Longitude
    prime = w_s * q
    cos_phi = np.array([(cos(prime*t)*cos(w_s*t)) + (sin(prime*t)*sin(w_s*t)*cos(incl)) for t in dt])
    sin_phi = np.array([(-sin(prime*t)*cos(w_s*t)) + (cos(prime*t)*sin(w_s*t)*cos(incl)) for t in dt])
    phi_t = np.array([math.atan(sin_phi[i]/cos_phi[i]) for i in range(len(dt))])

    return [phi_t, theta_t, dt]


# Plotting Figures
plt.figure(figsize=(9, 5))

# Converting radians to degrees
topex_long, topex_lat, topex_dt = ground_tracks(1336)
topex_long = topex_long * 180 / pi

# Temp variables
x = np.zeros(len(topex_long))
b = 0
for i in range(len(topex_long) - 1):
    # Add the longitudes together to get the full 0-360
    if topex_long[i+1] > topex_long[i]:
        x[i] = topex_long[i+1] + topex_long[i]
    else:
        plt.plot(x[b:i] + 180, topex_lat[b:i] * 180 / pi)
        x[i] = -180     # Resets the array
        b = i           # Start a new trace


plt.xlim(0, 360)
plt.ylim(-90, 90)
plt.show()

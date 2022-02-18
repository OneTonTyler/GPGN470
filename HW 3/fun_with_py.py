import numpy as np
import math
import matplotlib.pyplot as plt


# --- Part A --- #
x = np.array([math.sin(i)*math.cos(i) - math.cos(i)*math.cos(i) for i in np.linspace(0.1, 100, 600)])
y = np.linspace(1, 300, 300)

# meshgrid
X, Y = np.meshgrid(x, y)
Z = X * Y

c = plt.imshow(Z, cmap='gray')
plt.colorbar(c)

plt.xlabel('More Units!')
plt.ylabel('Some Units')

plt.title('Python Meshgrid with Imshow')
plt.savefig('Fun_with_Python')
plt.show()

# --- Part B --- #
x = np.zeros(1024)
y = np.zeros(1024)
X, Y = np.meshgrid(x, y)

X[256:768] = X[256:768] + 1
Z = X + Y

plt.figure()
c = plt.imshow(Z, cmap='gray')
plt.colorbar(c)
plt.show()

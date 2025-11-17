import numpy as np
import matplotlib.pyplot as plt

# Grid
n = 100
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
xx, yy = np.meshgrid(x,y)

# Test function
f = np.sin(np.pi*xx) * np.sin(np.pi*yy)

# Second-order finite difference Laplacian
dx = 1/(n-1)
L = (
    -4*f +
    np.roll(f,1,axis=0) + np.roll(f,-1,axis=0) +
    np.roll(f,1,axis=1) + np.roll(f,-1,axis=1)
) / dx**2

plt.imshow(L, cmap='plasma')
plt.colorbar()
plt.title("Discrete Laplacian of f")
plt.show()
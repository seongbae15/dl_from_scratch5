import os
import numpy as np
import matplotlib.pyplot as plt
from numpy_matrix import multivariate_normal

path = os.path.join("height_weight.txt")
xs = np.loadtxt(path)
mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False)

height = np.arange(xs[:, 0].min(), xs[:, 0].max(), 0.5)
weight = np.arange(xs[:, 1].min(), xs[:, 1].max(), 0.5)
X, Y = np.meshgrid(height, weight)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.plot_surface(X, Y, Z, cmap="viridis")

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.scatter(xs[:500, 0], xs[:500, 1])
ax2.contour(X, Y, Z)
plt.show()

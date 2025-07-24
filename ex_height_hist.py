import os
import numpy as np
import matplotlib.pyplot as plt
from norm_dist import normal

path = os.path.join("height.txt")
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)

x = np.linspace(150, 190, 1000)
y = normal(x, mu, sigma)

plt.hist(xs, bins="auto", density=True)
plt.plot(x, y)
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.show()

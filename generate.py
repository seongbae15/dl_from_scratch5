import os
import numpy as np
import matplotlib.pyplot as plt
from norm_dist import normal

path = os.path.join("height.txt")
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)
samples = np.random.normal(mu, sigma, 10000)

plt.hist(xs, bins="auto", density=True, alpha=0.7, label="original")
plt.hist(samples, bins="auto", density=True, alpha=0.7, label="generated")
plt.xlabel("Height")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

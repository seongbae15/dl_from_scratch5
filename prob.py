import os
import numpy as np
from scipy.stats import norm

path = os.path.join("height.txt")
xs = np.loadtxt(path)
mu = np.mean(xs)
sigma = np.std(xs)

p1 = norm.cdf(160, mu, sigma)
p2 = norm.cdf(180, mu, sigma)
p3 = norm.cdf(mu, mu, sigma)
print(p1)
print(1 - p2)
print(p3)

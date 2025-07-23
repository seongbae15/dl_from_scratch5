import numpy as np
import matplotlib.pyplot as plt
from norm_dist import normal


x_sums = []
N = 5

for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand()
        xs.append(x)
    t = np.sum(xs)
    x_sums.append(t)

x_norm = np.linspace(-5, 5, 1000)
mu = N / 2
sigma = np.sqrt(N / 12)
y_norm = normal(x_norm, mu, sigma)

plt.hist(x_sums, bins="auto", density=True)
plt.plot(x_norm, y_norm)
plt.title(f"N={N}")
plt.xlim(-1, 6)
plt.show()

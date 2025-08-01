import numpy as np
import matplotlib.pyplot as plt


def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return y


# x = np.linspace(-10, 10, 1000)
# y0 = normal(x, sigma=0.5)
# y1 = normal(x, sigma=1)
# y2 = normal(x, sigma=2)

# plt.plot(x, y0, label="$\sigma$=0.5")
# plt.plot(x, y1, label="$\sigma$=1")
# plt.plot(x, y2, label="$\sigma$=2")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

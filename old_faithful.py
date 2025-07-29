import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join("old_faithful.txt")
xs = np.loadtxt(path)

plt.scatter(xs[:, 0], xs[:, 1])
plt.xlabel("Eruption(Min)")
plt.ylabel("Wating(Min)")
plt.show()

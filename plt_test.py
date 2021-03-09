import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

for num in range(100):
    x.append(num)
    y.append(num)

plt.scatter(x,y)
plt.savefig("test_fig.png")
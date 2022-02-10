import numpy as np
import matplotlib.pyplot as plt

def guassian(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

f = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

x = np.linspace(0 ,100, 1000)
x_new = np.linspace(0, 100, 100)
vy = np.zeros_like(x)
y = np.zeros_like(x)
m = np.zeros_like(x)
o = np.zeros_like(x)
o_new = np.zeros_like(x_new)

for i in range(len(x)):
    vy[i] = f(x[i], 15, 4)
    y[i] = f(x[i], 25, 4)
    m[i] = f(x[i], 35, 4)
    o[i] = f(x[i], 50, 4)

for i in range(len(x_new)):
    o_new[i] = f(x_new[i], 50, 8)

plt.figure()
plt.plot(x, vy, label='VY')
plt.plot(x, y, label='Y')
plt.plot(x, m, label='M')
plt.plot(x, o, label='O')
plt.plot(x_new, o_new, 'm.-.', label='O_New')
plt.legend()
plt.show()

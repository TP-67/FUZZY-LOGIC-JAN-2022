import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Lambda Gaussian function
f = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# X-axis
x = np.linspace(0, 100, 200)
x_new = np.linspace(0, 100, 200)
alpha_cut = np.zeros_like(x)

# Y-axis
vy = np.zeros_like(x)
y = np.zeros_like(x)
m = np.zeros_like(x)
o = np.zeros_like(x)
o_new = np.zeros_like(x_new)

# Calculate Gaussian values
for i in range(len(x)):
    vy[i] = f(x[i], 15, 4)
    y[i] = f(x[i], 25, 4)
    m[i] = f(x[i], 35, 4)
    o[i] = f(x[i], 50, 4)

# New old group membership function
for i in range(len(x_new)):
    o_new[i] = f(x_new[i], 50, 8)

# Alpha-cut function
for i in range(len(x)):
    alpha_cut[i] = 1 if o[i] > 0.5 else 0

# Plot
plt.figure()
plt.plot(x, vy, label='VY')
plt.plot(x, y, label='Y')
plt.plot(x, m, label='M')
plt.plot(x, o, label='O')
plt.plot(x_new, o_new, 'm.-.', label='O_New')
plt.plot(x, alpha_cut, label='Alpha Cut')
plt.legend()
plt.show()

print('Sum vy', np.sum(vy))
print('Sum y', np.sum(y))
print('Sum m', np.sum(m))
print('Sum o', np.sum(o))
print('Sum o_new', np.sum(o_new))


if __name__ == '__main__':
    pass

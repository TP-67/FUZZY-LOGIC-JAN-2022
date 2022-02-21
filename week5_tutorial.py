import numpy as np
import matplotlib.pyplot as plt


gau = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
euc = lambda x, y: np.sqrt(np.sum(np.abs((x - y)) ** 2))
ham = lambda x, y: np.sum(np.abs(x - y))

x = np.linspace(0, 100, 200)
o = np.zeros_like(x)
vo = np.zeros_like(x)
union = np.zeros_like(x)
intersection = np.zeros_like(x)

for i in range(len(x)):
    o[i] = gau(x[i], 50, 8)
    vo[i] = gau(x[i], 60, 4)

    union[i] = np.maximum(o[i], vo[i])
    intersection[i] = np.minimum(o[i], vo[i])

# Contradiction
co = 1 - o
# Low of contradiction
loc = np.minimum(o, co)
# Low of excluded
loe = np.maximum(o, co)

print('Euc for o and vo: ', euc(o, vo))
print('Euc for neg-o and vo: ', euc(co, vo))

plt.figure(4)
plt.subplot(2, 2, 1)
plt.plot(x, o, label='O')
plt.plot(x, vo, label='VO')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, union, 'm.-.', label='U')
plt.plot(x, intersection, label='I')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, o, label='O')
plt.plot(x, loc, label='LOC')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, o, label='O')
plt.plot(x, loe, label='LOE')
plt.legend()

plt.show()


if __name__ == '__main__':
    pass

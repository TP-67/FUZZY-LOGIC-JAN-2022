import numpy as np
import matplotlib.pyplot as plt


# Membership functions
gau = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def sigmoid(x, a, b):
    """
    A:
        The magnitude of A determine how the sharpness of the S-curve.
        Positive A: increasing; Negative A: decreasing.
    B:
        Positions the center of S - curve at value B.
    """
    return 1 / (1 + np.exp(-a * (x - b)))


def dec(x, a, b):
    if x < a:
        return 1
    elif x > b:
        return 0
    else:
        return (x - b) / (a - b)


def inc(x, a, b):
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        return (x - a) / (b - a)


# Modeling
alc = np.linspace(0, 100, 200)
alc_less = np.zeros_like(alc)
alc_more = np.zeros_like(alc)

health = np.linspace(0, 50, 300)
health_good = np.zeros_like(health)
health_poor = np.zeros_like(health)

for i in range(len(alc)):
    alc_less[i] = dec(alc[i], 30, 70)
    alc_more[i] = inc(alc[i], 40, 80)

for i in range(len(health)):
    health_good[i] = sigmoid(health[i], 0.3, 25)
    health_poor[i] = sigmoid(health[i], -0.7, 25)


# Inputs/ Antecedents
input_alc = 65
input_alc_less = dec(input_alc, 30, 70)
input_alc_more = inc(input_alc, 40, 80)

# Rules Evaluation
# (1) Less -> Good
r1 = np.fmin(input_alc_less, health_good)
# (2) More -> Poor
r2 = np.fmin(input_alc_more, health_poor)

# Rules Aggregation/ Summarization (Union or Intersection)
r = np.maximum(r1, r2)

# Mamdani Defuzzification (Centroid)
cent = np.trapz(r * health, health) / np.trapz(r, health)
print('Anticipated health index (Mamdani):', cent)

# Tsukamoto Defuzzification (Weighted average)
# output = (r1 * x1 + r2 * x2 + ... + xn * rn) / (r1 + r2 + ... + rn)

# The output membership functions should be strictly monotonously increasing / decreasing.
# Not allowed Gaussian function or trapizoid function.
# This is because when we compute inverted function, we will get at least two different x-axis points, which in invalid when performing Tsukamoto defuzzification.

# Sigmoid function: y = \frac{1}{1 + e^{-a(x - b)}}
# Inverted sigmoid function: x = b + \frac{1}{a}ln[\frac{y}{1-y}]

# For S or R function, we only need to compute inverted function of the middle part.
r1_max = np.max(r1)
r2_max = np.max(r2)
x1 = 25 + 1/0.4 * np.log(r1_max / (1 - r1_max))
x2 = 25 + 1/(-0.6) * np.log(r2_max / (1 - r2_max))
x = (x1 * r1_max + x2 * r2_max) / (r1_max + r2_max)
print('Anticipated health index (Tsukamoto):', x)


# Plot
plt.figure(0)
plt.plot(alc, alc_less, label='LESS')
plt.plot(alc, alc_more, label='MORE')
plt.scatter([input_alc, input_alc], [input_alc_less, input_alc_more])
plt.legend()

plt.figure(1)
plt.plot(health, health_good, label='GOOD')
plt.plot(health, health_poor, label='POOR')
plt.fill_between(health, r1, label="R1")
plt.fill_between(health, r2, label="R2")
plt.scatter(cent, 0)
plt.legend()

plt.figure(2)
plt.fill_between(health, r, label="R1 U R2")
plt.scatter(cent, 0)
# plt.scatter([x1, x2, x], [r1, r2, 0])
plt.legend()

plt.show()


if __name__ == '__main__':
    pass
"""
The alcohol consumption of an individual is measured on a scale of 0 to 100.
The consumption is linguistically classified as less or more.

The health index of a person in measured on a scale of 0 to 50.
The health index is linguistically classified as good or poor.

Given the measure of consumption of alcohol determine the health index of the person.
"""

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
    return x, 1 / (1 + np.exp(-a * (x - b)))


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
    health_good[i] = sigmoid(health[i], 0.3, 25)[1]
    health_poor[i] = sigmoid(health[i], -0.7, 25)[1]


# Antecedents
input_alc = 45
input_alc_less = dec(input_alc, 30, 70)
input_alc_more = inc(input_alc, 40, 80)


# Rules
"""
Less -> Good
More -> Poor
"""
r1 = np.fmin(input_alc_less, health_good)
r2 = np.fmin(input_alc_more, health_poor)


# Defuzzification


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
plt.plot(health, r2, label="R2")
plt.legend()

plt.show()


if __name__ == '__main__':
    pass

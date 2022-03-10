import os
import sys
sys.path.append(os.path.abspath('../src'))

import numpy as np
import matplotlib.pyplot as plt

from mf import *


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
wind = np.linspace(0, 50, 250)
wind_slow = np.zeros_like(wind)
wind_fast = np.zeros_like(wind)

temperature = np.linspace(0, 50, 250)
temperature_low = np.zeros_like(temperature)
temperature_medium = np.zeros_like(temperature)
temperature_high = np.zeros_like(temperature)

humidity = np.linspace(0, 50, 250)
humidity_dry = np.zeros_like(humidity)
humidity_fair = np.zeros_like(humidity)
humidity_wet = np.zeros_like(humidity)

mountain = np.linspace(0, 50, 250)
mountain_hill = np.zeros_like(mountain)
mountain_huge = np.zeros_like(mountain)

raining = np.linspace(0, 50, 250)
raining_hardly = np.zeros_like(raining)
raining_certain = np.zeros_like(raining)

wind_slow_mf = ZMF(5.0, 25.0)
wind_fast_mf = SMF(10.0, 36.0)
for i in range(len(wind)):
    wind_slow[i] = wind_slow_mf.get_value(wind[i])
    wind_fast[i] = wind_fast_mf.get_value(wind[i])

temperature_low_mf = GaussianMF(3.0, 4.0)
temperature_medium_mf = GaussianMF(20.0, 4.0)
temperature_high_mf = GaussianMF(30.0, 4.0)
for i in range(len(temperature)):
    temperature_low[i] = temperature_low_mf.get_value(temperature[i])
    temperature_medium[i] = temperature_medium_mf.get_value(temperature[i])
    temperature_high[i] = temperature_high_mf.get_value(temperature[i])

humidity_dry_mf = AlterLinearMF(15.0, 1.0, 30.0, 0)
humidity_fair_mf = TriangularMF(15.0, 30.0, 45.0)
humidity_wet_mf = AlterLinearMF(30.0, 0, 45.0, 1.0)
for i in range(len(humidity)):
    humidity_dry[i] = humidity_dry_mf.get_value(humidity[i])
    humidity_fair[i] = humidity_fair_mf.get_value(humidity[i])
    humidity_wet[i] = humidity_wet_mf.get_value(humidity[i])

mountain_hill_mf = SigmoidMF(50, -0.2)
mountain_huge_mf = SigmoidMF(45, 0.7)
for i in range(len(mountain)):
    mountain_hill[i] = mountain_hill_mf.get_value(mountain[i])
    mountain_huge[i] = mountain_huge_mf.get_value(mountain[i])

raining_hardly_mf = SigmoidMF(25, -0.7)
raining_certain_mf = SigmoidMF(25, 0.3)
for i in range(len(raining)):
    raining_hardly[i] = raining_hardly_mf.get_value(raining[i])
    raining_certain[i] = raining_certain_mf.get_value(raining[i])


# Inputs/ Antecedents
input_wind = 25.0
input_wind_slow = wind_slow_mf.get_value(input_wind)
input_wind_fast = wind_fast_mf.get_value(input_wind)

input_temperature = 28.0
input_temperature_low = temperature_low_mf.get_value(input_temperature)
input_temperature_medium = temperature_medium_mf.get_value(input_temperature)
input_temperature_high = temperature_high_mf.get_value(input_temperature)

input_humidity = 33.0
input_humidity_dry = humidity_dry_mf.get_value(input_humidity)
input_humidity_fair = humidity_fair_mf.get_value(input_humidity)
input_humidity_wet = humidity_wet_mf.get_value(input_humidity)

input_mountain = 32.0
input_mountain_hill = mountain_hill_mf.get_value(input_mountain)
input_mountain_huge = mountain_huge_mf.get_value(input_mountain)

# Rules Evaluation
r1 = np.fmin(np.min([input_wind_fast, input_temperature_low]), raining_hardly)
r2 = np.fmin(np.min([input_wind_slow, input_humidity_wet]), raining_certain)
r3 = np.fmin(np.min([input_temperature_medium, input_humidity_wet]), raining_certain)
r4 = np.fmin(np.min([input_temperature_high, input_humidity_dry]), raining_hardly)
r5 = np.fmin(np.max([np.min([input_wind_slow, input_mountain_huge]), np.min([input_humidity_dry, input_mountain_hill])]), raining_hardly)
r6 = np.fmin(np.min([(1 - input_temperature_low), (1 - input_humidity_dry)]), raining_certain)

# Rules Aggregation/ Summarization (Union or Intersection)
r = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(r1, r2), r3), r4), r5), r6)

# Mamdani Defuzzification (Centroid)
cent = np.trapz(r * raining, raining) / np.trapz(r, raining)
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
r3_max = np.max(r3)
r4_max = np.max(r4)
r5_max = np.max(r5)
r6_max = np.max(r6)
x1 = 25 + 1/(-0.4) * np.log(r1_max / (1 - r1_max))
# x2 = 25 + 1/0.3 * np.log(r2_max / (1 - r2_max))
x2 = 50
x3 = 25 + 1/0.3 * np.log(r3_max / (1 - r3_max))
# x4 = 25 + 1/(-0.4) * np.log(r4_max / (1 - r4_max))
x4 = 0
# x5 = 25 + 1/(-0.4) * np.log(r5_max / (1 - r5_max))
x5 = 0
x6 = 25 + 1/0.3 * np.log(r6_max / (1 - r6_max))

x = (x1 * r1_max + x2 * r2_max + x3 * r3_max + x4 * r4_max + x5 * r5_max + x6 * r6_max) / (r1_max + r2_max + r3_max + r4_max + r5_max + r6_max)
print('Anticipated health index (Tsukamoto):', x)


# Plot
plt.figure(0)
plt.plot(wind, wind_slow, label='SLOW')
plt.plot(wind, wind_fast, label='FAST')
plt.scatter([input_wind, input_wind], [input_wind_slow, input_wind_fast])
plt.legend()
# plt.savefig('./1.jpg')

plt.figure(1)
plt.plot(temperature, temperature_low, label='LOW')
plt.plot(temperature, temperature_medium, label='MEDIUM')
plt.plot(temperature, temperature_high, label='HIGH')
plt.scatter([input_temperature, input_temperature, input_temperature], [input_temperature_low, input_temperature_medium, input_temperature_high])
plt.legend()
# plt.savefig('./2.jpg')

plt.figure(2)
plt.plot(humidity, humidity_dry, label='DRY')
plt.plot(humidity, humidity_fair, label='FAIR')
plt.plot(humidity, humidity_wet, label='WET')
plt.scatter([input_humidity, input_humidity, input_humidity], [input_humidity_dry, input_humidity_fair, input_humidity_wet])
plt.legend()
# plt.savefig('./3.jpg')

plt.figure(3)
plt.plot(raining, raining_hardly, label='HARDLY')
plt.plot(raining, raining_certain, label='CERTAIN')
plt.fill_between(raining, r1, label="R1")
plt.fill_between(raining, r2, label="R2")
plt.fill_between(raining, r3, label="R3")
plt.fill_between(raining, r4, label="R4")
plt.scatter(cent, 0)
plt.legend()
# plt.savefig('./4.jpg')

plt.figure(4)
plt.fill_between(raining, r, label="R U")
plt.scatter(cent, 0)
plt.legend()
# plt.savefig('./5.jpg')

plt.show()


if __name__ == '__main__':
    pass

"""
The IQ level of students were measured on a scale of 0 to 100. The IQ levels were classified as poor, average and good.
The parameters that affected the IQ level were as follows:
Interest in sports measured on a scale of 0 to 20. The sports interest was classified as less and more.
Interest in social media measured on a scale of 0 to 20. The social media interest was classified as light and heavy.
The following rules were derived.
more sport and light social media -> good IQ
more sport and more social media or less sport and less social media -> avg IQ
less sport and more social media -> poor IQ
Simulate the observations with a Fuzzy Model
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
sport = np.linspace(0, 20, 100)
sport_less = np.zeros_like(sport)
sport_more = np.zeros_like(sport)

media = np.linspace(0, 20, 100)
media_light = np.zeros_like(media)
media_heavy = np.zeros_like(media)

iq = np.linspace(0, 100, 300)
iq_poor = np.zeros_like(iq)
iq_avg = np.zeros_like(iq)
iq_good = np.zeros_like(iq)

for i in range(len(sport)):
    sport_more[i] = sigmoid(sport[i], 1, 12)
    sport_less[i] = sigmoid(sport[i], -2, 11)

for i in range(len(media)):
    media_heavy[i] = sigmoid(media[i], 1, 10)
    media_light[i] = sigmoid(media[i], -1, 10)

for i in range(len(iq)):
    iq_poor[i] = dec(iq[i], 30, 80)
    iq_avg[i] = gau(iq[i], 50, 7)
    iq_good[i] = inc(iq[i], 40, 70)


# Inputs/ Antecedents
input_sport = 10
input_sport_less = sigmoid(input_sport, -2, 11)
input_sport_more = sigmoid(input_sport, 1, 12)
input_media = 3
input_media_heavy = sigmoid(input_media, 1, 10)
input_media_light = sigmoid(input_media, -1, 10)

# Rules Evaluation
# (1) More sport and light social media -> good IQ
r1 = np.fmin(np.min([input_sport_more, input_media_light]), iq_good)
# (2) More sport and more social media or less sport and less social media -> avg IQ
r2 = np.fmin(np.max([np.min([input_sport_more, input_media_heavy]), np.min([input_sport_less, input_media_light])]),
             iq_avg)
# (3) Less sport and more social media -> poor IQ
r3 = np.fmin(np.min([input_sport_less, input_media_heavy]), iq_poor)

# Rule Aggregation/ Summarization
r = np.maximum(np.maximum(r1, r2), r3)

# Mamdani Defuzzification (Centroid)
cent = np.trapz(r * iq, iq) / np.trapz(r, iq)
print("Anticipated health index : ", cent)


# Plots
plt.figure(0)
plt.plot(sport, sport_less, label="LESS")
plt.plot(sport, sport_more, label="MORE")
plt.scatter([input_sport, input_sport], [input_sport_less, input_sport_more])
plt.legend()

plt.figure(1)
plt.plot(media, media_light, label="LIGHT")
plt.plot(media, media_heavy, label="HEAVY")
plt.scatter([input_media, input_media], [input_media_light, input_media_heavy])
plt.legend()

plt.figure(2)
plt.plot(iq, iq_poor, label="POOR")
plt.plot(iq, iq_avg, label="AVG")
plt.plot(iq, iq_good, label="GOOD")
plt.fill_between(iq, r1, label="R1")
plt.fill_between(iq, r2, label="R2")
plt.fill_between(iq, r3, label="R3")
plt.scatter(cent, 0)
plt.legend()

plt.figure(4)
plt.fill_between(iq, r, label="R1 U R2 U R3")
plt.scatter(cent, 0)
plt.legend()

plt.show()


if __name__ == '__main__':
    pass

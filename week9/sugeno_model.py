import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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


data = np.genfromtxt("DC.csv", delimiter=",")
x = data[:, 0]
y = data[:, 1]
x1 = x[0: 50]
x1 = np.reshape(x1, (-1, 1))
y1 = y[0: 50]
x2 = x[150: 200]
x2 = np.reshape(x2, (-1, 1))
y2 = y[150:200]
model = LinearRegression()
model1 = LinearRegression()
model.fit(x1, y1)
m1 = model.coef_[0]
c1 = model.intercept_
model.fit(x2, y2)
m2 = model.coef_[0]
c2 = model.intercept_
print("m1 ", m1, "c1 ", c1)
print("m2 ", m2, "c2 ", c2)

yp1 = m1 * x1 + c1
yp2 = m2 * x2 + c2

# Design of Membership functions for input
xdata = np.linspace(0, 15, 200)
x_small = np.zeros_like(xdata)
x_large = np.zeros_like(xdata)
for i in range(len(xdata)):
    x_small[i] = dec(xdata[i], 4, 8)
    x_large[i] = sigmoid(xdata[i], 1.5, 6)
plt.figure("mem")
plt.plot(xdata, x_small)
plt.plot(xdata, x_large)
# Given an input x predict the value of output y
input_x = 6
p1 = m1 * input_x + c1
p2 = m2 * input_x + c2
# Weighted average defuzzification
input_small = dec(input_x, 4, 8)
input_large = sigmoid(input_x, 1.5, 6)
output_y = (input_small * p2 + input_large * p1)/(input_small + input_large)

plt.figure(0)
plt.scatter(x, y)
plt.plot(x1, yp1, color=(1, 0, 0))
plt.plot(x2, yp2, color=(1, 0, 0))
plt.scatter(input_x, output_y)
plt.show()

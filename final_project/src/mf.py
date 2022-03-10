"""
A collection of membership functions.

Definition:
    Define the membership value of a input x in a fuzzy set.

Members:
    Singleton Function
    Linear Function
    Alternative Linear Function
    Rectangular Function
    Triangular Function
    Trapezoid Function
    Sigmoid Function
    Exponential Function
    Gaussian Function
    Generalized Bell Function
    Pi-shaped Function
    S-shaped Function
    Z-shaped Function

Keywords:
    Support: xs with non-zero membership value.
    Kernel: x with membership value equals to 1.
    Height: x with the largest membership value.

Note:
    In this project, we only deal with normal sets (at least one member of the set equals to 1).
    Because non-normal sets are rare in the industry.

Reference:
    https://www.mathworks.com/help/fuzzy/
"""

import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod


class MembershipFunction(ABC):

    @abstractmethod
    def get_value(self, x: float) -> float:
        ...


class SingletonMF(MembershipFunction):
    """
    Def: A single function.
        x = spike
    Args:
        x: input value.
        spike: location of a spike of the singleton function.
    """
    def __init__(self, spike: float):
        self.spike: float = spike

    def get_value(self, x: float) -> float:
        return 1.0 if x == self.spike else 0.0


class LinearMF(MembershipFunction):
    """
    Def: A clipped linear function.
        f(x) = kx + b
    Args:
        x: input value.
        k: slope.
        b: intercept on Y-axis.
    """
    def __init__(self, k: float, b: float):
        self.k: float = k
        self.b: float = b

    def get_value(self, x: float) -> float:
        y = x * self.k + self.b

        if y >= 1:
            return 1
        elif y <= 0:
            return 0
        else:
            return y


class AlterLinearMF(MembershipFunction):
    """
    Def: An alternative linear function.
    Args:
        x: input value.
        a1: x coordinate of the first inflexion.
        b1: y coordinate of the first inflexion.
        a2: x coordinate of the second inflexion.
        b2: y coordinate of the second inflexion.
    """
    def __init__(self, a1: float, b1: float, a2: float, b2: float):
        assert a1 < a2, f"{a1} >? {a2}"
        assert 0 <= b1 <= 1, "b1 is out of bounds"
        assert 0 <= b2 <= 1, "b2 is out of bounds"

        self.a1: float = a1
        self.a2: float = a2
        self.b1: float = b1
        self.b2: float = b2

    def get_value(self, x: float) -> float:
        y = ((self.b2 - self.b1) / (self.a2 - self.a1)) * (x - self.a1) + self.b1

        if self.b1 < self.b2:
            if x <= self.a1:
                return 0
            elif x >= self.a2:
                return 1
            else:
                return y
        elif self.b1 > self.b2:
            if x <= self.a1:
                return 1
            elif x >= self.a2:
                return 0
            else:
                return y


class RectangularMF(MembershipFunction):
    """
    Def: A rectangular function.
    Args:
        x: input value.
        a: x coordinate of foot.
        b: x coordinate of shoulder.
    """
    def __init__(self, a: float, b: float):
        assert a < b, f"{a} >? {b}"

        self.a: float = a
        self.b: float = b

    def get_value(self, x: float) -> float:
        if self.a <= x <= self.b:
            return 1
        else:
            return 0


class TriangularMF(MembershipFunction):
    """
    Def: A triangular function.
    Args:
        x: input value.
        a: left-foot.
        b: left-shoulder.
        c: right-shoulder.
    """
    def __init__(self, a: float, b: float, c: float):
        assert a <= b <= c, "Wrong order"

        self.a: float = a
        self.b: float = b
        self.c: float = c

    def get_value(self, x: float) -> float:
        if x <= self.a:
            return 0
        elif self.a <= x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b <= x <= self.c:
            return (self.c - x) / (self.c - self.b)
        elif self.c <= x:
            return 0


class TrapezoidMF(MembershipFunction):
    """
    Def: A trapezoid function.
    Args:
        x: input value.
        a: left-foot.
        b: left-shoulder.
        c: right-shoulder.
        d: right-foot.
    """
    def __init__(self, a: float, b: float, c: float, d: float):
        assert a < b < c < d, "Wrong order"

        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d

    def get_value(self, x: float) -> float:
        if x < self.a:
            return 0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x < self.c:
            return 1
        elif self.c < x < self.d:
            return (self.d - x) / (self.d - self.c)
        elif x > self.d:
            return 0


class SigmoidMF(MembershipFunction):
    """
    Def: A sigmoid function.
        f(x, a, b) = \frac{1}{1 + e^{-b(x - a )}}
    Args:
        x: input value.
        a: min-point of sigmoid function.
        b: steepness.
    """
    def __init__(self, a: float, b: float):
        self.a: float = a
        self.b: float = b

    def get_value(self, x: float) -> float:
        return 1 / (1 + np.exp(-self.b * (x - self.a)))


class ExponentialMF(MembershipFunction):
    """
    Def: A exponential function.
    Args:
        x: input value.
        k: steepness.
    """
    def __init__(self, k: float):
        assert k > 0, "k must > 0"

        self.k: float = k

    def get_value(self, x: float) -> float:
        assert x >= 0, "x must >= 0"

        return 1 / np.exp(self.k * x)


class GaussianMF(MembershipFunction):
    """
    Def: A gaussian function.
        f(x; \mu, \sigma) = e^{\frac{-(x - c)^2}{2 \sigma^2}}
    Args:
        x: input value
        mu: mean
        sigma: standard deviation
    """
    def __init__(self, mu: float, sigma: float):
        self.mu: float = mu
        self.sigma: float = sigma

    def get_value(self, x: float) -> float:
        return np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))


class GeneralizedBellMF(MembershipFunction):
    """
    Def: A generalized bell function.
        f(x; a, b, c) = \frac{1}{1 + \abs{\frac{x - c}{a}}^{2b}}
    Args:
        x: input value.
        a: width of the membership function. The larger, the wider.
        b: shape of curve on either side of the central plateau. The larger, the more steep transition.
        c: center of the function.
    """
    def __init__(self, a: float, b: float, c: float):
        # assert a < b < c, "Wrong order"

        self.a: float = a
        self.b: float = b
        self.c: float = c

    def get_value(self, x: float) -> float:
        tmp = ((x - self.c) / self.a) ** 2
        if tmp == 0 and self.b == 0:
            y = 0.5
        elif tmp == 0 and self.b < 0:
            y = 0
        else:
            tmp = tmp ** self.b
            y = 1 / (1 + tmp)

        return y


class PIMF(MembershipFunction):
    """
    Def: A pi-shaped function (This membership function is the product of a s-shape function and a z-shape function).
        f(x; a, b, c, d) = 0, if x <= a;
                         = 2 * (\frac{x - a}{b - a})^2, if a <= x <= \frac{a + b}{2}
                         = 1 - 2 * (\frac{x - b}{b - a})^2, if \frac{a + b}{2} <= x <= b
                         = 1, if b <= x <= c
                         = 1 - 2 * (\frac{x - c}{d - c})^2, if c <= x <= \frac{c + d}{2}
                         = 2^(\frac{x - d}{d - c})^2, if \frac{c + d}{2} <= x <= d
                         = 0, if x >= d
    Args:
        x: input value
        a: left-foot
        b: left-shoulder
        c: right-shoulder
        d: right-foot
    """
    def __init__(self, a: float, b: float, c: float, d: float):
        assert a < b < c < d, "Wrong order"

        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d

    def get_value(self, x: float) -> float:
        s = SMF(self.a, self.b)
        z = ZMF(self.c, self.d)

        return s.get_value(x) * z.get_value(x)


class SMF(MembershipFunction):
    """
    Def: A s-shaped function.
        s(x; a, b) = 0, if x <= a;
                   = 2 * (\frac{x - a}{b - a})^2, if a <= x <= \frac{a + b}{2}
                   = 1 - 2 * (\frac{x - b}{b - a})^2, if \frac{a + b}{2} <= x <= b
                   = 1, if x >= b
    Args:
        x: input value
        a: foot
        b: shoulder
    """
    def __init__(self, a: float, b: float):
        assert a < b, f"{a} >? {b}"

        self.a: float = a
        self.b: float = b

    def get_value(self, x: float) -> float:
        if x <= self.a:
            return 0
        elif self.a <= x <= (self.a + self.b) / 2:
            return 2 * ((x - self.a) / (self.b - self.a)) ** 2
        elif (self.a + self.b) / 2 <= x <= self.b:
            return 1 - 2 * ((x - self.b) / (self.b - self.a)) ** 2
        elif x >= self.b:
            return 1


class ZMF(MembershipFunction):
    """
    Def: A z-shaped function.
        z(x; a, b) = 1, if x <= a
                   = 1 - 2 * (\frac{x - a}{b - a})^2, if a <= x <= \frac{a + b}{2}
                   = 2 ^ (\frac{x - b}{b - a})^2, if \frac{a + b}{2} <= x <= b
                   = 0, if x >= b
    Args:
        x: input value
        a: shoulder
        b: foot
    """
    def __init__(self, a: float, b: float):
        assert a < b, f"{a} >? {b}"

        self.a: float = a
        self.b: float = b

    def get_value(self, x: float) -> float:
        if x <= self.a:
            return 1
        elif self.a <= x <= (self.a + self.b) / 2:
            return 1 - 2 * ((x - self.a) / (self.b - self.a)) ** 2
        elif (self.a + self.b) / 2 <= x <= self.b:
            return 2 * ((x - self.b) / (self.b - self.a)) ** 2
        elif x >= self.b:
            return 0

"""
Defuzzification using centroid (`center of gravity`) method and average maximum method.
"""

import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod

from inv_mf import *


class Defuzzifier(ABC):

    @abstractmethod
    def get_value(self, x: List) -> float:
        ...


class Mamdani(Defuzzifier):
    """
    Mamdani Defuzzification (Centroid)
    """
    def __init__(self,
                 out_min: int,
                 out_max: int
                 ):
        # Range of the output fuzzy variable.
        self.out_min = out_min
        self.out_max = out_max
        assert out_min < out_max, f"{out_min} >? {out_max}"
        self.resolution = int(out_max - out_min) * 5

    def get_value(self, x: List) -> float:
        out_list = np.linspace(self.out_min, self.out_max, self.resolution)
        x = np.array(x)
        cent = np.trapz(x * out_list, out_list) / np.trapz(x, out_list)

        return cent


class Tsukamoto(Defuzzifier):
    """
    Tsukamoto Defuzzification (Average maximum)

    Format:
        output = (r1 * x1 + r2 * x2 + ... + xn * rn) / (r1 + r2 + ... + rn)

    Note:
        (1) The output membership functions should be strictly monotonously increasing / decreasing.
        (2)This is because when we compute inverted function, we will get at least two different x-axis points, which in invalid when performing Tsukamoto defuzzification.

    """
    def __init__(self, inv_fun: List[InverseMembershipFunction]):
        self.inv_fun: List[InverseMembershipFunction] = inv_fun

    def get_value(self, y: List, min_value: float, max_value: float) -> float:
        fuzzy_max = []
        inv_fuzzy_max = []
        for i in range(len(y)):
            fuzzy_max.append(np.max(y[i]))
            inv_fuzzy_max.append(self.inv_fun[i].get_value(np.max(y[i]), min_value, max_value))

        numerator, denominator = 0, 0
        for i in range(len(y)):
            numerator += inv_fuzzy_max[i] * fuzzy_max[i]
            denominator += fuzzy_max[i]
        cent = numerator / denominator

        return cent

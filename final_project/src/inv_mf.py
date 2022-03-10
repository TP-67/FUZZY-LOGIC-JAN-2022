"""
A collection of inverse membership functions.

Definition:
    Define the inverse membership value of a input y in defuzzification process.

Members:
    Inverse Sigmoid Function

Note:
    The output membership functions should be strictly monotonously increasing / decreasing.
"""

import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod


class InverseMembershipFunction(ABC):

    @abstractmethod
    def get_value(self, x: float) -> float:
        ...


class InverseSigmoidMF(InverseMembershipFunction):
    """
    Def: An inverse sigmoid function.
        x = a + \frac{1}{b}ln[\frac{y}{1-y}]
    Args:
        x: input value.
        a: min-point of sigmoid function.
        b: steepness.
    """
    def __init__(self, a: float, b: float):
        self.a: float = a
        self.b: float = b

    def get_value(self, y: float, min_value: float, max_value: float) -> float:
        """
        Here, we need to deal with the 'divide by zero' problem.
        Considering the shape of sigmoid function, we solve this problem from two aspects. For instance,
            - If y == 0, and b > 0, then we return lower bound of the output variable because that is where sigmoid function approaching to 0.
            - If y == 0, and b < 0, then we return upper bound of the output variable for the same reason.
        """
        # if (y / (1 - y)) == 0:
        #     return 0
        if y == 0:
            if self.b > 0:
                return min_value
            else:
                return max_value
        elif y == 1:
            if self.b > 0:
                return max_value
            else:
                return min_value

        return self.a + 1 / self.b * np.log(y / (1 - y))

import os
import sys
sys.path.append(os.path.abspath('../src'))

"""
Unit test for membership functions
"""

import unittest

from mf import *


class TestSingletonMF(unittest.TestCase):
    def test_mf_singleton(self):
        self.assertEqual(SingletonMF(2).get_value(1), 0)


class TestLinearMF(unittest.TestCase):
    def test_mf_linear(self):
        self.assertEqual(LinearMF(2, 0.5).get_value(1), 1.0)


class TestAlterLinearMF(unittest.TestCase):
    def test_mf_alter_linear(self):
        self.assertEqual(AlterLinearMF(10, 1.0, 15, 0).get_value(1), 1.0)


class TestRectangularMF(unittest.TestCase):
    def test_mf_rectangular(self):
        self.assertEqual(RectangularMF(2, 5).get_value(3), 1.0)


class TestTriangularMF(unittest.TestCase):
    def test_mf_triangular(self):
        self.assertEqual(TriangularMF(2, 5, 7).get_value(6), 0.5)


class TestTrapezoidMF(unittest.TestCase):
    def test_mf_trapezoid(self):
        self.assertEqual(TrapezoidMF(1, 2, 3, 4).get_value(2.5), 1.0)


class TestSigmoidMF(unittest.TestCase):
    def test_mf_sigmoid(self):
        self.assertEqual(SigmoidMF(25, -0.7).get_value(25), 0.5)


class TestExponentialMF(unittest.TestCase):
    def test_mf_exponential(self):
        self.assertEqual(ExponentialMF(0.3).get_value(0), 1)


class TestGaussianMF(unittest.TestCase):
    def test_mf_gaussian(self):
        self.assertEqual(GaussianMF(10, 0.1).get_value(1), 0)


if __name__ == '__main__':
    unittest.main()

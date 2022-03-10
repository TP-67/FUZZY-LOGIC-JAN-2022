import os
import sys
sys.path.append(os.path.abspath('../src'))

import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from mf import *


# Load configuration
with open('../config.yml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# Create output folder
output_dir = os.path.join(config['output_path'], 'preview_mf')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Define plot function
def plot_mf_for_preview(x: List, y: List, name: str) -> None:
    plt.figure(0)
    plt.style.use('seaborn-darkgrid')
    plt.style.context('ggplot')
    plt.plot(x, y, label=name)
    plt.xlabel('x')
    plt.ylabel('fuzzy value')
    plt.title('Membership Functions of ' + name)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'membership_functions_' + name + '.jpg'))
    plt.show()


# Build steady x axis
x = np.linspace(0, 100, 500)


# Linear Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = LinearMF(0.045, -1.5).get_value(x[i])
plot_mf_for_preview(x, y, 'Linear')


# Alternative Linear Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = AlterLinearMF(30, 1.0, 65, 0).get_value(x[i])
plot_mf_for_preview(x, y, 'AlterLinear')


# Rectangular Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = RectangularMF(30, 65).get_value(x[i])
plot_mf_for_preview(x, y, 'Rectangular')


# Triangular Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = TriangularMF(40, 65, 85).get_value(x[i])
plot_mf_for_preview(x, y, 'Triangular')


# Trapezoid Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = TrapezoidMF(30, 45, 75, 80).get_value(x[i])
plot_mf_for_preview(x, y, 'Trapezoid')


# Sigmoid Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = SigmoidMF(55, -0.4).get_value(x[i])
plot_mf_for_preview(x, y, 'Sigmoid')


# Exponential Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = ExponentialMF(0.1).get_value(x[i])
plot_mf_for_preview(x, y, 'Exponential')


# Gaussian Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = GaussianMF(40, 9).get_value(x[i])
plot_mf_for_preview(x, y, 'Gaussian')


# GeneralizedBell Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = GeneralizedBellMF(30, 8, 60).get_value(x[i])
plot_mf_for_preview(x, y, 'GeneralizedBell')


# PI-shaped Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = PIMF(30, 50, 60, 80).get_value(x[i])
plot_mf_for_preview(x, y, 'PI')


# S-shaped Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = SMF(40, 70).get_value(x[i])
plot_mf_for_preview(x, y, 'S')


# Z-shaped Membership function
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = ZMF(40, 70).get_value(x[i])
plot_mf_for_preview(x, y, 'Z')

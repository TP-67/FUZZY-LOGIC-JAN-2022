import os

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

from variable import *

colors = ['red',
          'blue',
          'yellow',
          'green',
          'purple']


def plot_mf(variable: FuzzyVariable,
            sx: float,
            x_min: int,
            x_max: int,
            res: int,
            config: Dict):
    x = np.linspace(x_min, x_max, res)
    y_list = []
    for i in variable.domain:
        y_list.append(list(map(i.mf.get_value, x)))

    plt.figure(0)
    plt.style.use('seaborn-darkgrid')
    plt.style.context('ggplot')
    for i in range(len(y_list)):
        plt.plot(x, y_list[i], label=variable.domain[i].name)
        plt.scatter(sx, variable.domain[i].mf.get_value(sx))
        plt.vlines(sx, 0,  variable.domain[i].mf.get_value(sx), linestyle="dashed")
        plt.text(sx, variable.domain[i].mf.get_value(sx), '({}, {})'.format(sx, variable.domain[i].mf.get_value(sx)), fontsize=8)

    plt.xlabel('x')
    plt.ylabel('fuzzy value')
    plt.title('Membership Functions of ' + variable.name)
    plt.legend()
    # Save image
    plt.savefig(os.path.join(config['output_path'], 'membership_functions_' + variable.name + '.jpg'))
    plt.show()


def plot_conclusions(variable: FuzzyVariable,
                     x_min: int,
                     x_max: int,
                     res: int,
                     y: List,
                     config: Dict):
    x = np.linspace(x_min, x_max, res)
    for i in variable.domain:
        # y_list.append(list(map(i.mf.get_value, x)))
        plt.plot(x, list(map(i.mf.get_value, x)), label=i.name)

    plt.figure(1)
    plt.style.use('seaborn-darkgrid')
    plt.style.context('ggplot')

    for i in range(len(y)):
        # plt.plot(x, y[i], label='LESS' + str(i))
        plt.plot(x, y[i])
        plt.fill_between(x, y[i], alpha=0.3, label='rule' + str(i))

    plt.xlabel('x')
    plt.ylabel('fuzzy value')
    plt.title('Fuzzy Rules')
    plt.legend()
    # Save image
    plt.savefig(os.path.join(config['output_path'], 'fuzzy_rules.jpg'))
    plt.show()


def plot_fuzzy_result(x_min: int,
                      x_max: int,
                      res: int,
                      y: List,
                      config: Dict):
    x = np.linspace(x_min, x_max, res)

    plt.figure(2)
    plt.style.use('seaborn-darkgrid')
    plt.style.context('ggplot')
    plt.plot(x, y, label='rule')
    plt.xlabel('x')
    plt.ylabel('fuzzy value')
    plt.title('Aggregation Rule')
    plt.legend()
    # Save image
    plt.savefig(os.path.join(config['output_path'], 'aggregation_rule.jpg'))
    plt.show()


def plot_fuzzy_result_with_center(x_min: int,
                                  x_max: int,
                                  res: int,
                                  y: List,
                                  cent: float,
                                  config: Dict):
    x = np.linspace(x_min, x_max, res)

    plt.figure(3)
    plt.style.use('seaborn-darkgrid')
    plt.style.context('ggplot')
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3, label='rule')
    plt.scatter(cent, 0)
    plt.xlabel('x')
    plt.ylabel('fuzzy value')
    plt.title('Aggregation Rule with Centroid')
    plt.legend()
    # Save image
    plt.savefig(os.path.join(config['output_path'], 'aggregation_rule_with_centroid.jpg'))
    plt.show()


def alpha(y, floor_clip=None, ceiling_clip=None):
    """
    Def: Alpha-cut function for value clipping
    Args:
        y: input membership value of range [0, 1]
        floor_clip: lower bound of clipping
        ceiling_clip: upper bound of clipping
    """
    assert 0 <= y <= 1, "y is out of bounds"

    floor_clip = 0 if floor_clip is None else floor_clip
    ceiling_clip = 1 if ceiling_clip is None else ceiling_clip

    if y >= ceiling_clip:
        return ceiling_clip
    elif y <= floor_clip:
        return floor_clip
    else:
        return y


def check_valid(*variables) -> bool:
    for i in variables:
        if i[0] > i[1]:
            return False
    return True


def write_text(config: Dict, *text_str) -> None:
    with open(os.path.join(config['output_path'], 'defuzzification_results.txt'), 'w') as text_file:
        for i in text_str:
            text_file.write(i)
            text_file.write('\n')

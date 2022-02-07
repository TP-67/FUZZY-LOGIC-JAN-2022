import matplotlib.pyplot as plt
import numpy as np


def func_one(x: float, a: int, b: int) -> (float, float):
    if x <= a:
        return x, 0
    elif x >= b:
        return x, 1
    elif a < x < b:
        return x, float((x - a) / (b - a))
    else:
        raise ValueError


def func_two(x: int, a: int, b: int, c: int, d: int) -> (float, float):
    if x < a or x > d:
        return x, 0
    elif a <= x <= b:
        return x, (x - a) / (b - a)
    elif b < x < c:
        return x, 1
    elif c <= x <= d:
        return x, (d - x) / (d - c)
    else:
        raise ValueError


def plot_func(x: list, y: list):
    plt.figure()
    plt.plot(x, y)
    plt.title('Piecewise Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    # Define constants
    a, b = 4, 8
    c, d, e, f = 3, 5, 8, 9

    # Sample points
    temp = np.linspace(0, 10, 200)

    # Initialize plot
    fig, ax = plt.subplots()

    # Initialize container
    x_list = []
    y_list = []
    for i in temp:
        (x, y) = func_one(i, a, b)
        x_list.append(x)
        y_list.append(y)

    # Plot line 1
    ax.plot(x_list, y_list, label='Func1')

    # Initialize container
    x_list = []
    y_list = []
    for i in temp:
        (x, y) = func_two(i, c, d, e, f)
        x_list.append(x)
        y_list.append(y)

    # Plot line 2
    ax.plot(x_list, y_list, label='Func2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Piecewise Functions')
    # Automatically check elements that need to be shown
    ax.legend()

    # Visualization
    plt.show()

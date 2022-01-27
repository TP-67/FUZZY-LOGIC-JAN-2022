import numpy as np
import matplotlib.pyplot as plt


def myfunc(x: float, a: int, b: int) -> (float, float):
    if x <= a:
        return x, 0
    elif x >= b:
        return x, 1
    elif a < x < b:
        return x, float((x - a) / (b - a))
    else:
        raise ValueError


def myplot(x: list, y: list):
    plt.figure()
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # Define constants
    a = 3
    b = 7
    c = np.linspace(0, 10, 100)

    # Initialize container
    x_list = []
    y_list = []

    for i in c:
        (x, y) = myfunc(i, a, b)
        x_list.append(x)
        y_list.append(y)
    myplot(x_list, y_list)

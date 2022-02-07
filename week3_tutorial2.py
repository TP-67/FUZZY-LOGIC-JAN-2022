import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, a, b):
    """
    A:
        The magnitude of A determine how the sharpness of the S-curve.
        Positive A: increasing; Negative A: decreasing.
    B:
        Positions the center of S - curve at value B.
    """
    return x, 1 / (1 + np.exp(-a * (x - b)))


def plot_func(x: list, y: list, z: list, k: list, m: list, n: list, p: list, q: list, u: list, v: list):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='sigmoid-1-0')
    ax.plot(z, k, label='sigmoid-1-N5')
    ax.plot(m, n, label='sigmoid-N1-0')
    ax.plot(p, q, label='sigmoid-5-0')
    ax.plot(u, v, label='sigmoid-0.5-0')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Sigmoid Functions')
    # Automatically check elements that need to be shown
    ax.legend()

    # Visualization
    plt.show()


if __name__ == '__main__':
    # Define constants
    x, y = 1, 0
    z, k = 1, -5
    m, n = -1, 0
    p, q = 5, 0
    u, v = 0.5, 0

    # Sample points
    temp = np.linspace(-10, 10, 200)

    # Initialize container
    x_list = []
    y_list = []
    z_list = []
    k_list = []
    m_list = []
    n_list = []
    p_list = []
    q_list = []
    u_list = []
    v_list = []

    for i in temp:
        (s1, c1) = sigmoid(i, x, y)
        (s2, c2) = sigmoid(i, z, k)
        (s3, c3) = sigmoid(i, m, n)
        (s4, c4) = sigmoid(i, p, q)
        (s5, c5) = sigmoid(i, u, v)

        x_list.append(s1)
        y_list.append(c1)
        z_list.append(s2)
        k_list.append(c2)
        m_list.append(s3)
        n_list.append(c3)
        p_list.append(s4)
        q_list.append(c4)
        u_list.append(s5)
        v_list.append(c5)

    plot_func(x_list, y_list, z_list, k_list, m_list, n_list, p_list, q_list, u_list, v_list)

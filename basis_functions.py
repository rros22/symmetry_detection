"""
    Basis functions to construct the characteristic functions of the symmetry generators.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval

"""
    Monomial basis functions and their derivatives.
"""

def monomial_bf(x, u, m, n):
    return x**m * u**n

def monomial_bf_dx(x, u, m, n):
    return m * x**(m - 1) * u**n

def monomial_bf_du(x, u, m, n):
    return n * x**m * u**(n - 1)

"""
    Chebyshev basis functions and their derivatives.
"""

def T(n, x):
    c = np.zeros(n + 1)
    c[n] = 1
    return chebval(x, c)


def dT(n, x):
    if n == 0:
        return np.zeros_like(x)

    c = np.zeros(n + 1)
    c[n] = 1

    dc = chebder(c)

    return chebval(x, dc)


def chebyshev_bf(x, u, m, n):
    return T(m, x) * T(n, u)


def chebyshev_bf_dx(x, u, m, n):
    return dT(m, x) * T(n, u)


def chebyshev_bf_du(x, u, m, n):
    return T(m, x) * dT(n, u)


# Test chevyshev polynomials
if __name__ == "__main__":
    """
        Plot chevyshev basis ranging with m, n ranging from 0 to 3. Make a subplot 3d surface for each combination of m, n.
    """
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 100)
    u = np.linspace(-1, 1, 100)
    X, U = np.meshgrid(x, u)

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), subplot_kw={"projection": "3d"})
    for m in range(4):
        for n in range(4):
            ax = axs[m, n]
            Z = chebyshev_bf(X, U, m, n)
            ax.plot_surface(X, U, Z, cmap="viridis")
            ax.set_title(f"T({m}, x) * T({n}, u)")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_zlabel("T(m, x) * T(n, u)")


    """
     In a new figure, plot the monomial basis functions with m, n ranging from 0 to 3. Make a subplot 3d surface for each combination of m, n.
    """
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), subplot_kw={"projection": "3d"})
    for m in range(4):
        for n in range(4):
            ax = axs[m, n]
            Z = monomial_bf(X, U, m, n)
            ax.plot_surface(X, U, Z, cmap="viridis")
            ax.set_title(f"x^{m} * u^{n}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_zlabel("x^m * u^n")
    plt.tight_layout()
    plt.show()


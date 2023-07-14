import numpy as np


def fixed_point(a, b, x0=None, alpha=1.0, tol=1e-9, maxiter=1e6, callback=None, atol=None):
    """
    The fixed point iteration to solve ax = b as the sum_p (1 - a)^p @ b.
    This only converges if ||1-a|| < alpha.

    :param a: The linear operator to invert.
    :param b: The right hand side.
    :param x0: The optional start value
    :param alpha: Optional scaling of the residue in each step. This can be used in case ||1 - a|| >= 1.
    :param tol: The relative tolarance.
    :param maxiter: The maximum number of iterations
    :param callback: The optional callback called after each iteration.
    :param atol: The optional absolute tolarance. This overrides the relative tolerance.

    :return: (x, info) The solution x to Ax = b, and an integer info which should be 0 after convergence.
    """
    if atol is not None:
        tol = atol / np.linalg.norm(b)

    dx = b
    if x0 is not None:
        dx -= a @ x0
    x = alpha * dx
    it_nb = 1
    while (maxiter is None or it_nb < maxiter) and tol < np.linalg.norm(dx):
        dx = b - a @ x
        x += alpha * dx
        it_nb += 1

    if maxiter is None or it_nb < maxiter:
        info = 0
    else:
        info = maxiter

    if callback is not None:
        callback()

    return x, info

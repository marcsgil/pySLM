import numpy as np


def kaczmarz(matrix: np.ndarray, b: np.ndarray, nb_iterations=100, x_0=None, relaxation_parameter=1):
    """
    Solves a linear set of equations iteratively using the Algebraic Reconstruction Technique or Kaczmarz iteration

    :param matrix: The matrix of the linear problem.
    :param b: The right-hand side of the problem.
    :param nb_iterations: The number of iteratiosn to perform.
    :param x_0: An optional start vector.
    :param relaxation_parameter:
    :return: x, the result vector
    """
    nb_equations, nb_variables = matrix.shape

    # precalculate the equation sqd weights
    weights = relaxation_parameter / np.sum(np.abs(matrix) ** 2, axis=1)

    if x_0 is None:
        x_0 = np.zeros(matrix.shape[1])

    # do the Kaczmarz iteration
    x = x_0
    for it_idx in range(nb_iterations):
        for eqn_idx in range(nb_equations):  # should actually be chosen randomly with a distribution proportional to equationSqdNorms
            v = matrix[eqn_idx]
            x += (b[eqn_idx] - np.dot(v, x)) * weights[eqn_idx] * v
    
    return x

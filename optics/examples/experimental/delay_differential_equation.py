import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from optics import log


class Identity(LinearOperator):
    def __init__(self, shape, dtype):
        super().__init__(shape=shape, dtype=None)
    def _matvec(self, _):
        return _
    def _rmatvec(self, _):
        return _

class HermitianOperator(LinearOperator):
    def __init__(self, shape, dtype, matvec=None):
        super().__init__(shape=shape, dtype=dtype)
        self.__matvec = matvec
    def _matvec(self, _):
        return self.__matvec(_)
    def _rmatvec(self, _):
        return self @ _

class SkewHermitianOperator(LinearOperator):
    def __init__(self, shape, dtype, matvec=None):
        super().__init__(shape=shape, dtype=dtype)
        self.__matvec = matvec
    def _matvec(self, _):
        return self.__matvec(_)
    def _rmatvec(self, _):
        return self -self.__matvec(_).conj()

class DelayOperator(LinearOperator):
    def __init__(self, shape, dtype, delay_factor: float = 1.0):
        super().__init__(shape=shape, dtype=dtype)
        self.__delay_factor = delay_factor
    def _matvec(self, mat):
        if mat.ndim == 2:
            for _, vector in enumerate(mat.transpose()):
                mat[:, _] = self._matvec(vector)
            return mat
        else:
            t_range = np.arange(self.shape[0])
            return np.interp(self.__delay_factor * t_range, t_range, mat) / np.sqrt(self.__delay_factor)
    def _rmatvec(self, _):
        raise NotImplemented


def pantograph_equation():
    # Pantograph delay differential equation
    a = 10  # weight of direct derivative
    b = 5  # 20  weight of delayed derivative
    delay_factor = 0.5  # delay factor
    # d = lambda x: lambda t: a * x(t) + b * x(l * t)

    dt = 0.1  # sampling step
    t_init = 1  # Initial time period
    t_total = 5.0  # Time period to display
    t_range = np.arange(0, t_total, dt)

    # Initial conditions
    # The initial function on time interval [0, t_init]
    f_init = lambda _: 0.1 + np.exp(-(0.5 * (_-0.85) / 0.02)**2) - 0.5 * np.exp(-(0.5 * (_-0.80) / 0.05) ** 2)

    log.info(f'Solving df(t)/dt = {a:0.1f} f(t) + {b:0.1f} f({delay_factor:0.1f}t) for f(t) on interval [{t_init}s, {t_total}s]')

    def discrete_derivative(mat: np.ndarray) -> np.ndarray:
        if mat.ndim == 2:
            for _, vector in enumerate(mat.transpose()):
                mat[:, _] = discrete_derivative(vector)
            return mat
        else:
            d = np.diff(mat, axis=0, prepend=0, append=0)
            return (d[:-1] + d[1:]) / (2 * dt)
    def discrete_delay(mat: np.ndarray) -> np.ndarray:
        if mat.ndim == 2:
            for _, vector in enumerate(mat.transpose()):
                mat[:, _] = discrete_delay(vector)
            return mat
        else:
            return np.interp(delay_factor * t_range, t_range, mat) / np.sqrt(delay_factor)

    dtype = np.complex128
    I = Identity(shape=[t_range.size, t_range.size], dtype=dtype)
    D = SkewHermitianOperator(shape=[t_range.size, t_range.size], dtype=dtype, matvec=discrete_derivative)
    Λ = DelayOperator(shape=[t_range.size, t_range.size], dtype=dtype, delay_factor=delay_factor)

    a_mean = np.mean(a)
    r = 0.95 / (np.amax(np.abs(a - a_mean)) + np.amax(np.abs(b)))

    A = 1/r * (D + a * I +  b * Λ)
    L = 1/r * (D + a_mean * I)
    V = A - L  # = 1/r * ((a - a_mean) * I +  b * Λ) = 1/r * b * Λ
    LpI = L + I
    B = A - LpI
    LpI_inv = aslinearoperator(np.linalg.inv((L + I) @ np.eye(I.shape[1])))
    Γ_inv = B @ LpI_inv

    M = I - Γ_inv @ A

    A_mat = A @ np.eye(I.shape[1])
    V_mat = V @ np.eye(I.shape[1])
    Λ_mat = Λ @ np.eye(I.shape[1])
    M_mat = M @ np.eye(I.shape[1])
    w, v = np.linalg.eig(V_mat)
    # u, s, vh = np.linalg.svd(M_mat)

    log.info('Displaying...')
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(w.real, w.imag)
    axs[1].imshow(Λ_mat)
    # ax.plot(t_range, x0(t_range), label='x0(t)')
    # ax.plot(t_range, x, label='x(t)')
    # ax.plot(t_range, d(x0)(t_range), label='dx(t)/dt')
    # ax.set(ylim=[-0.1, 10])
    # plt.legend()


def delay_differential_equation():
    # epidemic equation
    # See also: https://doi.org/10.1016/j.nonrwa.2011.12.011

    a = -0.2
    b = 0.25
    l = 0.5
    d = lambda x: lambda t: a * x(t) + b * x(l * t)

    dt = 0.01
    t_range = np.arange(-1, 100, dt)

    x0 = lambda t: 0.5 * t * (t < 0)
    # x0 = lambda t: 0.5 * (np.abs(t - 0.5) < 0.05) * (t < 1)

    x = np.zeros_like(x0(t_range))

    for idx, t in enumerate(t_range):
        if t < 0:
            x[idx] = x0(t)
        else:
            t_int = interp1d(t_range, x)
            x[idx] = x[idx - 1] + d(t_int)(t-dt)

    log.info('Displaying...')
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_range, x0(t_range), label='x0(t)')
    ax.plot(t_range, x, label='x(t)')
    ax.plot(t_range, d(x0)(t_range), label='dx(t)/dt')
    ax.set(ylim=[-0.1, 10])
    plt.legend()


if __name__ == '__main__':
    pantograph_equation()
    # delay_differential_equation()

    plt.show(block=True)

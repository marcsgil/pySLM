import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy as sci
# import dask.array as da
# import cmath
import numpy.lib.scimath as sm

from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft import Grid

np.random.seed(0)
scan_shape = np.array([100])

sphere_radius = 5e-6
noise_level = 0.05
grid = Grid(shape=scan_shape, step=200e-9)

def long2complex(vec):
    vec = vec.reshape(2, *scan_shape).astype(np.complex128)
    return vec[0] + 1j * vec[1]


def complex2long(arr):
    vec = arr.ravel()
    vec = np.hstack((vec.real, vec.imag)).astype(np.float64)
    return vec

multiplier = np.array([1, -1j, -1, 1j], dtype=np.complex128)
def meas2img(measurements: np.ndarray):
    """
    :param measurements: shape 4 x dim1 x dim2 matrix of absolute values
    :return: shape dim1 x dim2 matrix
    """
    result = multiplier[:, np.newaxis] * measurements
    return np.sum(result, axis=0)

# H = np.array([np.eye(np.prod(scan_shape), k=-1, dtype=np.complex128) + np.eye(np.prod(scan_shape),
#               k=1, dtype=np.complex128) * phase_shift for phase_shift in multiplier.conj()], dtype=np.complex128)


ref = np.zeros(np.prod(scan_shape), dtype=np.complex128)
target = np.zeros(np.prod(scan_shape), dtype=np.complex128)


def H_mul(arr):
    ref[1:] = arr[:-1]
    target[:-1] = arr[1:]
    img = np.asarray([ref + target * _ for _ in multiplier.conj()])
    return img


def dic2cp(M):
    p_nsr = noise_level ** 2
    # minimisation_guess = np.ones(scan_shape, dtype=np.complex128)
    minimisation_guess = np.exp(1j * np.angle(meas2img(M)))
    for idx in np.arange(1, M.shape[-1]):
        minimisation_guess[idx] *= minimisation_guess[idx - 1]

    global counter000
    counter000 = 0
    def cost_fun(estimate):
        global counter000
        grad_est = np.array([(np.roll(estimate, 1, axis=_) - np.roll(estimate, 0, axis=_)) for _ in range(grid.ndim)])
        # M_est = np.abs(H @ estimate.ravel()) ** 2
        M_est = np.abs(H_mul(estimate)) ** 2

        result = (np.linalg.norm(M - M_est) ** 2 / M.size) + p_nsr * (np.linalg.norm(grad_est) ** 2 / grad_est.size)
        counter000 += 1
        if counter000 % 10 == 0:
            print(f'first_part {np.linalg.norm(M - M_est) / estimate.size:.6f};  second part {p_nsr * np.linalg.norm(grad_est) / estimate.size:.6f};  evals: {counter000}')
        return result

    def equation2minimise(v):
        return cost_fun(long2complex(v))

    minimsation_result = minimize(equation2minimise, complex2long(minimisation_guess),  #method='CG',
                                  tol=0.000000005, options={'maxiter': 1000, 'disp': True})
    return long2complex(minimsation_result.x)


thickness = 2 * sphere_radius * sm.sqrt(1 - sum((_ / sphere_radius) ** 2 for _ in grid)).real

gt_opd = (1.5151 - 1.3317) / 633e-9 * thickness

ground_truth = np.exp(2j * np.pi * gt_opd)  # np.cos(grid[0] * 2 * np.pi / 5e-6)
M_gt = np.abs(H_mul(ground_truth)) ** 2
# M_gt = np.abs(H @ ground_truth) ** 2
noise = np.random.randn(*M_gt.shape) * np.amax(M_gt) * noise_level
measured_values = M_gt + noise

fig, axs = plt.subplots(2)
if grid.ndim == 1:
    axs[0].plot(grid[0] * 1e6, np.abs(ground_truth), label='Ground truth')
    axs[1].plot(grid[0] * 1e6, np.angle(ground_truth), label='Ground truth angle')
else:
    axs[0].imshow(complex2rgb(ground_truth, normalization=1), label='Ground truth', extent=grid2extent(grid)*1e6)
plt.show(block=False)
plt.pause(0.01)

result = dic2cp(measured_values)
result *= np.exp(-1j * np.median(np.angle(result)))

if grid.ndim == 1:
    axs[0].plot(grid[0] * 1e6, np.abs(result), label='Estimate')
    axs[1].plot(grid[0] * 1e6, np.angle(result), label='Estimate angle')
else:
    axs[1].imshow(complex2rgb(result, normalization=1), label='Ground truth', extent=grid2extent(grid)*1e6)
plt.legend()
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from examples.instruments.deformable_mirror import log
from optics.utils import ft, reference_object
from optics.calc import zernike
from optics.utils.display import grid2extent
from optics.instruments import objective
from optics.calc.psf import PSF

simulate_noise = True

log.info('Defining the optical system...')
obj = objective.InfinityCorrected(40, 0.80, refractive_index=1.33)

wavelength = 532e-9

background_noise_level = 0.01
max_number_of_photons = 1000  # dynamic range of camera that is used

rng = np.random.Generator(np.random.PCG64(seed=1))  # For the noise generation
# This is the original aberration represented by zernike polynomial
aberration = zernike.Polynomial([0, 0, 0, 0.5, 0.5, -0.3, 0, 0.25])
# Define the point-spread function sampling
grid_2d = ft.Grid(np.full(2, 128), 0.2e-6)
grid_3d = ft.Grid(1) @ grid_2d

# original object: usaf1951 or other intensity image
log.info('Loading reference...')
original_object = np.asarray(reference_object.usaf1951(grid_2d.shape[-2:], scale=1.0)) / 255.0  # Maximum value = 1
# original_object = np.asarray(reference_object.boat(grid_2d.shape)) / 255.0  # Maximum value = 1
original_object_ft = ft.fft2(original_object)  # to speed up calc_image


def calc_otf(correction=None) -> np.ndarray:
    if correction is None:
        wavefront = lambda nu_x, nu_y: aberration.cartesian(nu_y, nu_x)
    else:
        wavefront = lambda nu_x, nu_y: aberration.cartesian(nu_y, nu_x) + correction.cartesian(nu_y, nu_x)
    psf = PSF(objective=obj, vacuum_wavelength=wavelength,
              pupil_function=lambda nu_x, nu_y: np.exp(2j * np.pi * wavefront(nu_y, nu_x))
              )
    psf_field_array = psf(*grid_3d)  # the actual calculations happen here
    psf_intensity_array = np.abs(psf_field_array[0, 0]) ** 2
    psf_intensity_array /= np.amax(np.sum(psf_intensity_array, axis=(-2, -1)))  # normalize

    # log.info('Calculating OTF...')
    otf_array = ft.fft2(ft.ifftshift(psf_intensity_array, axes=(-2, -1)))  # Maximum value = 1

    return otf_array


nb_evaluations = np.zeros(2, dtype=int)


def calc_image(correction_parameters=()) -> np.ndarray:
    """Returns a noisy image, as from a camera with aberrations, optionally corrected with some (Zernike) parameters."""
    correction = zernike.Polynomial([0, 0, 0, *np.asarray(correction_parameters).ravel()])
    otf_array = calc_otf(correction)

    # log.info('Simulating light propagation...')
    blurred_image = np.maximum(0.0, ft.ifft2(original_object_ft * otf_array).real)  # Non-negative, maximum value ~1

    if simulate_noise:
        log.debug('Simulating Poisson photon noise...')
        detected_image = rng.poisson(
            (blurred_image * (1.0 - background_noise_level) + background_noise_level) * max_number_of_photons).astype(np.float32) / max_number_of_photons
    else:
        detected_image = blurred_image

    nb_evaluations[0] += 1

    return detected_image


#
# Define sharpness
#
radial_frequency = np.sqrt(sum(_ ** 2 for _ in grid_2d.f))  # radial spatial frequency in 1/m
diffraction_limit = wavelength / (2 * obj.numerical_aperture)  # in meters
# relative_frequency = abs_frequency * diffraction_limit  # a.u.
minimum_spatial_frequency = np.linalg.norm(grid_2d.f.step)
# define a mask. to calculate the sharpness using the information inside the mask
sharpness_mask = np.logical_and(radial_frequency > 2 * minimum_spatial_frequency, radial_frequency <= 0.75 / diffraction_limit)

# fig_M, axs_M = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(10, 8))
# axs_M.imshow(radial_frequency, extent=grid2extent(grid_2d * 1e6))
# plt.show()

# noise_to_signal_level = 5.0 / (radial_frequency / diffraction_limit)
# dl_otf = calc_otf()
# wiener_filter = dl_otf.conj() / (np.abs(dl_otf) ** 2 + noise_to_signal_level ** 2)
# sharpness_mask = np.abs(wiener_filter)  #* sharpness_mask


def calc_sharpness_of_image_slow(img) -> float:
    img_ft = ft.fft2(img)
    img_ft /= img_ft.ravel()[0]  # Normalize
    return float(np.mean(np.abs(img_ft * sharpness_mask) ** 2))


def calc_sharpness_of_image_fast(img) -> float:
    return float(np.mean(img ** 2))


calc_sharpness_of_image = calc_sharpness_of_image_fast


def calc_sharpness(correction_parameters=()) -> float:
    sharpness = calc_sharpness_of_image(calc_image(correction_parameters))
    nb_evaluations[1] += 1
    log.debug(f'{correction_parameters} => {sharpness}')
    return sharpness


# def evaluate_on_grid(fun: Callable, bounds):
#     nb_steps = 20
#     if bounds.size > 0:
#         params = bounds[0, 0] + (bounds[0, 1] - bounds[0, 0]) * (np.arange(nb_steps) + 0.5) / nb_steps
#         values = [evaluate_on_grid(lambda _: fun([p, *_]), bounds[1:]) for p in params]
#         return np.amax(values), params[np.argmin(values)]
#     else:
#         return fun()
#
#
# def optimize_correction(fun: Callable, bounds) -> np.ndarray:
#     values = evaluate_on_grid(fun, bounds)
#     return p

# calculate the sharpness of the ground_truth image and the distorted image from psf of system with aberration

ground_truth_sharpness = calc_sharpness_of_image(original_object)
original_image_sharpness = calc_sharpness()
# log.info(f'noise: {np.std([calc_sharpness() for _ in range(100)])}')
log.info(f'Sharpness of ground truth = {ground_truth_sharpness:0.6f}, and that of the original image is {original_image_sharpness:0.6f}.')
log.info('Optimizing simulated deformable mirror shape for image sharpness...')


def simplex_from_bounds(bounds):
    minima, maxima = np.asarray(bounds).transpose()
    ndims = minima.size
    simplex = np.ones([ndims + 1, 1]) * minima
    for axis, (vertex, mx) in enumerate(zip(simplex[1:], maxima)):
        vertex[axis] = mx
    return simplex


# parameters_max = np.array([1, 1, 1])
parameters_max = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])
parameters_min = -parameters_max
initial_parameters = (parameters_min + parameters_max) / 2
# optim_result = minimize(lambda p: -calc_sharpness(p), initial_parameters)
bounds = [*np.stack((parameters_min, parameters_max)).transpose()]
# optim_result = minimize(lambda p: -calc_sharpness(p), initial_parameters, tol=1e-12, method='Nelder-Mead',
#                         options={'initial_simplex': simplex_from_bounds(bounds)})
# optim_result = dual_annealing(lambda p: -calc_sharpness(p), bounds=bounds)

# Here is how the correction generated. Direct search or by fitting.


def modal_optimization_single_axis(cost_function, bounds, nb_points=9):
    def parabola(x, *parabola_parameters):
        return parabola_parameters[0] + parabola_parameters[1] * x + parabola_parameters[2] * x ** 2

    bounds = np.asarray(bounds)

    class Result:
        x = np.zeros(bounds.shape[0])

    optim_result = Result()
    parameters = np.zeros(bounds.shape[0])
    for axis, axis_bounds in enumerate(bounds):
        axis_range = np.linspace(axis_bounds[0], axis_bounds[-1], nb_points)
        costs = np.zeros(nb_points)
        for _, coordinate in enumerate(axis_range):
            parameters[axis] = coordinate
            costs[_] = cost_function(parameters)
        parameters[axis] = 0.0
        # Fit a parabola to the costs
        log.info(f'x = {axis_range}, y: {costs}')
        optim_parabola_params, _ = scipy.optimize.curve_fit(parabola, axis_range, costs, p0=np.zeros(3))
        if optim_parabola_params[2] <= 0.0:
            log.warning(f'Parabola has a maximum in sharpness! Parameters: {optim_parabola_params}')
        optim_result.x[axis] = - 0.5 * optim_parabola_params[1] / optim_parabola_params[2]
        log.info(f'x={axis_range}, y={costs} x_best={optim_result.x[axis]}')
    return optim_result


def modal_optimization_multi_axis(cost_function, bounds, nb_points=7):
    bounds = np.asarray(bounds)
    nb_dims = bounds.shape[0]
    log.info(f'Number of dimensions {nb_dims}')

    def params2cs(parabola_parameters):
        parabola_parameters = np.asarray(parabola_parameters)
        c0 = parabola_parameters[0]
        c1 = parabola_parameters[np.newaxis, 1:1+nb_dims]
        c2 = np.zeros([nb_dims, nb_dims])
        c2[np.triu_indices(nb_dims)] = parabola_parameters[1 + nb_dims:] / 2
        c2 += c2.transpose().conj()
        return c0, c1, c2

    def paraboloid(x, *parabola_parameters):
        c0, c1, c2 = params2cs(parabola_parameters)
        results = []
        for x_col in x:
            results.append(c0 + ((c1 + x_col[np.newaxis, :] @ c2) @ x_col[:, np.newaxis]).item())  # for real
        return np.asarray(results)

    class Result:
        x = np.zeros(bounds.shape[0])

    optim_result = Result()
    measurements = dict()
    parameters = np.zeros(bounds.shape[0])
    for axis, axis_bounds in enumerate(bounds):
        axis_range = np.linspace(axis_bounds[0], axis_bounds[-1], nb_points)
        for coordinate in axis_range:
            parameters[axis] = coordinate
            measurements[tuple(parameters)] = cost_function(parameters)
        parameters[axis] = 0.0
    # Fit a parabola to the costs
    params = []
    costs = []
    for p, c in measurements.items():
        params.append(p)
        costs.append(c)
    optim_parabola_params, _ = scipy.optimize.curve_fit(paraboloid, params, costs, p0=np.zeros(1 + nb_dims + nb_dims * (nb_dims + 1) // 2))
    c0, c1, c2 = params2cs(optim_parabola_params)
    optim_result.x = - np.linalg.lstsq(c2, c1.transpose())[0].flatten()  # - inv(c2) @ c1
    return optim_result


nb_evaluations[:] = 0  # Set counters to 0
# optim_result = differential_evolution(lambda p: -calc_sharpness(p), bounds=bounds, maxiter=5)
optim_result = modal_optimization_single_axis(lambda p: -calc_sharpness(p), bounds=bounds)
# optim_result = modal_optimization_multi_axis(lambda p: -calc_sharpness(p), bounds=bounds)
# optim_result = scipy.optimize.minimize(lambda p: -calc_sharpness(p), x0=np.zeros(len(bounds)))
log.info(f'Obtained a sharpness of {calc_sharpness(optim_result.x):0.6f} using {nb_evaluations} measurements.')

# optim_result = shgo(lambda p: -calc_sharpness(p), bounds=bounds)
# optim_result = basinhopping(lambda p: -calc_sharpness(p), bounds=bounds)
# if not optim_result.success:
#     log.warning(f'Optimization failed with status {optim_result.status} and message {optim_result.message}')
correction_parameters = optim_result.x
# correction_parameters = brute(lambda p: -calc_sharpness(p), ranges=bounds, Ns=20)
# correction_parameters = optimize_correction(lambda p: -calc_sharpness(p), bounds)
log.info(f'Zernike coefficients for the correction are {correction_parameters}.')
coefficient_error = np.zeros(max(correction_parameters.size, aberration.coefficients.size-3))
coefficient_error[:aberration.coefficients.size-3] = aberration.coefficients[3:]
coefficient_error[:correction_parameters.size] += correction_parameters
log.info(f'Zernike coefficient difference: {coefficient_error}, residual aberration MSE = {np.linalg.norm(coefficient_error):0.6f}.')

#
# Display
#
log.info('Displaying...')
fig_E, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(10, 8))
axs[0].imshow(original_object, extent=grid2extent(grid_2d * 1e6))
axs[0].set(xlabel=r'x [$\mu$m]', ylabel=r'y [$\mu$m]', title=f"ground truth ({ground_truth_sharpness:0.6f})")
axs[1].imshow(calc_image(), extent=grid2extent(grid_2d * 1e6))
axs[1].set(xlabel=r'x [$\mu$m]', ylabel=r'y [$\mu$m]', title=f"original ({original_image_sharpness:0.6f})")
axs[2].imshow(calc_image(correction_parameters), extent=grid2extent(grid_2d * 1e6))
axs[2].set(xlabel=r'x [$\mu$m]', ylabel=r'y [$\mu$m]', title=f"corrected ({calc_sharpness(correction_parameters):0.6f})")

plt.show()

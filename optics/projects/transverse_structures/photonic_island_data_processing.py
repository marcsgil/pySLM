import numpy as np
import matplotlib.pyplot as plt
import pathlib

from projects.transverse_structures import log
from optics.utils.ft.subpixel import Reference


def normalize_image(images: np.ndarray) -> np.ndarray:
    traces = np.mean(images, axis=-2)
    traces -= np.mean(traces, axis=-1, keepdims=True)
    traces /= np.linalg.norm(traces, axis=-1, keepdims=True)
    return np.atleast_2d(traces)


def measure_correlations(test_images):
    test_traces = normalize_image(test_images)
    return np.mean(calibration_traces[:, np.newaxis, :] * test_traces, axis=-1)


def measure_phase(test_image):
    correlations = measure_correlations(test_image)
    maximum_correlation_index = np.argmax(correlations, axis=0)
    return calibration_phases[maximum_correlation_index]


def measure_phase_robust(test_images, calibration_reference):
    test_traces = normalize_image(test_images)
    measured_phases = np.empty(test_traces.shape[0])
    test_trace_image = np.zeros_like(calibration_traces)
    for _, test_trace in enumerate(test_traces):
        test_trace_image[0] = test_trace
        registration = calibration_reference.register(test_trace_image)
        measured_phases[_] = (-registration.shift[0] * 2 * np.pi / calibration_traces.shape[0]) % (2 * np.pi) - np.pi

    return measured_phases


if __name__ == '__main__':
    input_path = pathlib.Path(__file__).parent.absolute() / 'output'
    # calibration_file_path = input_path / 'photonic_island_phase_images_2022-05-11_15-58-48.510177.npz'  # every 30 degrees
    # measurement_file_path = input_path / 'photonic_island_phase_images_2022-05-11_15-58-43.124905.npz'
    calibration_file_path = input_path / 'photonic_island_nb_images_to_average_20_2022-08-24_14-34-49.npz'  # every 10 degrees
    measurement_file_path = input_path / 'photonic_island_nb_images_to_average_20_2022-08-24_14-40-41.npz'
    #calibration_file_path = input_path / 'data_version1_ (1).npz'  # every 10 degrees
    #measurement_file_path = input_path / 'data_version4_ (2).npz'
    # measurement_file_path = calibration_file_path  # todo: remove
    test = False
    log.info(f'Loading calibration images from {calibration_file_path}...')
    calibration_data = np.load(calibration_file_path.as_posix())
    measurement_data = np.load(measurement_file_path.as_posix())

    # todo change the calibration images definition
    log.info(f'File contents: {[_ for _ in calibration_data.keys()]}')
    step = 1
    calibration_images = calibration_data['images']
    calibration_phases = calibration_data['phases']

    log.info(f'Testing with {calibration_phases.size} phases.')

    test_images = measurement_data['images'][::step, :]
    test_phases = measurement_data['phases'][::step]
    #test_phases = 0


    difference_between_images = np.mean(np.mean(np.abs(np.diff(test_images, axis=0))**2, axis=-1), axis=-1)
    log.info(f'difference_between_images = {difference_between_images}')

    # for test_image in test_images:
    #     test_image[:] = roll(test_image, np.random.randn(2) * np.asarray(test_image.shape) / 2).real

    log.info('Normalizing calibration data...')
    calibration_traces = normalize_image(calibration_images)

    log.info('Registering calibration traces with themselves to check...')
    calibration_traces_registered = calibration_traces.copy()
    for _ in range(1, calibration_traces.shape[0]):
        ref = Reference(reference_data=np.mean(calibration_traces_registered[:_], axis=0), precision=0.1)
        calibration_traces_registered[_] = ref.register(calibration_traces[_]).__array__().real

    calibration_reference = Reference(reference_data=calibration_traces, precision=0.1)

    log.info('Measuring correlations...')
    measured_correlations = measure_correlations(test_images)  # for verification
    log.info('Measuring phases...')
    measured_phases = measure_phase(test_images)
    print(measured_phases * 180 / np.pi)
    # measured_phases = measure_phase_robust(test_images, calibration_reference)
    phase_error = ((measured_phases - test_phases) + np.pi) % (2 * np.pi) - np.pi
    log.info(f'phase MSE = {np.sqrt(np.mean(np.abs(phase_error) ** 2))}')

    if test == False:
        u, s, vh = np.linalg.svd(calibration_traces)
        u, s_robust, vh = np.linalg.svd(calibration_traces_registered)
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(calibration_traces_registered)
        axs[0].set(xlabel='x', ylabel='phase')
        axs[1].imshow(np.diff(calibration_traces_registered, axis=0))
        axs[1].set(xlabel='x', ylabel='phase')
        axs[2].semilogy(s / s[0], linewidth=3, label='using position')
        axs[2].semilogy(s_robust / s_robust[0], linewidth=1, label='robust to movement')
        axs[2].set(xlabel='mode', ylabel='singular value')
        axs[2].legend()

        log.info('Displaying...')
        coefficients = np.polyfit(test_phases * 180 / np.pi, measured_phases * 180 / np.pi, 1)
        fitted_polynomial = np.poly1d(coefficients)
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(calibration_traces)
        axs[0].set(xlabel='x', ylabel='phases')
        axs[0].set_title('calibration traces')
        axs[1].imshow(measured_correlations)
        axs[1].set(xlabel='test phases [$^o$]', ylabel='measured phases [$^o$]')
        axs[1].set_title('correlations')
        axs[-1].plot(test_phases * 180 / np.pi, measured_phases * 180 / np.pi,'.')
        axs[-1].plot(test_phases * 180 / np.pi, test_phases * 180 / np.pi)
        axs[-1].plot(test_phases * 180 / np.pi, fitted_polynomial(test_phases * 180 / np.pi),'-')
        axs[-1].set(xlabel='test phases [$^o$]', ylabel='measured phases [$^o$]')
        # fig = plt.figure()
        # plt.plot(test_phases * 180 / np.pi, measured_phases * 180 / np.pi,'.',  test_phases * 180 / np.pi, test_phases * 180 / np.pi,'b', test_phases * 180 / np.pi, fitted_polynomial(test_phases * 180 / np.pi),'r')
        # plt.xlabel("test phases [$^o$]")
        # plt.ylabel("measured phases [$^o$]")

        plt.show()
    log.info('Done.')

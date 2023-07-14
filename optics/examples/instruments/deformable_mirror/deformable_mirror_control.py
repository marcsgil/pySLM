import numpy as np
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
from pathlib import Path
import time
from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils import Roi
from optics.calc import zernike
from examples.instruments.deformable_mirror import log
from optics.instruments.slm import PhaseSLM

# aberration = zernike.BasisPolynomial(n=0, m=0).cartesian  # piston, j_index = 0
# aberration = zernike.BasisPolynomial(n=1, m=-1).cartesian  # tilt, j_index = 1
# aberration = zernike.BasisPolynomial(n=1, m=1).cartesian  # tip, j_index = 2
# aberration = zernike.BasisPolynomial(n=2, m=-2).cartesian  # oblique-astigmatism, j_index = 3
# aberration = zernike.BasisPolynomial(n=2, m=0).cartesian  # defocus, j_index = 4
# aberration = zernike.BasisPolynomial(n=2, m=2).cartesian  # vertical-astigmatism, j_index = 5
# ...
# aberration = zernike.BasisPolynomial(4).cartesian  # defocus
aberration = zernike.BasisPolynomial(n=3, m=3).cartesian  # trefoil
# spherical = zernike.BasisPolynomial(n=4, m=0).cartesian

input_file_path = Path('../../results/aberration_correction_2023-02-02_15-53-02.920955.npz').resolve()


if __name__ == '__main__':
    # To load the SLM correction
    log.info(f'Loading from {input_file_path}.npz...')
    # correction_data = np.load(input_file_path.as_posix() + '.npz')
    correction_data = dict(deflection_frequency=[0.1, 0.1], two_pi_equivalent=1.0, slm_roi=[0, 0, 512, 512])

    with PhaseSLM(display_target=0, deflection_frequency=correction_data['deflection_frequency'],
                  two_pi_equivalent=correction_data['two_pi_equivalent']) as slm:
        log.info(Roi(correction_data['slm_roi']))
        slm.roi = Roi(correction_data['slm_roi'])
        slm.modulate(1.0)
        time.sleep(1)
        log.info('SLM modulation complete.')

        # with WebCam(color=False, normalize=True) as cam:
        # with SimulatedCam(normalize=True) as cam:
        with IDSCam(index=1, normalize=True, exposure_time=50e-3, gain=1, black_level=150) as cam:
            # Define how to measure the intensity at a specific point on the camera
            def measurement_function() -> float:
                img = cam.acquire()
                # return np.mean(img[img.shape[0]//2 + np.arange(3)-1, img.shape[1]//2 + np.arange(3)-1])  # pick the center of the region of interest
                return np.amax(img)  # pick the maximum (might be noisy)

            with AlpaoDM(serial_name='BAX240') as dm:
                dm.wavelength = 532e-9

                def set_dm_to_zernike(coefficients_from_astigmatism):
                    # Program the deformable mirror
                    coefficients = [0.0, 0.0, 0.0, *coefficients_from_astigmatism]
                    modulation = zernike.Polynomial(coefficients).cartesian
                    dm.modulate(lambda u, v: modulation(u, v))
                    # wait
                    time.sleep(0.250)

                def modulate_and_get_intensity(coefficients_from_astigmatism) -> float:
                    # Program the deformable mirror
                    set_dm_to_zernike(coefficients_from_astigmatism)

                    # Measure intensity
                    value = measurement_function()
                    log.info(f'For coefficients {[0, 0, 0, *coefficients_from_astigmatism]} we got an intensity of {value}.')
                    return -value

                # Reset DM
                set_dm_to_zernike([])

                log.info('Centering area of interest around brightest spot...')
                cam.center_roi_around_peak_intensity(shape=(51, 51), target_graylevel_fraction=0.30)
                log.info(f"The camera's region of interest is set to {cam.roi} with integration time {cam.exposure_time * 1000:0.3f} ms.")

                log.info('Optimizing...')
                initial_simplex = np.array([[0, 0], [1, 0], [0, 1]]) * 0.10
                # initial_simplex = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.10
                # optimized_zernike_coefficients = fmin(modulate_and_get_intensity, [0, 0], maxfun=100,
                #                                       initial_simplex=initial_simplex)
                optimization_result = differential_evolution(modulate_and_get_intensity,
                                                             bounds=[(-0.2, 0.2)] * 3, maxiter=1, popsize=15)
                optimized_zernike_coefficients = optimization_result.x
                log.info(optimized_zernike_coefficients)
                optimal_value = modulate_and_get_intensity(optimized_zernike_coefficients)
                log.info(f'Optimum of {optimal_value} at {optimized_zernike_coefficients}.')

                # dm.modulate(lambda u, v: np.zeros_like(u))
                # dm.modulate(lambda u, v: aberration(u, v) * 0.05)
                # dm.modulate(lambda u, v: (u**2 + v**2) * 0.0)
                # dm.modulate(lambda u, v: ((u**3 + v**3) - (u + v)) * 0.2)
                actuator_positions = dm.actuator_position

                # log.info(f'Actuator positions: {actuator_positions} are set to: {dm.modulation}.')

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(actuator_positions[:, 0], actuator_positions[:, 1], dm.modulation)

                plt.show()

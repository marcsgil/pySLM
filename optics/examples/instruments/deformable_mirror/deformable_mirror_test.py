import time

from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from examples.instruments.deformable_mirror import log

if __name__ == '__main__':

    with AlpaoDM(serial_name='BAX240') as dm:
        dm.wavelength = 532e-9

        for _ in range(25):
            aberration = zernike.BasisPolynomial(3 + _).cartesian  # defocus, j_index = 4
            dm.modulate(lambda u, v: aberration(u, v) * 0.5)
            actuator_positions = dm.actuator_position

            log.info(f'Displaying {_}...')
            time.sleep(0.2)

    log.info('Done!')


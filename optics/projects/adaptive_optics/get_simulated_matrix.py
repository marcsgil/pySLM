import numpy as np
from pathlib import Path

from projects.adaptive_optics import log
from optics.calc.zernike import index2noll

from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike


if __name__ == '__main__':
    output_file_path = Path('matrix.npz').absolute()

    std_coefficients = 1 + np.arange(35)  # Do a few more so that we have all Noll up to 30.
    matrix = []  # The matrix with standard Zernike coefficients.
    with AlpaoDM(serial_name='BAX240') as dm:
        log.info(f'Opened {dm}.')
        for zernike_idx in std_coefficients:
            basis_polynomial = zernike.BasisPolynomial(zernike_idx)
            transformed_aberration = lambda u, v: basis_polynomial.cartesian(v, -u) / 17.0
            dm.modulate(transformed_aberration)
            matrix.append(dm.actual_stroke_vector)
            log.info(f'{basis_polynomial.symbol}: index={basis_polynomial.j}, noll={index2noll(basis_polynomial.j)}: {dm.actual_stroke_vector[:4]}...')

    log.info('Recording matrix for deformable mirror...')
    log.info('Calculating Noll version too.')
    matrix = np.array(matrix)
    noll_coefficients = index2noll(std_coefficients)
    matrix_noll_order = matrix[np.argsort(noll_coefficients), :]

    log.info(f'The matrix of shape {matrix.shape} and values in the range [{np.amin(matrix)}, {np.amax(matrix)}] is:\n{matrix}')
    log.info(f'Standard Zernike coefficients: {std_coefficients}')
    log.info(f'Noll Zernike coefficients: {noll_coefficients}, ordered {noll_coefficients[np.argsort(noll_coefficients)]}')

    log.info(f'Saving to {output_file_path}...')
    np.savez(output_file_path, matrix=matrix, std_coefficients=std_coefficients, noll_coefficients=noll_coefficients, matrix_noll_order=matrix_noll_order)

    log.info('Done!')


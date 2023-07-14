import numpy as np
import logging
from typing import Union, Sequence

from optics.utils import ft
from optics.instruments.slm.phase_slm import encode_amplitude_as_phase

log = logging.getLogger(__name__)

array_like = Union[np.ndarray, Sequence]


__all__ = ['retrieve_phase']


def retrieve_phase(target_out: array_like, max_rel_residual: float = None, max_iter: Union[int, float] = np.infty):
    """
    The Gerchberg–Saxton algorithm wstimates the complex source function with its magnitude and that of its Fourier
    transform within the specified bounds. If the intensity is desired, use np.sqrt(target_intensity) as input argument.

    TODO: Implement the Hybrid Input-Output algorithm variation:
    https://en.wikipedia.org/wiki/Phase_retrieval

    :param target_out: The magnitude of the field after Fourier transform.
    :param max_rel_residual: Stops iteration when the relative residue has dropped below this level.
    :param max_iter: Stop the iteration after this number.

    :return: The estimated source field that satisfies the above bounds.
    """
    scaling_factor = np.sqrt(target_out.size)

    start_with_checkerboard_amplitude_modulation = False  # only seems useful for the first few iterations

    log.info('Determining target output region.')
    target_region_out = np.logical_not(np.isnan(target_out))
    target_out[np.isnan(target_out)] = 0.0
    target_out_norm = np.linalg.norm(target_out)

    def fix_magnitude(fld, target_magnitude):
        return np.exp(1j * np.angle(fld)) * target_magnitude

    target_magnitude_out = np.abs(target_out)
    # if start_with_checkerboard_amplitude_modulation:
    #     log.info('Starting from the checkerboard amplitude modulation.')
    #     estimate_in = ft.ifftn(target_out)
    #     estimate_in = np.exp(1j * (encode_amplitude_as_phase(np.abs(estimate_in)) + np.angle(estimate_in)))
    # else:
    #     estimate_out = fix_magnitude(target_out, target_magnitude_out)
    #     estimate_in = ft.ifftn(estimate_out)
    # estimate_in = fix_magnitude(estimate_in, 1.0)
    # estimate_out = ft.fftn(estimate_in)
    # residual_out = (np.abs(estimate_out) - np.abs(target_out)) * target_region_out
    # rel_residual_out_norm = np.linalg.norm(residual_out) / target_out_norm
    # log.info(f'Residual norm: {rel_residual_out_norm:0.6f}')
    iteration = 1
    while iteration < max_iter and rel_residual_out_norm > max_rel_residual:
        estimate_out = fix_magnitude(estimate_out, target_magnitude_out)
        estimate_in = ft.ifftn(estimate_out)
        estimate_in = fix_magnitude(estimate_in, 1.0)
        estimate_out = ft.fftn(estimate_in)
        residual_out = (np.abs(estimate_out) - np.abs(target_out)) * target_region_out
        rel_residual_out_norm = np.linalg.norm(residual_out) / target_out_norm
        log.info(f'{iteration}/{max_iter}: residual norm = {rel_residual_out_norm:0.6f}')
        iteration += 1

    estimate_in *= scaling_factor
    
    return estimate_in

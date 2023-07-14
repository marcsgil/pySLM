"""
A simple implementation of the Fast Fourier Transform.
"""
import numpy as np
import numpy.fft as ft
from typing import Union, Sequence

array_like = Union[np.ndarray, Sequence]


# TODO: Can only handle powers of 2!
def fft(arr: array_like, axis=-1, dtype=np.complex128) -> np.ndarray:
    arr = np.asarray(arr, dtype=dtype)
    arr = np.moveaxis(arr, axis, -1)
    return fft_cpow2_br(arr)


def bit_reverse_order(arr:array_like) -> np.ndarray:
    transform_length = arr.shape[-1]
    nb_bits = int(np.log2(transform_length))
    def reverse_bits(n: int):
        return int(f'{n:0{nb_bits}b}'[::-1], 2)
    indexes = [reverse_bits(_) for _ in range(transform_length)]
    arr[..., :] = arr[..., indexes]
    return arr


def fft_cpow2_br(arr: array_like) -> np.ndarray:
    arr = bit_reverse_order(arr)
    nb_loops = int(np.log2(arr.shape[-1]))
    for m in 2 ** np.arange(nb_loops):
        arr_view = arr.reshape((*arr.shape[:-1], -1, 2, m))
        phasors = np.exp(-1j*np.pi / m * np.arange(m))
        twiddled = arr_view[..., 1, :] * phasors
        arr_view[..., 1, :] = arr_view[..., 0, :] - twiddled
        arr_view[..., 0, :] += twiddled

    return arr


def fft_cpow2(arr: array_like, axis: int = -1) -> np.ndarray:
    def butterfly(arr: np.ndarray, axis=-1):
        transform_size = arr.shape[axis]
        remaining_axes = arr.shape[axis:][1:]
        new_shape = [*arr.shape[:axis], 2, transform_size//2, *remaining_axes]
        tmp_view = arr.reshape(new_shape)
        tmp_view = np.moveaxis(tmp_view, (axis-1, axis), (-2, -1))
        transform_view = np.moveaxis(arr, axis, -1)
        # Combine results by multiplying the odd part with the 'twiddle' factors
        even_odd_view = np.moveaxis(transform_view.reshape((*transform_view.shape[:-1], -1, 2)), -1, 0)
        half_transform_size = transform_size // 2
        twiddle_factors = np.exp(-1j * np.pi * np.arange(half_transform_size) / half_transform_size)
        even_odd_view[1] *= twiddle_factors
        # in the 'butterfly' operation: TODO: can be done more efficiently in-place
        right = even_odd_view[0] - even_odd_view[1]
        even_odd_view[0] += even_odd_view[1]
        even_odd_view[1] = right  # TODO: Reordering not strictly required
        tmp_view[..., 0, :] = even_odd_view[0]
        tmp_view[..., 1, :] = right

    transform_size = arr.shape[axis]
    if transform_size >= 2:
        remaining_axes = arr.shape[axis:][1:]
        new_shape = [*arr.shape[:axis], transform_size//2, 2, *remaining_axes]
        # Apply fft recursively to both halves (in-place)
        fft_cpow2(arr.reshape(new_shape), axis=axis-1)
        butterfly(arr, axis=axis)

    return arr


def fftn(arr: array_like, axes=None, dtype=np.complex128) -> np.ndarray:
    arr = np.asarray(arr, dtype=dtype)
    if axes is None:
        axes = range(arr.ndim)
    for _ in axes:  # Create view for each axis and Fourier transform on the chosen axis
        fft(arr, axis=_, dtype=dtype)
    return arr


def ifft(arr: array_like, axis=-1, dtype=np.complex128) -> np.ndarray:
    arr = np.asarray(arr, dtype=dtype)
    return fft(arr.conj(), axis, dtype).conj() / arr.shape[axis]


def ifftn(arr: array_like, axes=None, dtype=np.complex128) -> np.ndarray:
    arr = np.asarray(arr, dtype=dtype)
    if axes is None:
        axes = np.arange(arr.ndim)
    return fftn(arr.conj(), axes, dtype).conj() / np.prod(np.asarray(arr.shape)[axes])


if __name__ == '__main__':
    from optics.experimental import log

    # arr = [1+0.5j, 2j+1, 3, 2.5-3j]
    arr = [0, 1, 1j, 0]
    arr = np.random.randn(4, 8) + 1j * np.random.randn(4, 8)
    f = ifftn(arr.copy())  # This is an in-place operation, make sure to make a copy first!
    ftf = ft.ifftn(arr)
    log.info(f'fft(arr)={f}, ft.fft(arr)={ftf}\ndiff={np.linalg.norm(f-ftf)/np.linalg.norm(ftf)}')

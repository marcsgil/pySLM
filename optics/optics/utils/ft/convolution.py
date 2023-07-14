import numpy as np
from typing import Union, Sequence, Callable
from numbers import Complex

from .ft import fftn, ifftn, fftshift
from .grid import Grid


array_like = Union[Complex, Sequence, np.ndarray, Callable]


__all__ = ['Convolution', 'AcyclicConvolution', 'padded_conv', 'cyclic_conv', 'acyclic_conv']


def crop_pad(arr: array_like, shape: array_like, axes: array_like = None, origin_at_0: bool = True) -> np.ndarray:
    """
    Zero pads the input array to a new shape, optionally along specific axes.
    If the new shape is smaller than the current shape, the array is cropped from both ends.

    :param arr: The input array with the origin element at index 0.
    :param shape: The new shape.
    :param axes: (optional) The axes to which the new_shape applies.
    :param origin_at_0: Indicates whether the input and output arrays have the origin at 0 as after an ft.ifftshift
    (default: True)
    :return: An array of shape new_shape for the specified (or all) axes, with the origin element at index 0.
    """
    if isinstance(arr, Callable):
        arr = arr(*Grid(shape))
    padded_shape = np.asarray(arr.shape)
    if axes is None:
        axes = range(arr.ndim)
    padded_shape[axes] = shape

    if origin_at_0:
        arr = fftshift(arr, axes=axes)
    padding_thickness = shape - np.array(arr.shape)[axes]
    for ax, th in zip(axes, padding_thickness):
        if th < 0:
            nb_after = arr.shape[ax] + th
            center_before = arr.shape[ax] // 2
            center_after = nb_after // 2
            first_index = center_before - center_after
            last_index = first_index + nb_after
            arr = np.moveaxis(arr, ax, 0)
            arr = arr[first_index:last_index]
            arr = np.moveaxis(arr, 0, ax)

    padded_arr = np.zeros(padded_shape, dtype=arr.dtype)
    padded_arr[tuple(slice(_) for _ in arr.shape)] = arr
    if origin_at_0:
        padded_arr = np.roll(padded_arr, -np.asarray(arr.shape)[axes] // 2, axis=axes)
    else:
        padded_arr = np.roll(padded_arr, padded_shape[axes] // 2 - np.asarray(arr.shape)[axes] // 2, axis=axes)

    return padded_arr


class Convolution:
    """
    A class representing cyclic convolutions using the Fast Fourier Transform.
    It caches the intermediate filter, i.e. the Fourier transform of the kernel for efficiency.
    """
    def __init__(self, kernel: array_like = None, kernel_ft: array_like = None, shape: array_like = None, axes: array_like = None):
        """
        Constructs a convolution object that performs convolutions using the Fast Fourier transform. The results are
        wrapped around cyclically.
        :param kernel: The kernel as an nd-array.
        :param kernel_ft: (alternative to kernel) The non-fftshifted Fourier transform of the kernel, a.k.a. the filter.
        :param shape: (optional) The kernel shape. If larger than the provided nd-array, the kernel is zero-padded.
        :param axes: The axis to apply the convolution to.
        """
        self.__axes = axes
        if shape is not None:
            if kernel is not None:
                kernel = crop_pad(kernel, shape=shape, origin_at_0=True)
            elif kernel_ft is not None:
                kernel_ft = crop_pad(kernel_ft, shape=shape, origin_at_0=True)
        if kernel_ft is None:
            if isinstance(kernel, Callable):
                kernel = kernel(*Grid(shape))
            kernel_ft = fftn(kernel, axes=self.axes)

        if isinstance(kernel_ft, Callable):
            kernel_ft = kernel(*Grid(shape).k)
        self.__kernel_ft = kernel_ft

    @property
    def axes(self) -> np.ndarray:
        return self.__axes

    @property
    def kernel_ft(self) -> np.ndarray:
        return self.__kernel_ft

    @property
    def kernel(self) -> np.ndarray:
        return ifftn(self.__kernel_ft, axes=self.axes)

    def ft(self, obj: array_like = None, obj_ft: array_like = None) -> np.ndarray:
        if obj_ft is None:
            obj_ft = fftn(obj, axes=self.axes)

        return obj_ft * self.kernel_ft

    def __call__(self, obj: array_like = None, obj_ft: array_like = None) -> np.ndarray:
        return ifftn(self.ft(obj=obj, obj_ft=obj_ft), axes=self.axes)


class ShiftedFT:
    """
    Class representing a discrete Fourier transform shifted by shift (fractional) points.
    The returned frequencies remain equally spaced by dk.
    Instead of returning the DC component k=0 first, now the frequency shift * dk is returned.

    Calling this object returns the shifted Fourier transform of the argument.
    Calling inv() on it returns the inverse operation.
    """
    def __init__(self, shape: array_like, shift: array_like, axes: array_like = None):
        self.__axes = axes
        grid = Grid(np.atleast_1d(shape))
        k_shift = np.atleast_1d(shift) * 2 * np.pi / grid.shape
        self.__tilts = [np.exp(1j * s * rng) for rng, s in zip(grid, k_shift)]

    @property
    def axes(self) -> np.ndarray:
        return self.__axes

    def __call__(self, obj: array_like) -> np.ndarray:
        obj = np.array(obj, dtype=np.complex)
        # pre-tilt
        for _ in self.__tilts:
            obj *= _
        return fftn(obj, axes=self.axes)

    def inv(self, img: array_like) -> np.ndarray:
        obj = ifftn(img, axes=self.axes)
        # remove tilt
        for _ in self.__tilts:
            obj *= np.conj(_)
        return obj


class AcyclicConvolution(Convolution):
    def __init__(self, kernel: array_like = None, kernel_ft: array_like = None,
                 shape: array_like = None, axes: array_like = None):
        super().__init__(kernel=kernel, kernel_ft=kernel_ft, shape=shape, axes=axes)
        self.__left_ft = ShiftedFT(shape=kernel.shape, shift=-0.25, axes=axes)
        self.__right_ft = ShiftedFT(shape=kernel.shape, shift=+0.25, axes=axes)

        def shifted_kernel(shift: array_like):
            padded_kernel_shape = np.asarray(self.kernel.shape) + self.kernel.shape - 1
            if axes is not None:
                padded_kernel_shape = padded_kernel_shape[axes]
            # pad-convolve-crop
            padded_kernel_ft = crop_pad(self.kernel_ft, padded_kernel_shape, axes=axes, origin_at_0=True)  # 0-pad
            # convolve
            padded_kernel = ifftn(padded_kernel_ft, axes=axes)
            grid = Grid(np.atleast_1d(padded_kernel.shape), origin_at_center=False)
            k_shift = np.atleast_1d(shift) * 2 * np.pi / grid.shape
            tilts = [np.exp(1j * s * rng) for rng, s in zip(grid, k_shift)]
            for _ in tilts:
                padded_kernel *= _
            padded_kernel_ft = fftn(padded_kernel, axes=axes)
            # crop
            # return crop_pad(padded_kernel_ft, self.kernel.shape, axes=axes, origin_at_0=True)
            grid = Grid([128])
            return 1 / ((grid.k[0] + shift * np.diff(grid.k.as_flat[0][:2]))**2 + (0.1j))
        self.__left_kernel_ft = shifted_kernel(-0.25)
        self.__right_kernel_ft = shifted_kernel(+0.25)
        if isinstance(kernel_ft, Callable):
            self.__left_kernel_ft = kernel_ft(_ - 0.25 * np.diff(_.flatten()[:2]) for _ in Grid(kernel.shape).k)
            self.__right_kernel_ft = kernel_ft(_ + 0.25 * np.diff(_.flatten()[:2]) for _ in Grid(kernel.shape).k)

    def __call__(self, obj: array_like = None, obj_ft: array_like = None) -> np.ndarray:
        if obj is None:
            obj = ifftn(obj_ft, axes=self.axes)

        left_obj_ft = self.__left_ft(obj)
        left_img_ft = left_obj_ft * self.__left_kernel_ft
        left_img = self.__left_ft.inv(left_img_ft)

        right_obj_ft = self.__right_ft(obj)
        right_img_ft = right_obj_ft * self.__right_kernel_ft
        right_img = self.__right_ft.inv(right_img_ft)

        return 0.5 * (left_img + right_img)

    def ft(self, obj: array_like = None, obj_ft: array_like = None) -> np.ndarray:
        return fftn(self(obj=obj, obj_ft=obj_ft))


def cyclic_conv(obj: array_like = None, kernel: array_like = None,
                obj_ft: array_like = None, kernel_ft: array_like = None, shape: array_like = None, axes: array_like = None):
    """
    Calculates the cyclic convolution of obj with kernel. The input arrays should broadcast and the output array is the
    broadcast form of the inputs.
    :param obj: The object ndarray to convolve with the kernel.
    :param kernel: The kernel ndarray, with origin at 0 (ifftshifted).
    :param obj_ft: (alternative to obj) The non-fftshifted Fourier transform of the object, i.e. ft.fftn(obj).
    :param kernel_ft: (alternative to kernel) The non-fftshifted Fourier transform of the kernel, i.e. ft.fftn(kernel).
    :param shape: (optional) The kernel shape. If larger than the provided nd-array, the kernel is zero-padded.
    :param axes: The axis to apply the convolution to.
    :return: The convolution.
    """
    return Convolution(kernel=kernel, kernel_ft=kernel_ft, shape=shape, axes=axes)(obj=obj, obj_ft=obj_ft)


def acyclic_conv(obj: array_like = None, kernel: array_like = None,
                 obj_ft: array_like = None, kernel_ft: array_like = None, shape: array_like = None, axes: array_like = None):
    """
    Calculates the acyclic convolution of obj with kernel. The input arrays should broadcast and the output array is the
    broadcast form of the inputs.

    :param obj: The object ndarray to convolve with the kernel.
    :param kernel: The kernel ndarray, with origin at 0 (ifftshifted).
    :param obj_ft: (alternative to obj) The non-fftshifted Fourier transform of the object, i.e. ft.fftn(obj).
    :param kernel_ft: (alternative to kernel) The non-fftshifted Fourier transform of the kernel, i.e. ft.fftn(kernel).
    :param shape: (optional) The kernel shape. If larger than the provided nd-array, the kernel is zero-padded.
    :param axes: The axis to apply the convolution to.
    :return: The convolution.
    """
    return AcyclicConvolution(kernel=kernel, kernel_ft=kernel_ft, shape=shape, axes=axes)(obj=obj, obj_ft=obj_ft)


def padded_conv(obj: array_like = None, kernel: array_like = None,
                obj_ft: array_like = None, kernel_ft: array_like = None, shape: array_like = None, axes: array_like = None):
    """
    Calculates the ideal convolution of obj with kernel by zero-padding the inputs.
    The input arrays should broadcast and the output array is the broadcast form of the inputs.

    :param obj: The object ndarray to convolve with the kernel.
    :param kernel: The kernel ndarray.
    :param obj_ft: (alternative to obj) The non-fftshifted Fourier transform of the object, i.e. ft.fftn(obj).
    :param kernel_ft: (alternative to kernel) The non-fftshifted Fourier transform of the kernel, i.e. ft.fftn(kernel).
    :param shape: (optional) The kernel shape. If larger than the provided nd-array, the kernel is zero-padded.
    :param axes: The axis to apply the convolution to.
    :return: The convolution.
    """
    if kernel is None:
        kernel = ifftn(kernel_ft, axes=axes)
    if shape is None:
        shape = np.asarray(kernel.shape)
    padded_kernel_shape = np.asarray(kernel.shape) + shape - 1
    if axes is not None:
        padded_kernel_shape = padded_kernel_shape[axes]
    padded_kernel = crop_pad(kernel, padded_kernel_shape, axes=axes, origin_at_0=True)

    if obj is None:
        obj = ifftn(obj_ft, axes=axes)
    padded_obj = crop_pad(obj, padded_kernel_shape, axes=axes, origin_at_0=False)  # 0-pad

    padded_img = Convolution(kernel=padded_kernel, axes=axes)(obj=padded_obj)  # todo: a axes-by-axes implementation would be more memory efficient.

    img = crop_pad(padded_img, obj.shape, origin_at_0=False)
    return img

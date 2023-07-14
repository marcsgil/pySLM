import numpy as np
from typing import Union, Optional, Sequence, Callable
from numbers import Number

from . import ft
from . import subpixel
from .grid import Grid
from optics.utils.array import vector_to_axis

array_like = Union[Number, Sequence, np.ndarray]


__all__ = ['CZT', 'czt', 'cztn', 'zoomftn', 'zoomft', 'izoomft', 'izoomftn', 'SincInterpolator', 'interp']


class CZT(Callable):
    def __init__(self,
                 nb_output_points: Optional[int] = None,
                 phasor_ratio: Union[None, float, complex] = None,
                 phasor_0: Union[float, complex] = 1.0,
                 input_origin_centered: bool = False,
                 output_origin_centered: bool = False,
                 nb_input_points: Optional[int] = None):
        """
        Applies the chirp-z transform to a single dimension of the data cube, by default the left-most dimension.
        Implicit zero-padding will occur to the right of each dimension. To ensure correct sinc-interpolation, fftshift
        the input (and output) and use origin_centered=True
    
        This version is optimized for more than two dimensions and for use of the fftw library.
        Chirp-z transform algorithm paper: https://doi.org/10.1109/TAU.1969.1162034
    
        :param nb_output_points: The size of the output dimension, axis, all other dimensions are maintained as is.
            This is sometimes referred to by the letter M.

        :param phasor_ratio: The complex phasor indicating the ratio between output samples in the complex plane,
            often refered to by W. Its complex argument is the same fraction of 2pi as the ratio of the frequency difference
            between two consecutive outputs and the maximum input frequency that can be represented.

        :param phasor_0: The complex phasor indicating the first output frequency, often refered to by A.

        :param input_origin_centered: bool, default False. Indicates whether the input array has the origin in the central
            pixel (as defined by fftshift, i.e. index :code:`np.array(input_shape / 2, dtype=int) )`. Fft-shifting the input can assure
            that the zero padding leads to sinc-interpolation. This argument enables the cancelation of the incurred output phase.
            Note that the returned output centering is independent.

        :param output_origin_centered: bool, default False. Indicates whether the returned frequencies are centered around
            the origin in the central pixel (as defined by fftshift, i.e. :code:`index np.array(input_shape / 2, dtype=int) )`.
            Otherwise the frequencies start at 0 at index 0, changing monotonously.
            Note that input array is independent.

        :param nb_input_points: The number of time-points that this Chirp-z transform works on as input.
            If not specified, the data's shape[axis] is used when called.

        :return: The chirp-z transformation callable function.

        """
        if nb_output_points is None and nb_input_points is not None:
            nb_output_points = nb_input_points

        # Define phasor ratio if necessary
        if phasor_ratio is None:
            phasor_ratio = np.exp(-2j * np.pi / nb_output_points)

        # If requested, shift the output window by offsetting phasor_0
        if output_origin_centered:
            # phasor_0 = phasor_0 * f_ratio**np.floor(nb_output_frequencies / 2)  # Not accurate enough!
            half_length = np.floor(nb_output_points / 2)
            phasor_0 = phasor_0 * np.exp(1j * np.angle(phasor_ratio) * half_length)
            # The prior only works for Fourier transforms, adjust for actual z-transforms where np.abs(f_ratio) != 1:
            if not np.isclose(np.abs(phasor_ratio), 1.0, atol=16*np.finfo(complex).eps):
                phasor_0 *= np.abs(phasor_ratio)**half_length

        # store input arguments
        self.__nb_output_points = nb_output_points
        self.__phasor_ratio = phasor_ratio
        self.__phasor_0 = phasor_0
        self.__input_origin_centered = input_origin_centered

        # Pre-calculated values are stored here
        self.__nb_input_points = None
        self.__convolution_length = None
        self.__output_weights = None
        self.__chirp_filter = None
        self.__input_weights = None

        if nb_input_points is not None:
            self.__precalculate_for_input_length(nb_input_points)

    def __precalculate_for_input_length(self, nb_input_points: int):
        #
        # pre-calculated vectors: time and frequency weights, and the chirp-filter
        #
        # Prepare multiplications and chirp_filter
        chirped_exponent = np.arange(1 - nb_input_points, np.maximum(self.__nb_output_points, nb_input_points),
                                     dtype=np.min_scalar_type(-max(self.__nb_output_points, nb_input_points) ** 2)) ** 2
        output_weights = np.exp(0.5j * np.angle(self.__phasor_ratio) * chirped_exponent)
        abs_phasor_ratio = np.abs(self.__phasor_ratio)
        if not np.isclose(abs_phasor_ratio, 1):
            output_weights *= abs_phasor_ratio ** (chirped_exponent / 2)
        # Determine the convolution length when using CZT
        convolution_support = nb_input_points + self.__nb_output_points - 1
        convolution_length = convolution_support  # 2**np.ceil(np.log2(convolution_support)).astype(np.int)  # optimal for fft, length must be at least as long as the support
        # import scipy.fftpack.helper as h
        # next_optimal_length = h.next_fast_len(nb_output_frequencies - 1 + original_length)
        # Pre-calculate chirp chirp_filter
        chirp_filter = ft.fft(1.0 / output_weights[:convolution_support], convolution_length)

        # The time weights will be applied to the input first
        output_exponent = np.arange(nb_input_points)
        input_weights = complex(self.__phasor_0) ** (-output_exponent)
        input_weights *= output_weights[nb_input_points - 1 + output_exponent]

        output_weights = output_weights[slice(nb_input_points - 1, self.__nb_output_points + nb_input_points - 1)]

        # Store pre-calculated vectors
        self.__nb_input_points = nb_input_points
        self.__convolution_length = convolution_length
        self.__output_weights = output_weights
        self.__chirp_filter = chirp_filter
        self.__input_weights = input_weights

    @property
    def points(self) -> np.ndarray:
        """The complex-plane values at which this Chirp-z transform will computes the output. In the case of the Fourier
        transform, these would be uniformly distributed on the unit circle in the complex plane."""
        return self.__phasor_0 * complex(self.__phasor_ratio) ** (-np.arange(self.__nb_output_points))

    def __call__(self, data_cube: Union[Sequence, np.ndarray], axis: Optional[int]=-1) -> np.ndarray:
        """
        Applies the chirp-z transform to a single dimension of the data cube, by default the left-most dimension.
        Implicit zero-padding will occur to the right of each dimension. To ensure correct sinc-interpolation,
        fftshift the input (and output) and use origin_centered=True
    
        This version is optimized for more than two dimensions and for use of the fftw library.
        Chirp-z transform algorithm paper: https://doi.org/10.1109/TAU.1969.1162034
    
        :param data_cube: An n-dimensional array to be transformed along dimension 'axis'.
        :param axis: The dimension to perform the chirp-z transform on. Default: -1, the last (right-most) axis.

        :return: The chirp-z transformed data_cube, with identical shape as the input data_cube, bar dimension, axis.
            Along the dimension axis, the number of array elements is nb_output_frequencies.
        """
        # log.debug(f"data_cube={data_cube}, nb_output_frequencies={nb_output_frequencies}, phasor_ratio={phasor_ratio}, phasor_0={phasor_0}")
        # Check whether the input data cube shape is correct
        data_cube = np.atleast_1d(data_cube)
        nb_input_points = data_cube.shape[axis]
        if self.__nb_input_points != nb_input_points:
            self.__precalculate_for_input_length(nb_input_points)
        # Make the axis count backwards from the right-hand side
        axis = np.mod(axis, data_cube.ndim) - data_cube.ndim

        # Pre-multiply the data with the time-weights.
        result = data_cube * vector_to_axis(self.__input_weights, axis=axis)  # The pre-multiplied input data
    
        # Do the fast convolution using an FFT.
        result = ft.fft(result, self.__convolution_length, axis=np.mod(axis, result.ndim))  # Fourier transform of pre-multiplied input
        result *= vector_to_axis(self.__chirp_filter, axis=axis)  # Fourier transform of extended result
        result = ft.ifft(result, axis=np.mod(axis, result.ndim))  # Extended result
    
        # Select requested frequencies and do the final multiplication
        result = result.swapaxes(0, axis)
        result = result[np.arange(self.__nb_output_points) + self.__nb_input_points - 1, ...]
        result = result.swapaxes(0, axis)
        result *= vector_to_axis(self.__output_weights, axis=axis)
    
        if self.__input_origin_centered:  # add the phase modulation corresponding to the shift in input space
            largest_freq_in_input_units = np.floor(np.array(data_cube.shape[axis]) / 2)
            delta_rng = largest_freq_in_input_units * np.imag(np.log(self.__phasor_ratio)) / (-2 * np.pi)
            if not np.isclose(delta_rng, 0, atol=16*np.finfo(complex).eps):
                output_pixel_shift = np.angle(self.__phasor_0) / np.angle(self.__phasor_ratio)
                result *= vector_to_axis(
                    np.exp(2j*np.pi * (np.arange(result.shape[axis]) - output_pixel_shift) * delta_rng),
                    axis=axis)  # Correct the pre-shift induced phase error
    
        return result


def czt(data_cube: array_like,
        nb_output_points: Optional[int] = None,
        phasor_ratio: Union[None, float, complex] = None,
        phasor_0: Union[None, float, complex] = 1.0,
        axis: Optional[int]=-1,
        input_origin_centered: bool=False,
        output_origin_centered: bool=False) -> np.ndarray:
    """
    Applies the chirp-z transform to a single dimension of the data cube, by default the left-most dimension.
    Implicit zero-padding will occur to the right of each dimension. To ensure correct sinc-interpolation, fftshift the
    input (and output) and use origin_centered=True
    All but the first argument are optional.

    This version is optimized for more than two dimensions and for use of the fftw library.
    Chirp-z transform algorithm paper: https://doi.org/10.1109/TAU.1969.1162034

    :param data_cube: An n-dimensional array to be transformed along dimension 'axis'.
    :param nb_output_points: The size of the output dimension, axis, all other dimensions are maintained as is.
        This is sometimes referred to by the letter M.
    :param phasor_ratio: The complex phasor indicating the ratio between output samples in the complex plane,
        often refered to by W. Its complex argument is the same fraction of 2pi as the ratio of the frequency difference
        between two consecutive outputs and the maximum input frequency that can be represented.
    :param phasor_0: The complex phasor indicating the first output frequency, often refered to by A.
    :param axis: The dimension to perform the chirp-z transform on. Default: -1, the last (right-most) axis.
    :param input_origin_centered: bool, default False. Indicates whether the input array has the origin in the central
        pixel (as defined by fftshift, i.e. index np.array(input_shape / 2, dtype=int) ). Fft-shifting the input can assure
        that the zero padding leads to sinc-interpolation. This argument enables the cancelation of the incurred output phase.
        Note that the returned output centering is independent.
    :param output_origin_centered: bool, default False. Indicates whether the returned frequencies are centered around
        the origin in the central pixel (as defined by fftshift, i.e. index np.array(input_shape / 2, dtype=int) ).
        Otherwise the frequencies start at 0 at index 0, changing monotonously.
        Note that input array is independent.
    :return: The chirp-z transformed data_cube, with identical shape as the input data_cube, bar dimension, axis.
        Along the dimension axis, the number of array elements is nb_output_frequencies.
    """
    data_cube = np.atleast_1d(data_cube)
    return CZT(nb_output_points=nb_output_points, phasor_ratio=phasor_ratio, phasor_0=phasor_0,
               input_origin_centered=input_origin_centered, output_origin_centered=output_origin_centered,
               nb_input_points=data_cube.shape[axis])(data_cube, axis=axis)


def cztn(data_cube: array_like,
         output_shape: Union[int, Sequence, np.ndarray] = None,
         phasor_ratio: array_like = None,
         phasor_0: array_like = 1.0,
         axes: Union[int, Sequence, np.ndarray] = None,
         input_origin_centered: Union[bool, Sequence, np.ndarray] = False,
         output_origin_centered: Union[bool, Sequence, np.ndarray] = False):
    """
    Calculate the partial spectrum of x using the n-dimensional chirp-z transform.
    The phasors can be specified for less dimensions than those of the input data. The argument, axes, can indicate to
    which axes the grids apply. By default, it applies to the right-most axes.
    All but the first argument are optional.

    :param data_cube: input matrix, centered on central element (rounded up when the dimension is odd as with the
        fftshift). If required, zero padding will occur at the end of each dimension, and will be corrected for in
        the final result.
    :param output_shape: vector containing the shape of output. This is sometimes referred to by M.
        Default: the shape of the input data_cube.
    :param phasor_ratio: vector containing the scaling factors corresponding to each chirp-z transform per dimension.
        These factors are often referred as by the letter W.
    :param phasor_0: The first frequencies for each dimension (often called A).
    :param axes: Axes over which to compute the chrip-z transform. If not given, the last len(output_shape) axes are
        used, or all axes if output_shape is also not specified. Repeated indices in axes are ignored.
    :param input_origin_centered: bool or Sequence thereof, default False. Indicates whether the input array has the origin in the central
        pixel (as defined by fftshift, i.e. index np.array(input_shape / 2, dtype=int) ). Fft-shifting the input can assure
        that the zero padding leads to sinc-interpolation. This argument enables the cancelation of the incurred output phase.
        Note that the returned output centering is independent.
    :param output_origin_centered: bool or Sequence thereof, default False. Indicates whether the returned frequencies are centered around
        the origin in the central pixel (as defined by fftshift, i.e. index np.array(input_shape / 2, dtype=int) ).
        Otherwise the frequencies start at 0 at index 0, changing monotonously.
        Note that input array is independent.

    :return: output matrix of size [M, size(x,length(M)+1) size(x,length(M)+2) ... size(x,ndims(x))]
    """
    # Make sure that the inputs are all numpy ndarrays
    data_cube = np.asarray(data_cube)

    output_shape = np.array(output_shape, dtype=int).flatten()

    if axes is None:
        axes = list(range(-output_shape.size, 0))
    else:
        if np.isscalar(axes):
            axes = [axes]
        axes = list(set(axes))

    # Complete the arrays for all dimensions
    full_output_shape = np.array(data_cube.shape, dtype=int).flatten()
    if output_shape is not None:
        full_output_shape[axes] = output_shape

    full_f_ratio = np.exp(-2j * np.pi / full_output_shape)
    if phasor_ratio is not None:
        full_f_ratio[axes] = np.array(phasor_ratio).flatten()

    full_f_0 = np.zeros_like(full_f_ratio)
    if phasor_0 is not None:
        full_f_0[axes] = np.array(phasor_0).flatten()

    full_input_origin_centered = np.zeros(shape=full_f_ratio.shape, dtype=bool)
    if input_origin_centered is not None:
        full_input_origin_centered[axes] = np.array(input_origin_centered, dtype=bool).flatten()

    full_output_origin_centered = np.zeros(shape=full_f_ratio.shape, dtype=bool)
    if output_origin_centered is not None:
        full_output_origin_centered[axes] = np.array(output_origin_centered, dtype=bool).flatten()

    result = data_cube

    for axis in axes:
        if result.shape[axis] > 1:
            result = czt(result, full_output_shape[axis], full_f_ratio[axis], full_f_0[axis],
                         axis=axis,
                         input_origin_centered=full_input_origin_centered[axis],
                         output_origin_centered=full_output_origin_centered[axis])
        else:
            result = np.repeat(result, repeats=full_output_shape[axis], axis=axis)

    return result


def zoomftn(data_cube: array_like,
            k: Union[tuple, Sequence, np.ndarray, Grid] = None,
            x: Union[tuple, Sequence, np.ndarray, Grid] = None,
            f: Union[tuple, Sequence, np.ndarray, Grid] = None,
            magnification: Union[int, float, tuple, Sequence, np.ndarray] = 1.0,
            shape: Union[int, float, tuple, Sequence, np.ndarray] = None,
            k_offset: Union[int, float, tuple, Sequence, np.ndarray] = None,
            x_offset: Union[int, float, tuple, Sequence, np.ndarray] = 0.0,
            f_offset: Union[int, float, tuple, Sequence, np.ndarray] = 0.0,
            axes: Union[int, tuple, Sequence, np.array] = None):
    """
    Efficiently calculates the ND Fourier transform at frequencies with uniform but arbitrary spacing.
    The spatial and frequency grids can specify less dimensions than those of the input data. In that case the argument
    `axes` indicates to which axes the grids apply. By default, it applies to the right-most axes.

    Its efficiency stems from the use of the chirp-z transform which has a time-complexity of the order of
        ``2(n+m-1)log(n+m-1) < Mn log(Mn) when todo,``
    where M is the magnification factor and n is the number of points in the input and m is the number of points
    in the output.

    In its simples form, this returns the same result as the fft:
        np.allclose(zoomftn(data_cube) == fftn(data_cube)) == True
    It may be used to deal with translated input or outputs, e.g.:
        zoomftn(data_cube, k_offset=2*np.pi * np.array(5, 10))

    :param data_cube: The data to be transformed as an np.ndarray
    :param k: The wavenumbers, angular frequencies, or Grid or tuple thereof to return.
        These should be uniformly spaced and preferably fftshifted.
    :param x: The spatial coordinates, time points, or Grid or tuple thereof. These should be uniformly spaced and
        preferably ifftshifted.
    :param f: The spatial frequencies, time frequencies, or Grid or tuple thereof to return.
        These should be uniformly spaced and preferably fftshifted. This is only used when k is not specified.
    :param magnification: The desired magnification factor of frequency space, and
        the respective demagnification of the view-extent. This is ignored if either k or f is specified.
    :param shape: The new shape after magnification. This is ignored if either k or f is specified.
    :param k_offset: The wavenumber offset as a scalar or a vector. A scalar is applied to all axes.
        This is ignored if either k or f, is specified.
    :param x_offset: The offset of the input as a scalar or a vector. A scalar is applied to all axes.
        This is ignored if x is specified.
    :param f_offset: The wavenumber offset as a scalar or a vector. A scalar is applied to all axes.
        This is ignored if either k_offset, k, or f, is specified.
    :param axes: The axes to apply the zoom Fourier transform to. Default: None, all axes.

    :return: A complex nd-array with the Fourier transform sampled at the specified frequencies.
    """
    # axes=None is the default of zoomftn => all axes
    # Default to all axes instead of just the last one
    data_cube = np.asarray(data_cube)
    # default step size is 1 and ifftshifted
    if x is None:
        x_shape = np.asarray(data_cube.shape)
        if axes is not None:
            x_shape = x_shape[axes]
        x = Grid(x_shape, center=x_offset, origin_at_center=False)
    elif not isinstance(x, Grid):
        x = Grid.from_ranges(*x)
    # Determine the k-space grid
    if k is None:
        if f is not None:
            if not isinstance(f, Grid):
                f = Grid.from_ranges(*f)
            k = f * (2 * np.pi)
        else:  # k and f are both None
            if k_offset is None:
                k_offset = 2*np.pi * np.asarray(f_offset).flatten()

            k = x.k / np.asarray(magnification).flatten() + np.asarray(k_offset).flatten()
            k = k.mutable
            k.shape = shape  # ignored if shape is None
    elif not isinstance(k, Grid):
        k = Grid.from_ranges(*k)

    # Normalize the x and k Grids so that x.step == 1 and k is still consistent with x for the Fourier transform.
    k = k * x.step
    x = x / x.step

    # Let k define the axes unless otherwise specified
    if axes is None:
        axes = np.arange(-k.ndim, 0)
    else:
        axes = np.mod(axes, data_cube.ndim) - data_cube.ndim  # count back from right-most

    # The input and output data of cztn are fftshifted
    # List the axes that need to be fftshifted and ifftshifted
    axes_to_fftshift = axes[np.arange(-x.ndim, 0, 1)[np.logical_not(x.origin_at_center)]]

    # Transform to frequency-space
    # print(data_cube.shape)
    # print(f"k.shape={k.shape}, axis={axes}, data_cube.shape={data_cube.shape}")
    data_cube_ft = cztn(ft.fftshift(data_cube, axes=axes_to_fftshift), output_shape=k.shape,
                        phasor_ratio=np.exp(-1j * k.step), phasor_0=np.exp(-1j * k.center),
                        axes=axes, input_origin_centered=True, output_origin_centered=True)
    # Ifftshift the axes as indicated in the k-grid.
    axes_to_ifftshift = axes[np.arange(-k.ndim, 0, 1)[np.logical_not(k.origin_at_center)]]
    data_cube_ft = ft.ifftshift(data_cube_ft, axes=axes_to_ifftshift)

    # Account for a shifted center at the input
    for idx, axis in enumerate(axes):
        if not np.isclose(x.center[idx], 0.0):
            data_cube_ft *= np.exp(-1j * k[idx] * x.center[idx])

    # Correct amplitude for zoom
    # The energy is spread over more sample points when sub-sampling. The next lines makes sure that the input
    # and output norm are the same when the output space size M goes to infinity.
    # step_size = k.step[axes] / (2 * np.pi)
    # amplitude_correction = np.prod(np.sqrt(step_size[np.logical_not(np.isclose(step_size, 0.0))]))  # But, do skip any singleton dimensions as the sample size would be undefined
    # data_cube_ft *= amplitude_correction

    return data_cube_ft


def zoomft(data_cube: Union[Sequence, np.ndarray],
           k: Union[Sequence, np.ndarray, Grid] = None,
           x: Union[Sequence, np.ndarray, Grid] = None,
           f: Union[Sequence, np.ndarray, Grid] = None,
           magnification: Union[int, float] = 1.0,
           length: Union[int, None] = None,
           k_offset: Union[int, float, None] = None,
           x_offset: Union[int, float] = 0.0,
           f_offset: Union[int, float] = 0.0,
           axis: Optional[int] = -1):
    r"""
    Efficiently calculates the 1D Fourier transform at frequencies with uniform but arbitrary spacing.
    Its efficiency stems from the use of the chirp-z transform which has a time-complexity of the order of
    :math:`2(n+m-1)\log(n+m-1) < Mn \log(Mn)` when todo: do complexity analysis,
    where M is the magnification factor and n is the number of points in the input and m is the number of points
    in the output.
    All but the first argument are optional.

    In its simples form, this returns the same result as the fft:
        `np.allclose(zoomft(data_cube) == fft(data_cube)) == True`
    It may be used to deal with translated input or outputs, e.g.:
        `zoomft(data_cube, k_offset=2*np.pi * 10)`

    :param data_cube: The data to be transformed as an np.ndarray
    :param k: The wavenumbers, angular frequencies, or Grid thereof to return.
        These should be uniformly spaced and preferably ifftshifted.
    :param x: The spatial coordinates, time points, or Grid thereof. These should be uniformly spaced and preferably
        fftshifted.
    :param f: The spatial frequencies, time frequencies, or Grid thereof to return.
        These should be uniformly spaced and preferably fftshifted. This is only used when k is not specified.
    :param magnification: The desired magnification factor of frequency space, and
        the respective demagnification of the view-extent. This is ignored if either k or f is specified.
    :param length: The new axis length after magnification. This is ignored if either k or f is specified.
    :param k_offset: The wavenumber offset as a scalar. A scalar is applied to all axes.
        This is ignored if either k or f, is specified.
    :param x_offset: The offset of the input as a scalar. A scalar is applied to all axes.
        This is ignored if x is specified.
    :param f_offset: The wavenumber offset as a scalar. A scalar is applied to all axes.
        This is ignored if either k_offset, k, or f, is specified.
    :param axis: The axis to apply the zoom Fourier transform to. Default: -1, the right-most axis.

    :return: A complex nd-array with the Fourier transform sampled at the specified frequencies.
    """
    # Default to the right-most axis

    # Wrap scalar parameters in tuples and pass to zoomftn
    if k is not None and not isinstance(k, Grid):
        k = (k, )

    if x is not None and not isinstance(x, Grid):
        x = (x, )

    if f is not None and not isinstance(f, Grid):
        f = (f, )

    if length is not None:
        shape = (length, )
    else:
        shape = None

    return zoomftn(data_cube,
                   k=k, x=x, f=f,
                   magnification=magnification, shape=shape,
                   k_offset=k_offset, x_offset=x_offset, f_offset=f_offset,
                   axes=(axis, ))


def izoomftn(data_cube_ft: Union[Sequence, np.ndarray],
             k: Union[tuple, Sequence, np.ndarray, Grid] = None,
             x: Union[tuple, Sequence, np.ndarray, Grid] = None,
             f: Union[tuple, Sequence, np.ndarray, Grid] = None,
             magnification: Union[int, float, tuple, Sequence, np.ndarray] = 1.0,
             new_shape: Union[int, float, tuple, Sequence, np.ndarray] = None,
             k_offset: Union[int, float, tuple, Sequence, np.ndarray] = None,
             x_offset: Union[int, float, tuple, Sequence, np.ndarray] = 0.0,
             f_offset: Union[int, float, tuple, Sequence, np.ndarray] = 0.0,
             axes: Union[int, tuple, Sequence, np.array] = None):
    """
    Efficiently calculates the ND Fourier transform at frequencies with uniform but arbitrary spacing.
    The spatial and frequency grids can specify less dimensions than those of the input data. The argument axes can
    indicate to which axes the grids apply. By default, it applies to the right-most axes.

    Its efficiency stems from the use of the chirp-z transform which has a time-complexity of the order of
        2(n+m-1)log(n+m-1) < Mn log(Mn) when todo: complexity analysis,
    where M is the magnification factor and n is the number of points in the input and m is the number of points
    in the output.
    All but the first argument are optional.

    In its simples form, this returns the same result as the fft:
        np.allclose(izoomftn(data_cube) == ifftn(data_cube)) == True
    It may be used to deal with translated input or outputs, e.g.:
        izoomftn(data_cube, k_offset=2*np.pi * np.array(5, 10))

    :param data_cube_ft: The data to be inverse Fourier transformed as an np.ndarray
    :param k: The wavenumbers, angular frequencies, or Grid or tuple thereof as input.
        These should be uniformly spaced and ifftshifted for efficiency.
    :param x: The spatial coordinates, time points, or Grid or tuple thereof. These should be uniformly spaced and
        preferably fftshifted.
    :param f: The spatial frequencies, time frequencies, or Grid or tuple thereof as input.
        These should be uniformly spaced and preferably fftshifted. This is only used when k is not specified.
    :param magnification: The desired magnification factor of frequency space that needs to be undone. This is ignored
        if either k or f is specified.
    :param new_shape: The new shape after magnification. This is ignored if either k or f is specified.
    :param k_offset: The wavenumber offset as a scalar or a vector that needs to be undone.
        A scalar is applied to all axes. This is ignored if either k or f, is specified.
    :param x_offset: The offset of the input as a scalar or a vector that needs to be undone.
        A scalar is applied to all axes. This is ignored if x is specified.
    :param f_offset: The wavenumber offset as a scalar or a vector that needs to be undone.
        A scalar is applied to all axes. This is ignored if either k_offset, k, or f, is specified.
    :param axes: The axes to apply the zoom Fourier transform to. Default: None, all axes.

    :return: A complex nd-array with the Fourier transform sampled at the specified frequencies.
    """
    if f is None:
        if k is None:
            f_shape = np.array(np.asarray(data_cube_ft).shape)
            if axes is not None:
                f_shape = f_shape[axes]
            f = Grid(shape=f_shape, extent=1.0, origin_at_center=False)
        else:
            if not isinstance(k, Grid):
                k = Grid.from_ranges(*k)
            f = k / (2 * np.pi)
    elif not isinstance(f, Grid):
        f = Grid.from_ranges(*f)
    # f should be a Grid from this point onwards

    if magnification is None:
        inv_magnification = None
    else:
        inv_magnification = 1 / np.asarray(magnification).ravel()

    if f_offset is None:
        f_offset = np.asarray(k_offset).ravel() / (2 * np.pi)

    if x is None:
        if new_shape is None:
            new_shape = f.shape
        x = Grid(shape=new_shape, origin_at_center=False)
    else:
        if not isinstance(x, Grid):
            x = Grid.from_ranges(*x)
    # x should be a Grid from this point onwards

    scaling_factor = x.size

    return zoomftn(np.conj(data_cube_ft),
                   k=None, x=f, f=x,
                   magnification=inv_magnification, shape=new_shape,
                   k_offset=None, x_offset=-f_offset, f_offset=-x_offset,
                   axes=axes
                   ).conj() / scaling_factor


def izoomft(data_cube_ft: Union[Sequence, np.ndarray],
            k: Union[Sequence, np.ndarray, Grid] = None,
            x: Union[Sequence, np.ndarray, Grid] = None,
            f: Union[Sequence, np.ndarray, Grid] = None,
            magnification: Union[int, float] = 1.0,
            length: Union[int, None] = None,
            k_offset: Union[int, float, None] = None,
            x_offset: Union[int, float] = 0.0,
            f_offset: Union[int, float] = 0.0,
            axis: Optional[int] = -1):
    """
    Efficiently calculates the 1D inverse Fourier transform at positions with uniform but arbitrary spacing.

    Its efficiency stems from the use of the chirp-z transform which has a time-complexity of the order of
        2(n+m-1)log(n+m-1) < Mn log(Mn) when todo: complexity analysis,
    where M is the magnification factor and n is the number of points in the input and m is the number of points
    in the output.
    All but the first argument are optional.

    In its simples form, this returns the same result as the fft:
        np.allclose(izoomft(data_cube) == ifft(data_cube)) == True

    :param data_cube_ft: The data to be inverse Fourier transformed as an np.ndarray
    :param k: The wavenumbers, angular frequencies, or Grid or tuple thereof as input.
        These should be uniformly spaced and preferably ifftshifted.
    :param x: The spatial coordinates, time points, or Grid or tuple thereof. These should be uniformly spaced and
        preferably fftshifted.
    :param f: The spatial frequencies, time frequencies, or Grid or tuple thereof as input.
        These should be uniformly spaced and preferably fftshifted. This is only used when k is not specified.
    :param magnification: The desired magnification factor of frequency space that needs to be undone. This is ignored
        if either k or f is specified.
    :param length: The new length after magnification. This is ignored if either k or f is specified.
    :param k_offset: The wavenumber offset as a scalar or a vector that needs to be undone.
        A scalar is applied to all axes. This is ignored if either k or f, is specified.
    :param x_offset: The offset of the input as a scalar or a vector that needs to be undone.
        A scalar is applied to all axes. This is ignored if x is specified.
    :param f_offset: The wavenumber offset as a scalar or a vector that needs to be undone.
        A scalar is applied to all axes. This is ignored if either k_offset, k, or f, is specified.
    :param axis: The axis to apply the zoom Fourier transform to. Default: -1, the last (right-most) axis.

    :return: A complex nd-array with the Fourier transform sampled at the specified frequencies.
    """
    # Wrap scalar parameters in tuples and pass to zoomftn
    if k is not None and not isinstance(k, Grid):
        k = (k, )

    if x is not None and not isinstance(x, Grid):
        x = (x, )

    if f is not None and not isinstance(f, Grid):
        f = (f, )

    if length is not None:
        shape = (length, )
    else:
        shape = None

    return izoomftn(data_cube_ft,
                    k=k, x=x, f=f,
                    magnification=magnification, new_shape=shape,
                    k_offset=k_offset, x_offset=x_offset, f_offset=f_offset,
                    axes=(axis, ))


class SincInterpolator(Callable):
    def __init__(self, data_cube: array_like, grid: Grid = None):
        """
        A class to represent sinc-interpolatable nd-arrays that permit the selection of sub-sets using fraction indices.

        * zoomable(grid or ranges) for selecting on physical coordinates.

        * zoomable[:, ..., indices] for selecting on axes indices (not necessarily integer).

        :param data_cube: The n-dimensional data array.
        :param grid: An optional Grid object for the physical coordinates. Default: the integer indices of Grid(data_cube_shape, first=0)

        :return: An object that can be called or index on a sub(pixel)-grid
        """
        data_cube = np.asarray(data_cube)
        if grid is None:
            grid = Grid(data_cube.shape, first=0)
        elif grid.ndim != data_cube.ndim or np.any(grid.shape != np.asarray(data_cube.shape)):
            data_cube = np.broadcast_to(data_cube, grid.shape)

        self.__data_cube_ft = ft.ifftn(data_cube)
        self.__data_cube_ft = subpixel.roll_ft(self.__data_cube_ft, shift=grid.first / grid.step, axes=np.arange(-grid.ndim, 0))  # Shift data for a grid with the first pixel coordinate at the origin
        self.__grid = grid
        self.__grid_px = Grid(shape=grid.shape, first=0)

    @property
    def dtype(self):
        return self.__data_cube_ft.dtype

    @property
    def grid(self) -> Grid:
        """The physical grid. This is used when selecting using self(...)."""
        return self.__grid

    @property
    def ndim(self) -> int:
        return self.__grid.ndim

    @property
    def shape(self) -> np.ndarray:
        return self.__grid.shape

    def __getitem__(self, key: Union[Sequence, None, slice, int, bool, array_like]) -> np.ndarray:
        """
        Select a slice of the zoomable area using pixel subscripts instead of physical coordinates.

        :param key: The selection indices (start at 0 but may be fractional).
        :return: The complex interpolated data.
        """
        # Make sure that the key is always a tuple of indices
        if isinstance(key, Grid):
            sub_grid = key
        else:
            if isinstance(key, Sequence):
                if not isinstance(key, tuple):
                    key = tuple(key)
            else:
                key = (key, )
            # split key into the tuples key and right_key
            index_is_split = [_ is Ellipsis for _ in key]  # using is instead of == because arrays use == per element
            if True in index_is_split:
                ellipsis_index = index_is_split.index(True)
                right_key = key[ellipsis_index+1:]
                key = key[:ellipsis_index]  # left-key
            else:
                right_key = []

            # Make a list of ranges and complete any gap
            ranges = []
            for _, rng in enumerate(key):
                if isinstance(rng, Grid) and rng.ndim > 0:
                    ranges += rng
                else:
                    ranges.append(rng)
            right_ranges = []
            for _, rng in enumerate(right_key):
                if isinstance(rng, Grid) and rng.ndim > 0:
                    right_ranges += rng
                else:
                    right_ranges.append(rng)
            for _ in range(len(ranges), self.ndim - len(right_ranges)):
                ranges.append(slice(None, None, None))
            ranges += right_ranges
            # Replace any slices
            for _, rng in enumerate(ranges):
                if isinstance(rng, slice):
                    start = 0 if rng.start is None else rng.start
                    stop = self.__grid_px.shape[_] if rng.stop is None else rng.stop
                    step = 1 if rng.step is None else rng.step
                    rng = np.arange(start, stop, step)
                ranges[_] = rng
            sub_grid = Grid.from_ranges(*ranges)
        return zoomftn(self.__data_cube_ft, x=self.__grid_px.f, f=sub_grid)

    def __call__(self, *ranges: Sequence[array_like]) -> np.ndarray:
        """
        Select a slice of the zoomable area using physical coordinates instead of pixel subscripts.

        :param ranges: The selection coordinates.
        :return: The complex interpolated data.
        """
        if isinstance(ranges[0], Grid):
            sub_grid = ranges[0]
        else:
            sub_grid = Grid.from_ranges(*ranges)
        return zoomftn(self.__data_cube_ft, x=self.__grid.f, f=sub_grid)


def interp(data_cube: array_like, factor: array_like = 1.0, from_grid: Grid = None, to_grid: Grid = None) -> np.ndarray:
    """
    Sinc-interpolate an nd-imensional array from one Cartesian grid to another.

    :param data_cube: The n-dimensional data that is to be interpolated.
    :param factor: The factor to up-sample with. The is ignored when the argument to_grid is specified. This must be
        either a vector with one element per dimension or a scalar that scales all dimensions equally.
    :param from_grid: The optional grid of the data_cube. Default: the integer indices of Grid(data_cube.shape, first=0).
    :param to_grid: The selection grid in the same coordinates as the from_grid when specified, otherwise as
        (potentially fractional) axis indices. The default value is calculated from the up-sampling-factor, using the same
        extent as the original data but steps that are smaller by that factor.

    :return: The chirp-z transformed data_cube, with identical shape as the input data_cube, bar dimension, axis.
        Along the dimension axis, the number of array elements is nb_output_frequencies.
    """
    if from_grid is None:
        if to_grid is None:
            from_grid = Grid(data_cube.shape, first=0)
        else:
            from_grid = Grid(extent=to_grid.extent, step=to_grid.step * factor, center=to_grid.center)
    if to_grid is None:
        to_grid = Grid(extent=from_grid.extent, step=from_grid.step / factor, center=from_grid.center)
    return SincInterpolator(data_cube=data_cube, grid=from_grid)(to_grid)

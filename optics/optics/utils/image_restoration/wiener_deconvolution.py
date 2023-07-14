import numpy as np
from optics.utils import ft
from optics.utils.ft import Grid


def wiener_2d(blurred_noisy_img: np.ndarray, point_spread_function: np.ndarray,
              noise_level: float = 0.01, signal_level: float = 0.025):
    """
    Deconvolves an image using the linear Wiener deconvolution.
    For didactic reasons, this is limited to 2D and not very efficient.

    :param blurred_noisy_img: The to-be-deconvolved image as an nd-array.
    :param point_spread_function: The point-spread function as an nd-array.
    :param noise_level: Standard deviation as a fraction of the dynamic range. The noise_level depends on the sensitivity of the sensor.
    :param signal_level: The ground-truth contrast expected at the highest spatial frequency (alternating pixels) when the mean intensity is 50# of the dynamic range. The signal level depends on the dynamic range.
    :return:
    """
    #
    # Determine the filter
    #
    # First get the OTF
    optical_transfer_function = ft.fftshift(ft.fft2(ft.ifftshift(point_spread_function)))
    # Construct a signal-to-noise model
    # Determine the spatial frequencies in the image
    grid = Grid(blurred_noisy_img.shape[:2])
    fx_range, fy_range = grid.f
    f_absolute2 = fx_range**2 + fy_range**2  # This is squared later, so not really needed.
    # normalize the spatial frequencies to the pixel size
    f_relative2 = f_absolute2 / (np.max(np.max(np.abs(fx_range[0])), np.max(np.abs(fy_range[0])))**2)
                                                                                                                                                                                # a power-law signal level: signal_level = signal_level ./ f_relative
    signal_level_per_color = signal_level * (np.mean(np.mean(blurred_noisy_img)) / 0.50)  # The signal is modeled as proportional to the brighness of each color channel
    nsr2 = f_relative2 * (noise_level / signal_level_per_color)**2  # Side-step division-by-zero

    # Calculate the image restoration Wiener filter
    wiener_filter = np.conj(optical_transfer_function) / (np.abs(optical_transfer_function)**2 + nsr2)
    # Determine the deconvolution kernel (the filter in the spatial frequency domain is actually sufficient)
    deconvolution_kernel = ft.fftshift(ft.ifft2(ft.ifftshift(wiener_filter)))
    # Deconvolve the blurred image
    restored_img = convolve(blurred_noisy_img, deconvolution_kernel)

    return restored_img


def convolve(input_data, kernel):
    """
    TODO: Replace by  scipy.signal.fftconvolve(in1, in2, mode='full', axes=None)
    Convolve an input image with a kernel image.
    The output shape is the same as the size as the input.

    The intensity outside the edge of the input image is assumed to be the
    same as that of the nearest pixel. This avoids the darkened edges of
    zero-padded convolutions.

    The kernel must have its center pixel at the central row and column, or
    one more in case of an even number of columns or rows.

    Only works with real-values inputs

    :param input_data: The to-be-convolved input.
    :param kernel: The convolution kernel.
    :return: The convolved data
    """
    calculation_shape = input_data.shape[:2] + kernel.shape[:2] - 1  # This is overkill
    
    # Extend the  the pixels of the input up to the calculation shape size
    input_data = extend(input_data, calculation_shape)
    kernel = extend(kernel, calculation_shape)
    # Place the central pixel of the kernel in the top left corner
    kernel = np.roll(kernel, shift=-int(kernel.shape[:2] / 2), axis=(0, 1))
    
    # Convert to the spatial frequency domain for faster convolution
    input_ft = ft.fft2(input_data)
    kernel_ft = ft.fft2(kernel)
    
    # A convolution is a multiplication in the spatial frequency domain
    convolved_ft = input_ft * kernel_ft
    
    # Convert back to spatial domain
    convolved = ft.ifft2(convolved_ft)  # If both inputs are real-valued, so should be the output.
    
    # Crop again to the original shape
    convolved = convolved[:input_data.shape[0], :input_data.shape[1], :]
    return convolved


def extend(input_data, new_shape):
    """
    Extends the edge pixels outward to the larger shape,
    though rotate the pixels such that the top-left pixel stays in the same
    position.

    :param input_data: The to-be-extended data as an nd-array.
    :param new_shape: The new shape
    :return: The extended data as an nd-array.
    """
    # Replicate the pixels of the input up to the calculation shape size
    ranges = [np.clip(np.arange(n) - int(o/2), 0, o-1)
              for n, o in zip(new_shape, input_data.shape)]
    extended = input_data[ranges[0], ranges[1], :]

    return extended

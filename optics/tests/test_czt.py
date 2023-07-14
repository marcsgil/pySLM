import unittest
import numpy as np
import numpy.testing as npt

from optics.utils import ft
from optics.utils.ft import Grid
from optics.utils.ft.czt import czt, cztn, zoomft, zoomftn, izoomft, CZT


def pad(arr, magnifications):
    arr = np.array(arr)
    result = np.zeros(np.array(arr.shape) * np.array(magnifications), dtype=arr.dtype)
    result[tuple(slice(s) for s in arr.shape)] = ft.fftshift(arr)
    result = np.roll(result, shift=-(np.array(arr.shape) / 2).astype(int), axis=range(arr.ndim))
    return result


class TestCZT(unittest.TestCase):
    def test_czt_vs_fft_accuracy_for_long_lengths(self):
        rng = np.random.RandomState(seed=123)  # Deterministic randomness
        random_lengths = rng.exponential(100000, size=10).astype('int')
        for n in random_lengths:
            a = rng.randn(n)
            npt.assert_allclose(czt(a), ft.fft(a), rtol=1e-7, atol=1e-8)

    def test_int_args(self):
        integer_czt = czt([0, 1], 10, 2)
        npt.assert_allclose(integer_czt.imag, 0)
        npt.assert_allclose(integer_czt.real, 2**np.arange(10.0), rtol=6e-7)  # Can be a bit better when using ffts with a power-of-2 length.

    def test_czt_points(self):
        czt_points = CZT(phasor_0=1, phasor_ratio=2, nb_output_points=11).points
        npt.assert_allclose(czt_points, 1/(2**np.arange(11)))
        czt_points = CZT(phasor_0=-1j, phasor_ratio=-1j, nb_output_points=3).points
        npt.assert_allclose(czt_points, [-1j, 1, 1j])

    def test_czt_equal_to_fft_for_random_inputs(self):
        rtol = 1e-8  # todo: 1e-10 according to the scipy CZT branch

        def check_czt(x):
            # Check that czt is the equivalent of normal fft
            y = ft.fft(x)
            y1 = czt(x)
            err = np.linalg.norm(y-y1) / np.linalg.norm(y)
            npt.assert_allclose(y1, y, rtol=rtol)

            # Check that interpolated czt is the equivalent of the normal fft
            y = ft.fft(x, 100*len(x))
            y1 = czt(x, 100*len(x))
            npt.assert_allclose(y1, y, rtol=rtol)

        def check_zoomfft(arr):
            # Check that zoomfft is the equivalent of normal fft
            y = ft.fft(arr)
            y1 = zoomft(arr)
            npt.assert_allclose(y1, y, rtol=rtol)

            # Test fn scalar
            y1 = zoomft(arr)
            npt.assert_allclose(y1, y, rtol=rtol)

            # # Check that zoomfft with oversampling is equivalent to zero padding
            # arr = np.array([1, 0])
            # over = 1
            # yover = ft.fft(ft.ifftshift(arr), over * len(arr))
            # x = Grid(len(arr))
            # f = Grid(len(yover), extent=1, origin_at_center=False)  # ifftshifted
            # y2 = zoomft(arr, x=x, f=f)  #f=[0, 2-2./len(yover)], m=len(yover))
            # npt.assert_allclose(y2, yover, rtol=1e-10)
            #
            # # Check that zoomfft works on a subrange
            # w = np.linspace(0, 2-2./len(x), len(x))
            # f1, f2 = w[3], w[6]
            # y3 = zoomft(x, f=[f1, f2])  #, m=3*over+1)
            # idx3 = slice(3*over, 6*over+1)
            # npt.assert_allclose(y3, yover[idx3], rtol=1e-10)

        rng = np.random.RandomState(seed=0)  # Deterministic randomness
        # Random signals
        lengths = rng.randint(8, 200, 20)
        np.append(lengths, 1)
        for length in lengths:
            x = rng.rand(length)
            # yield check_zoomfft, x
            check_czt(x)
            check_zoomfft(x)

    def test_large_prime_lengths(self):
        rng = np.random.RandomState(seed=0)  # Deterministic randomness
        for N in (101, 1009, 10007):
            x = rng.rand(N)
            y = ft.fft(x)
            y1 = czt(x)
            npt.assert_allclose(y, y1, rtol=1e-9)

    def test_czt_defaults(self):
        npt.assert_array_equal(czt(3, 1, 1, 1), np.array([3]))
        npt.assert_array_equal(czt(4, 1, 1), np.array([4]))
        npt.assert_array_equal(czt(5, 1), np.array([5]))
        npt.assert_array_equal(czt(6), np.array([6]))

        npt.assert_almost_equal(czt([4, 1, 2, 3], 4, 1, 1), np.array([10, 10, 10, 10]))
        npt.assert_almost_equal(czt([4, 1, 2, 3], 4, 1), np.array([10, 10, 10, 10]), 12, 'default for f_0 is not 1.')
        npt.assert_almost_equal(czt([4, 1, 2, 3], 4), np.array([10, 2 + 2j, 2, 2 - 2j]))

    def test_czt(self):
        npt.assert_almost_equal(czt([4, 1, 2, 3], 2, 1, 1), np.array([10, 10]))
        npt.assert_almost_equal(czt([4, 1, 2, 3], 3, 1, 1), np.array([10, 10, 10]))
        npt.assert_almost_equal(czt([4, 1, 2, 3], 5, 1, 1), np.array([10, 10, 10, 10, 10]))
        npt.assert_almost_equal(czt([4, 1, 2, 3], 15, 1, 1), np.ones(15) * 10.0)
        npt.assert_almost_equal(czt([4, 1, 2, 3], 2), np.array([10, 2]))
        npt.assert_almost_equal(czt([4, 1], 3, np.exp(2j * np.pi / 4.0)), np.array([5, 4 + 1j, 3]))
        npt.assert_almost_equal(czt([[4], [1]], 3, np.exp(2j * np.pi / 4.0), axis=0), np.array([[5], [4 + 1j], [3]]))
        npt.assert_almost_equal(czt([[4, 1]], 3, np.exp(2j * np.pi / 4.0)), np.array([[5, 4 + 1j, 3]]))
        npt.assert_almost_equal(czt([4, 1, 2, 3], 2, 1, 1, input_origin_centered=True, output_origin_centered=True),
                                np.array([10, 10]))
        npt.assert_almost_equal(czt([1, 0, 0, 0], 4, 1, 1), np.array([1, 1, 1, 1]))
        npt.assert_almost_equal(czt([1, 1, 1, 1], 4, np.exp(-2j*np.pi / 4), 1,
                                    input_origin_centered=False, output_origin_centered=False),
                                np.array([4, 0, 0, 0]))
        npt.assert_almost_equal(czt([1, 1, 1, 1], 4, np.exp(-2j*np.pi / 4), 1,
                                    input_origin_centered=True, output_origin_centered=True),
                                np.array([0, 0, 4, 0]))
        npt.assert_almost_equal(czt([1, 1, 1, 1], 2, np.exp(-2j*np.pi / 2), 1), np.array([4, 0]))
        npt.assert_almost_equal(czt([1, 1, 1, 1], 2, np.exp(-2j*np.pi / 2), 1,
                                    input_origin_centered=True, output_origin_centered=True),
                                np.array([0, 4]))
        npt.assert_almost_equal(czt([1, 1, 1], 6, np.exp(-2j*np.pi / 6), 1,
                                    input_origin_centered=True, output_origin_centered=True),
                                np.array([-1, 0, 2, 3, 2, 0]))
        npt.assert_almost_equal(czt([0, 1, 1, 1], 8, np.exp(-2j*np.pi / 8), 1,
                                    input_origin_centered=True, output_origin_centered=True),
                                np.array([-1, 1-np.sqrt(2), 1, 1+np.sqrt(2), 3, 1+np.sqrt(2), 1, 1-np.sqrt(2)]))

    def test_cztn(self):
        npt.assert_almost_equal(cztn([4, 1, 2, 3], 2, 1, 1), np.array([10, 10]))
        npt.assert_almost_equal(cztn([4, 1, 2, 3], 3, 1, 1), np.array([10, 10, 10]))
        npt.assert_almost_equal(cztn([4, 1, 2, 3], 5, 1, 1), np.array([10, 10, 10, 10, 10]))
        npt.assert_almost_equal(cztn([4, 1, 2, 3], 15, 1, 1), np.ones(15)*10.0)
        npt.assert_almost_equal(cztn([4, 1, 2, 3], 2), np.array([10, 2]))
        npt.assert_almost_equal(cztn([4, 1], 3, np.exp(2j*np.pi/4.0)), np.array([5, 4+1j, 3]))
        npt.assert_almost_equal(cztn([[4], [1]], 3, np.exp(2j*np.pi/4.0), axes=0), np.array([[5], [4+1j], [3]]))
        npt.assert_almost_equal(cztn([[4, 1]], 3, np.exp(2j*np.pi/4.0), axes=1), np.array([[5, 4+1j, 3]]))

        npt.assert_almost_equal(cztn([[4], [1]], [3, 2], np.exp(2j*np.pi/4.0)), np.array([[5, 5], [4+1j, 4+1j], [3, 3]]))
        npt.assert_almost_equal(cztn([[4, 1], [4, 1]], 3, np.exp(2j*np.pi/4.0), axes=1), np.array([[5, 4+1j, 3], [5, 4+1j, 3]]))

        npt.assert_almost_equal(cztn([[4, 1], [4, 1]], [1, 3], np.exp(2j*np.pi/4.0), axes=None), np.array([[10, 8+2j, 6]]))
        npt.assert_almost_equal(cztn([[4, 1], [4, 1]], [1, 3], np.exp(2j*np.pi/4.0), axes=(0, 1)), np.array([[10, 8+2j, 6]]))


class TestZoomFT(unittest.TestCase):
    def test_as_fft(self):
        arr = [1, 0, 0, 0]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [1, 1, 1, 1]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [1, 0, 0]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [1, 1, 1]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [0, 1, 0, 0]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [0, 1, 0]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [0, 1, 2j, -3]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [0, 1, 2j]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [2, 3, 1]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))

        arr = [[0, 1, 2j, -3]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, 1, 2j]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, 1, 2j, -3], [-4j, 5, 6j, -7]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, 1, 2j], [-3, 4j, 5]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))

        arr = [[0], [1], [2j], [-3]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0], [1], [2j]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, -4j], [1, 5], [2j, 6j], [-3, -7]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, -3], [1, 4j], [2j, 5]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))

        arr = [[0, 1, 2j, -3]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, 1, 2j]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, 1, 2j, -3], [-4j, 5, 6j, -7]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, 1, 2j], [-3, 4j, 5]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))

        arr = [[0], [1], [2j], [-3]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0], [1], [2j]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, -4j], [1, 5], [2j, 6j], [-3, -7]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, -3], [1, 4j], [2j, 5]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))

    def test_as_scaled_fft(self):
        arr = [1, 2j, -3, 4j]
        x = Grid(shape=np.asarray(arr).shape, step=1, origin_at_center=False)
        npt.assert_array_almost_equal(zoomft(arr, x=x, f=x.f), ft.fft(arr))
        x = Grid(shape=np.asarray(arr).shape, step=2, origin_at_center=False)
        npt.assert_array_almost_equal(zoomft(arr, x=x, f=x.f), ft.fft(arr))
        x = Grid(shape=np.asarray(arr).shape, step=3.14, origin_at_center=False)
        npt.assert_array_almost_equal(zoomft(arr, x=x, f=x.f), ft.fft(arr))
        x = Grid(shape=np.asarray(arr).shape, step=0.1, origin_at_center=False)
        npt.assert_array_almost_equal(zoomft(arr, x=x, f=x.f), ft.fft(arr))
        arr_fftshifted = [-3, 4j, 1, 2j]
        x = Grid(shape=np.asarray(arr_fftshifted).shape, step=1)
        npt.assert_array_almost_equal(zoomft(arr_fftshifted, x=x, f=x.f), ft.fft(arr))
        x = Grid(shape=np.asarray(arr_fftshifted).shape, step=1e-6)
        npt.assert_array_almost_equal(zoomft(arr_fftshifted, x=x, f=x.f), ft.fft(arr))

    def test_x2(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=6),
                                      ft.fft(arr_padded))
        arr = [2, 1]
        arr_padded = [2, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2), ft.fft(arr_padded)[[0, -1]])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=4), ft.fft(arr_padded))
        arr_padded = [2, 0, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=3), ft.fft(arr_padded)[[0, -1]])
        npt.assert_array_almost_equal(zoomft(arr, magnification=3, length=6), ft.fft(arr_padded))
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=6),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(np.abs(zoomft(arr, f=[0, 1/6, 2/6, -3/6, -2/6, -1/6])),
                                      np.abs(ft.fft(arr_padded)))
        arr = [2, 1j]
        arr_padded = [2, 0, 0, 1j]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2), ft.fft(arr_padded)[[0, -1]])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=4), ft.fft(arr_padded))

    def test_f(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, f=[0, 1/6, 2/6, -3/6, -2/6, -1/6]), ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, f=[-3/6, -2/6, -1/6, 0, 1/6, 2/6]), ft.fftshift(ft.fft(arr_padded)))
        npt.assert_array_almost_equal(zoomft(arr, f=Grid(shape=6, step=1/6, origin_at_center=False)),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, f=Grid(shape=(6,), step=(1/6,), origin_at_center=(False,))),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, f=Grid(shape=[6], step=[1/6], origin_at_center=[False])),
                                      ft.fft(arr_padded))

    def test_x(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, -1]), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(ft.fftshift(arr), x=[-1, 0, 1]), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, 2, -3, -2, -1]), ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft([3, 0, 0, 0, 1, 2], x=[1, 2, 3, -2, -1, 0]), ft.fft(arr_padded))

    def test_1d(self):
        arr = [[2, 3, 1], [4, 8, -2]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(arr, magnification=1), ft.fft(arr))
        arr_padded = pad(arr, (1, 2))
        npt.assert_array_almost_equal(zoomft(arr, magnification=2), ft.fft(arr_padded)[np.ix_([0, -1], [0, 1, -1])])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=arr_padded.shape[-1]),
                                      ft.fft(arr_padded))

        npt.assert_array_almost_equal(zoomft(np.array(arr).swapaxes(0, 1), axis=1).swapaxes(0, 1), ft.fft(arr, axis=0))
        npt.assert_array_almost_equal(zoomftn(arr, axes=(0, )), ft.fft(arr, axis=0))

        npt.assert_array_almost_equal(zoomft(arr, magnification=1, axis=0), ft.fft(arr, axis=0))
        arr_padded = pad(arr, (2, 1))
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, axis=0),
                                      ft.fft(arr_padded, axis=0)[np.ix_([0, -1], [0, 1, -1])])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=arr_padded.shape[0], axis=0),
                                      ft.fft(arr_padded, axis=0))
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=arr_padded.shape[-2], axis=-2),
                                      ft.fft(arr_padded, axis=-2))

    def test_2d(self):
        arr = [[2, 3, 1], [4, 8, -2]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        npt.assert_array_almost_equal(zoomftn(arr, magnification=1), ft.fftn(arr))
        arr_padded = pad(arr, 2)
        npt.assert_array_almost_equal(zoomftn(arr, magnification=2), ft.fftn(arr_padded)[np.ix_([0, -1], [0, 1, -1])])
        npt.assert_array_almost_equal(zoomftn(arr, magnification=2, shape=arr_padded.shape),
                                      ft.fftn(arr_padded))


class TestIZoomIFT(unittest.TestCase):
    def test_as_ifft(self):
        arr = [1, 0, 0, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [1j, 0, 0, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 0, 1, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 0, 1j, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 1, 0, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 1j, 0, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [1, 1, 1, 1]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [1, 0, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [1, 1, 1]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 1, 0, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 1, 0]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 1, 2j, -3]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [0, 1, 2j]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))
        arr = [2, 3, 1]
        npt.assert_array_almost_equal(izoomft(arr), ft.ifft(arr))

        arr = [[0, 1, 2j, -3]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, 1, 2j]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, 1, 2j, -3], [-4j, 5, 6j, -7]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, 1, 2j], [-3, 4j, 5]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))

        arr = [[0], [1], [2j], [-3]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0], [1], [2j]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, -4j], [1, 5], [2j, 6j], [-3, -7]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        arr = [[0, -3], [1, 4j], [2j, 5]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))

        arr = [[0, 1, 2j, -3]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, 1, 2j]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, 1, 2j, -3], [-4j, 5, 6j, -7]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, 1, 2j], [-3, 4j, 5]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))

        arr = [[0], [1], [2j], [-3]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0], [1], [2j]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, -4j], [1, 5], [2j, 6j], [-3, -7]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        arr = [[0, -3], [1, 4j], [2j, 5]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))

    def test_x2(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=6),
                                      ft.fft(arr_padded))
        arr = [2, 1]
        arr_padded = [2, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2), ft.fft(arr_padded)[[0, -1]])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=4), ft.fft(arr_padded))
        arr_padded = [2, 0, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=3), ft.fft(arr_padded)[[0, -1]])
        npt.assert_array_almost_equal(zoomft(arr, magnification=3, length=6), ft.fft(arr_padded))
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=6),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(np.abs(zoomft(arr, f=[0, 1/6, 2/6, -3/6, -2/6, -1/6])),
                                      np.abs(ft.fft(arr_padded)))
        arr = [2, 1j]
        arr_padded = [2, 0, 0, 1j]
        npt.assert_array_almost_equal(zoomft(arr, magnification=2), ft.fft(arr_padded)[[0, -1]])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=4), ft.fft(arr_padded))

    def test_f(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, f=[0, 1/6, 2/6, -3/6, -2/6, -1/6]), ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, f=[-3/6, -2/6, -1/6, 0, 1/6, 2/6]), ft.fftshift(ft.fft(arr_padded)))
        npt.assert_array_almost_equal(zoomft(arr, f=Grid(shape=6, step=1/6, origin_at_center=False)),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, f=Grid(shape=(6,), step=(1/6,), origin_at_center=(False,))),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, f=Grid(shape=[6], step=[1/6], origin_at_center=[False])),
                                      ft.fft(arr_padded))

    def test_x(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, -1]), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(ft.fftshift(arr), x=[-1, 0, 1]), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, 2, -3, -2, -1]), ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft([3, 0, 0, 0, 1, 2], x=[1, 2, 3, -2, -1, 0]), ft.fft(arr_padded))

    def test_x_f(self):
        arr = [2, 3, 1]
        arr_padded = [2, 3, 0, 0, 0, 1]
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, -1], f=Grid.from_ranges([0, 1, -1]).f), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, -1], f=[0, 1/3, -1/3]), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(ft.fftshift(arr), x=[-1, 0, 1], f=[0, 1/3, -1/3]), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(ft.fftshift(arr), x=[-1, 0, 1], f=[-1/3, 0, 1/3]),
                                      ft.fftshift(ft.fft(arr)))
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, 2, -3, -2, -1], f=[0, 1/6, 2/6, -3/6, -2/6, -1/6]),
                                      ft.fft(arr_padded))
        npt.assert_array_almost_equal(zoomft(arr, x=[0, 1, 2, -3, -2, -1], f=[-3/6, -2/6, -1/6, 0, 1/6, 2/6]),
                                      ft.fftshift(ft.fft(arr_padded)))
        npt.assert_array_almost_equal(zoomft([3, 0, 0, 0, 1, 2],
                                             x=[1, 2, 3, -2, -1, 0], f=[0, 1/6, 2/6, -3/6, -2/6, -1/6]),
                                      ft.fft(arr_padded))

    def test_1d(self):
        arr = [[2, 3, 1], [4, 8, -2]]
        npt.assert_array_almost_equal(zoomft(arr), ft.fft(arr))
        npt.assert_array_almost_equal(zoomft(arr, magnification=1), ft.fft(arr))
        arr_padded = pad(arr, (1, 2))
        npt.assert_array_almost_equal(zoomft(arr, magnification=2), ft.fft(arr_padded)[np.ix_([0, -1], [0, 1, -1])])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=arr_padded.shape[-1]),
                                      ft.fft(arr_padded))

        npt.assert_array_almost_equal(zoomft(np.array(arr).swapaxes(0, 1), axis=1).swapaxes(0, 1), ft.fft(arr, axis=0))
        npt.assert_array_almost_equal(zoomftn(arr, axes=(0, )), ft.fft(arr, axis=0))

        npt.assert_array_almost_equal(zoomft(arr, magnification=1, axis=0), ft.fft(arr, axis=0))
        arr_padded = pad(arr, (2, 1))
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, axis=0),
                                      ft.fft(arr_padded, axis=0)[np.ix_([0, -1], [0, 1, -1])])
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=arr_padded.shape[0], axis=0),
                                      ft.fft(arr_padded, axis=0))
        npt.assert_array_almost_equal(zoomft(arr, magnification=2, length=arr_padded.shape[-2], axis=-2),
                                      ft.fft(arr_padded, axis=-2))

    def test_2d(self):
        arr = [[2, 3, 1], [4, 8, -2]]
        npt.assert_array_almost_equal(zoomftn(arr), ft.fftn(arr))
        npt.assert_array_almost_equal(zoomftn(arr, magnification=1), ft.fftn(arr))
        arr_padded = pad(arr, 2)
        npt.assert_array_almost_equal(zoomftn(arr, magnification=2), ft.fftn(arr_padded)[np.ix_([0, -1], [0, 1, -1])])
        npt.assert_array_almost_equal(zoomftn(arr, magnification=2, shape=arr_padded.shape),
                                      ft.fftn(arr_padded))


if __name__ == '__main__':
    unittest.main()

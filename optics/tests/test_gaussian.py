import unittest
import numpy.testing as npt

from optics.calc.gaussian import Gaussian, Hermite, Laguerre
# from optics.calc.special_beams import Airy
# from optics.calc.beam import Beam
from optics.utils import ft
from tests import log

import numpy as np


class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.wavelength = 500e-9  # in vacuum
        self.waists = [None, 1, 2, 10e-6]
        self.wavelengths = [None, 1, 500e-9]

        self.basis_beams = [
            Gaussian(waist=2),
            Laguerre(0, 1),
            Laguerre(0, 2),
            Laguerre(3, 2, waist=1),
            Hermite(2, -1),
            Laguerre(azimuthal_index=3, radial_index=1),
            Hermite(0),
            Hermite(2),
            Hermite(0, 4),
            Laguerre(1, 3)
        ]
        # beam = abs(Hermite(4) + Hermite(0) + Laguerre(2) + 2j * Gaussian()) ** 2]
        self.super_positions = [
            Gaussian(waist=2) + Gaussian(),
            Gaussian(waist=2) * 2 - Gaussian(),
            Laguerre(3, 2, waist=1) + Gaussian(),
            2 * Hermite(2) - 3 * Hermite(0) + Hermite(2, -1),
            3 * Laguerre(azimuthal_index=3, radial_index=1) + 2 * Laguerre(1, 3),
            Hermite(2, -1) + Laguerre(1, 3),
            Hermite(2, -1) + 2 * Laguerre(1, 3)
        ]
        self.all_beams = [*self.basis_beams, *self.super_positions]

    def test_Gaussian_construction(self):
        for waist in self.waists:
            for wavelength in self.wavelengths:
                desc = f'w={waist} lambda={wavelength}'
                if waist is None and wavelength is None:
                    beam = Gaussian()
                    waist = 1.0
                    wavelength = 1.0
                elif wavelength is None:
                    beam = Gaussian(waist=waist)
                    wavelength = 1.0
                elif waist is None:
                    beam = Gaussian(wavelength=wavelength)
                    waist = 1.0
                else:
                    beam = Gaussian(waist=waist, wavelength=wavelength)
                z = np.arange(-10 * wavelength, 10 * wavelength, wavelength / 10)[:, np.newaxis, np.newaxis]
                r = np.arange(0, beam.waist * 5, beam.waist / 10)
                npt.assert_equal(beam.waist, waist, f'Beam waist not set correctly for {desc}.')
                npt.assert_equal(beam.wavelength, wavelength, f'Beam wavelength not set correctly for {desc}.')
                npt.assert_equal(beam.wavenumber, 2 * np.pi / wavelength, f'Beam wavenumber not set correctly for {desc}.')
                npt.assert_almost_equal(beam.rayleigh_range, np.pi * waist**2 / wavelength,
                                        err_msg=f'Rayleigh range not calculated correctly for {desc}.')
                npt.assert_almost_equal(beam.divergence, np.arctan2(wavelength, np.pi * waist),
                                        err_msg=f'Beam divergence not calculated correctly for {desc}.')
                npt.assert_equal(beam.combined_order, 0, f'Total order of the {type(beam)} beam is not correct for {desc}.')
                npt.assert_equal(beam.gouy_phase_at(z), np.arctan2(z, beam.rayleigh_range),
                                 f'Gouy phase as a function of z of {type(beam)} beam is not correct for {desc}.')
                npt.assert_equal(beam.waist_at(z), beam.waist * np.sqrt(1 + (z / beam.rayleigh_range) ** 2),
                                 f'Beam waist as a function of z for {type(beam)} beam is not as expected for {desc}.')
                npt.assert_equal(beam.curvature_at(z), z / (z ** 2 + beam.rayleigh_range ** 2),
                                 f'Wavefront curvature as a function of z for {type(beam)} beam is not as expected for {desc}.')
                npt.assert_equal(beam.complex_beam_parameter_at(z), z + 1j * beam.rayleigh_range,
                                 f'Complex beam parameter as a function of z for {type(beam)} beam is not as expected for {desc}.')

                npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi) / beam.waist,
                                 err_msg=f'Field value at origin for the {type(beam)} beam is not correct for {desc}.')

                npt.assert_array_almost_equal(beam(r), np.sqrt(2 / np.pi) / beam.waist * np.exp(- (r / beam.waist) ** 2),
                                 err_msg=f'Transverse field value for the {type(beam)} beam is not correct for {desc}.')
                npt.assert_array_almost_equal(beam(r[:, np.newaxis], r), np.sqrt(2 / np.pi) / beam.waist * np.exp(- (r[:, np.newaxis] ** 2 + r ** 2) / beam.waist ** 2),
                                              err_msg=f'xy-transverse field value for the {type(beam)} beam is not correct for {desc}.')
                npt.assert_array_almost_equal(beam(z, 0, 0),
                                              np.sqrt(2 / np.pi) / beam.waist_at(z)
                                              * np.exp(1j * (beam.gouy_phase_at(z) - beam.wavenumber * z)),
                                              err_msg=f'On-axis field value for the {type(beam)} beam is not correct for {desc}.')
                npt.assert_array_almost_equal(beam(z, 0, r),
                                              np.sqrt(2 / np.pi) / beam.waist_at(z)
                                              * np.exp(- (r / beam.waist_at(z)) ** 2
                                                       - 1j * beam.wavenumber * (z + beam.curvature_at(z) / 2 * (r**2))
                                                       + 1j * beam.gouy_phase_at(z)
                                                       ),
                                              err_msg=f'Field values for {type(beam)} beam is not correct for {desc}.')
                npt.assert_array_almost_equal(beam.cylindrical(z, r, 0),
                                              np.sqrt(2 / np.pi) / beam.waist_at(z)
                                              * np.exp(- (r / beam.waist_at(z)) ** 2
                                                       - 1j * beam.wavenumber * (z + beam.curvature_at(z) / 2 * (r**2))
                                                       + 1j * beam.gouy_phase_at(z)
                                                       ),
                                              err_msg=f'Field values in cylindrical coordinates for {type(beam)} beam is not correct for {desc}.')

    def test_Hermite_construction(self):
        args_list = [
            [],
            [0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, -2],
        ]
        for indices in args_list:
            for waist in self.waists:
                for wavelength in self.wavelengths:
                    kwargs = dict()
                    if waist is not None:
                        kwargs['waist'] = waist
                    else:
                        waist = 1.0
                    if wavelength is not None:
                        kwargs['wavelength'] = wavelength
                    else:
                        wavelength = 1.0
                    desc = f'{indices}, ' + str(kwargs)
                    beam = Hermite(*indices, **kwargs)

                    npt.assert_equal(beam.waist, waist, f'Beam waist not set correctly for {desc}.')
                    npt.assert_equal(beam.wavelength, wavelength, f'Beam wavelength not set correctly for {desc}: {repr(beam)}.')
                    npt.assert_equal(beam.wavenumber, 2 * np.pi / wavelength, f'Beam wavenumber not set correctly for {desc}.')
                    npt.assert_almost_equal(beam.rayleigh_range, np.pi * waist**2 / wavelength,
                                            err_msg=f'Rayleigh range not calculated correctly for {desc}.')
                    npt.assert_almost_equal(beam.divergence, np.arctan2(wavelength, np.pi * waist),
                                            err_msg=f'Beam divergence not calculated correctly for {desc}.')
                    while len(indices) < 2:
                        indices.append(0)
                    npt.assert_equal(beam.combined_order, np.sum(np.abs(indices)), f'Total order of the {type(beam)} beam is not correct for {desc}.')
                    npt.assert_array_equal(beam.indices, indices, f'Indices not as specified for {type(beam)}: {desc}')
                    npt.assert_equal(beam.l_index, indices[-1], f'Indices not as specified for {type(beam)}: {desc}')
                    npt.assert_equal(beam.m_index, indices[-2], f'Indices not as specified for {type(beam)}: {desc}')

                    if np.any(beam.indices % 2 != 0):
                        npt.assert_almost_equal(beam(0, 0, 0), 0.0,
                                         err_msg=f"Hermite-Gaussian beams as {beam} should have an on-axis singularity as long as it has any odd index.")
                    else:
                        npt.assert_almost_equal(beam(0, 0, 0),
                                                np.sqrt(2.0 ** (1 - beam.combined_order)) / np.sqrt(np.pi * np.prod([np.math.factorial(_) for _ in beam.indices])) / beam.waist,
                                                err_msg=f"Hermite-Gaussian {beam} has incorrect on-axis field")

    def test_Laguerre_construction(self):
        kwargs_list = [
            {},
            {'radial_index': 0},
            {'azimuthal_index': 0},
            {'azimuthal_index': 0, 'radial_index': 0},
            {'azimuthal_index': 0, 'radial_index': 1},
            {'azimuthal_index': 1, 'radial_index': 0},
            {'azimuthal_index': 0, 'radial_index': 2},
            {'azimuthal_index': 2, 'radial_index': 0},
            {'azimuthal_index': 1, 'radial_index': 1},
            {'azimuthal_index': 1, 'radial_index': 2},
            {'azimuthal_index': 1, 'radial_index': -2},
        ]
        for kwargs in kwargs_list:
            for waist in self.waists:
                for wavelength in self.wavelengths:
                    kwargs_all = kwargs.copy()
                    if waist is not None:
                        kwargs_all['waist'] = waist
                    else:
                        waist = 1.0
                    if wavelength is not None:
                        kwargs_all['wavelength'] = wavelength
                    else:
                        wavelength = 1.0
                    desc = str(kwargs_all)
                    beam = Laguerre(**kwargs_all)

                    npt.assert_equal(beam.waist, waist, f'Beam waist not set correctly for {desc}.')
                    npt.assert_equal(beam.wavelength, wavelength, f'Beam wavelength not set correctly for {desc}: {repr(beam)}.')
                    npt.assert_equal(beam.wavenumber, 2 * np.pi / wavelength, f'Beam wavenumber not set correctly for {desc}.')
                    npt.assert_almost_equal(beam.rayleigh_range, np.pi * waist**2 / wavelength,
                                            err_msg=f'Rayleigh range not calculated correctly for {desc}.')
                    npt.assert_almost_equal(beam.divergence, np.arctan2(wavelength, np.pi * waist),
                                            err_msg=f'Beam divergence not calculated correctly for {desc}.')
                    indices = [kwargs.get('azimuthal_index', 0), kwargs.get('radial_index', 0)]
                    npt.assert_equal(beam.combined_order, abs(indices[0]) + 2 * abs(indices[1]), f'Total order of the {type(beam)} beam is not correct for {desc}.')
                    npt.assert_array_equal(beam.indices, indices, f'Indices not as specified for {type(beam)}: {desc}')
                    npt.assert_equal(beam.azimuthal_index, indices[-2], f'Indices not as specified for {type(beam)}: {desc}')
                    npt.assert_equal(beam.radial_index, indices[-1], f'Indices not as specified for {type(beam)}: {desc}')

                    if beam.azimuthal_index != 0:
                        npt.assert_equal(beam(0, 0, 0), 0.0,
                                         err_msg=f"Laguerre-Gaussian beams as {beam} should have a singularity for any l not equal to 0.")
                    else:
                        log.debug(f"Testing on-axis intensity for {repr(beam)}...")
                        abs_l = abs(beam.azimuthal_index)
                        p = abs(beam.radial_index)
                        npt.assert_equal(beam(0, 0, 0),
                                         np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + abs_l))) / beam.waist,
                                         err_msg="Laguerre-Gaussian beams as {beam} does not have the correct on-axis field.")

    def test_mul(self):
        beam = 1 * Gaussian(waist=1)
        npt.assert_almost_equal(beam(0, 0, 0).imag, 0)
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))
        beam = Gaussian() * 1
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))
        beam = 1.0 * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))
        beam = Gaussian() * 1.0
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))
        beam = 2 * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 2 * np.sqrt(2 / np.pi))
        beam = Gaussian() * 2
        npt.assert_almost_equal(beam(0, 0, 0), 2 * np.sqrt(2 / np.pi))
        beam = Gaussian() / 2
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi) / 2)

        beam = 0 * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 0)
        beam = Gaussian() * 0
        npt.assert_almost_equal(beam(0, 0, 0), 0)

        beam = Gaussian() * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 2 / np.pi)

        beam = 2 * Gaussian() / 2
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi),
                                err_msg=f"Product with scalars on both sides of Gaussian (1G) is not as expected: {beam}: {repr(beam)}")

        beam = 2 * Gaussian() * 3 * Gaussian() * 4
        npt.assert_almost_equal(beam(0, 0, 0), 24 * 2 / np.pi,
                                err_msg=f"Product with scalars on both sides of Gaussians (24G^2) is not as expected: {beam}: {repr(beam)}")

        beam = 2j * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 2j * np.sqrt(2 / np.pi),
                                err_msg=f"Product with imaginary 2i is not as expected: {beam}: {repr(beam)}")

        beam = (3 - 2j) * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), (3 - 2j) * np.sqrt(2 / np.pi),
                                err_msg=f"Product with imaginary (3 - 2i) is not as expected: {beam}: {repr(beam)}")

    def test_pow(self):
        beam = Gaussian() ** 1
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))

        beam = Gaussian() ** 2
        npt.assert_almost_equal(beam(0, 0, 0), 2 / np.pi)

        beam = Gaussian() ** 3
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi) ** 3)

        beam = Gaussian() ** np.pi
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi) ** np.pi)

        beam = Gaussian() ** 0
        npt.assert_almost_equal(beam(0, 0, 0), 1.0)

    def test_add(self):
        beam = Gaussian() + 0
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))

        beam = Gaussian() + Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 2 * np.sqrt(2 / np.pi))

        beam = Gaussian() + Gaussian() + Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 3 * np.sqrt(2 / np.pi))

        beam = Gaussian() - Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 0)

        beam = Gaussian() + 2 * Gaussian(waist=2)  # area: 2**2, but intensity also squared
        npt.assert_almost_equal(beam(0, 0, 0), 2 * np.sqrt(2 / np.pi))

        beam = Gaussian() + Hermite(3) + Hermite(1, 1) + Laguerre(1)
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi))

        beam = 2 + Gaussian()
        npt.assert_equal(str(beam), '2+G',
                         err_msg=f"sum of scalar and Gaussian (2 + G) does not have the expected description {beam}: {repr(beam)}")
        npt.assert_almost_equal(beam(0, 0, 0), 2 + np.sqrt(2 / np.pi),
                                err_msg=f"sum of scalar and Gaussian (2 + G) is not as expected: {beam}: {repr(beam)}")

        beam = 1 + Gaussian()
        npt.assert_equal(str(beam), '1+G',
                         err_msg=f"sum of scalar and Gaussians (1 + G) does not have the expected description {beam} {repr(beam)}")
        npt.assert_almost_equal(beam(0, 0, 0), 1 + np.sqrt(2 / np.pi),
                                err_msg=f"sum of scalar and Gaussians (1 + G) is not as expected: {beam}: {repr(beam)}")

        beam = Gaussian() + 1
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi) + 1,
                                err_msg=f"sum of scalars and Gaussians (G + 1) is not as expected: {beam}: {repr(beam)}")

        beam = Gaussian() + 2
        npt.assert_almost_equal(beam(0, 0, 0), np.sqrt(2 / np.pi) + 2,
                                err_msg=f"sum of scalars and Gaussians (G + 1) is not as expected: {beam}: {repr(beam)}")

        beam = 2 + Gaussian() + 3 + Gaussian() + 4
        npt.assert_almost_equal(beam(0, 0, 0), 9 + 2 * np.sqrt(2 / np.pi),
                                err_msg=f"sum of scalars and Gaussians (9 + 2G) is not as expected: {beam}: {repr(beam)}")

        beam = 2 + 3 * Gaussian()
        npt.assert_almost_equal(beam(0, 0, 0), 2 + 3 * np.sqrt(2 / np.pi),
                                err_msg=f"sum of scalars and scaled Gaussian (2 + 3G) is not as expected: {beam}: {repr(beam)}")

        beam = 2 + Gaussian() + 3
        npt.assert_almost_equal(beam(0, 0, 0), 2 + np.sqrt(2 / np.pi) + 3,
                                err_msg=f"sum of scalars and Gaussians (5 + G) is not as expected: {beam}: {repr(beam)}")

    def test_intensity(self):
        grid = ft.Grid(extent=[20, 10, 10], step=0.1)
        for beam in self.all_beams:
            desc = f"{beam}"
            fld = beam(*grid)
            intensity_grid = beam.intensity(grid)
            intensity_ranges = beam.intensity(*grid)
            npt.assert_array_equal(np.abs(fld) ** 2, intensity_grid,
                                   err_msg=f"Intensity did not return the square of the absolute field amplitude for {desc} using a Grid.")
            npt.assert_array_equal(np.abs(fld) ** 2, intensity_ranges,
                                   err_msg=f"Intensity did not return the square of the absolute field amplitude for {desc} using ranges.")

    def test_flux(self):
        grid = ft.Grid(extent=[30, 126, 128], step=[5, 0.25, 0.25])

        def flux(b):
            return np.sum(b.intensity(*grid), axis=(-2, -1)) * np.prod(grid.step[1:])

        for beam in self.basis_beams:
            log.debug(f"Testing: {repr(beam)}...")
            npt.assert_almost_equal(flux(beam), 1, decimal=4, err_msg=f"Flux of {beam} is not correct.")

        for beam in [Gaussian() + Laguerre(1), Gaussian() + Hermite(1), Hermite(1) + Hermite(2), Laguerre(1) + Laguerre(0, 1)]:
            log.debug(f"Testing: {repr(beam)}...")
            npt.assert_almost_equal(flux(beam), 2, decimal=4, err_msg=f"Flux of orthogonal superposition {beam} is not correct.")

        # beam = Laguerre(3, 2, waist=1)
        # beam = Hermite([2, -1])
        # beam = Laguerre(azimuthal_index=3, radial_index=1)
        # beam = Hermite(0) + Hermite(2)
        # beam = Hermite(0, 4) + Laguerre(1, 3) / 2
        # beam = abs(Hermite(4) + Hermite(0) + Laguerre(2) + 2j * Gaussian()) ** 2

        # beam = Laguerre(3, 3) * 2 + 1 * Airy([1, 2]) + Airy(1) + 1j * Airy(2) - Airy(3) -1j * Airy(4) - 1 * Airy(5) + 0 * Airy(7) - 0 * Airy(8)
        # beam = - Airy(8) * 3

    def test_incorrect_argument_shape(self):
        g = Gaussian()

        grid_2d = ft.Grid(extent=[20, 20], step=0.1)
        z_range = [-1, 0, 1, 2, 5, 10]
        ranges = [z_range, *grid_2d]

        npt.assert_raises_regex(ValueError,
                                "Make sure that the beam coordinates are three-dimensional vectors or arrays that broadcast.",
                                lambda: g(*ranges))

    def test_equivalence(self):
        g = Gaussian()
        h = Hermite()
        l = Laguerre()

        grid_2d = ft.Grid(extent=[20, 20], step=0.1)
        z_range = np.asarray([-1, 0, 1, 2, 5, 10])[:, np.newaxis, np.newaxis]
        ranges = [z_range, *grid_2d]

        fld_g = g(*ranges)
        npt.assert_array_almost_equal(h(*ranges), fld_g, err_msg="Basic Hermite Gaussian is not the basic Gaussian!")
        npt.assert_array_almost_equal(l(*ranges), fld_g, err_msg="Basic Laguerre Gaussian is not the basic Gaussian!")


if __name__ == '__main__':
    unittest.main()


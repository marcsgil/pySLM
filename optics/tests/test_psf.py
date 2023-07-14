import unittest
import numpy.testing as npt
import numpy as np

from optics.calc.psf import PSF, RichardsWolfPSF
from optics.instruments import objective
from optics.utils.ft import Grid

from tests import log


class TestBeamSection(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(4, 1e-6, center=1e-6) @ Grid(np.full(2, 64), 0.1e-6)
        self.objectives = [
            objective.InfinityCorrected(5, 0.05),
            objective.InfinityCorrected(20, 0.50),
            objective.InfinityCorrected(40, 0.80, refractive_index=1.33)
        ]
        self.vacuum_wavelength = 500e-9
        self.polarizations = [(1, 0), (0, 1), (1, 1), (1, 1j)]  # todo: 1 for scalar

        def tophat(nu_y, nu_x):
            return 1.0

        def cubic(nu_y, nu_x):
            return np.exp(2j * np.pi * 5 * (nu_y**3 + nu_x**3))

        self.scalar_pupil_functions = [1, tophat]  #todo: , cubic]

    def test_psf(self):
        for obj in self.objectives:
            f_max = obj.numerical_aperture / self.vacuum_wavelength
            pupil_grid = self.grid.project(axes_to_remove=-3).f / f_max
            for scalar_pupil_function in self.scalar_pupil_functions:
                for polarization in self.polarizations:
                    desc = f'{obj}, polarization={polarization}'
                    def pupil_function(nu_y, nu_x):
                        if callable(scalar_pupil_function):
                            result = scalar_pupil_function(nu_y, nu_x)
                        else:
                            result = np.full(np.broadcast_shapes(np.asarray(nu_y).shape, np.asarray(nu_x).shape), scalar_pupil_function)
                        return result * np.asarray(polarization)[..., np.newaxis, np.newaxis]
                    psf = PSF(objective=obj, vacuum_wavelength=self.vacuum_wavelength, pupil_function=pupil_function)
                    # reference_psf = RichardsWolfPSF(objective=obj, vacuum_wavelength=self.vacuum_wavelength, pupil_function=1.0)

                    # Define the point-spread function sampling
                    grid = Grid(4, 1e-6, center=1e-6) @ Grid(np.full(2, 64), 0.1e-6)

                    # Calculate the PSF
                    field_array = psf(*self.grid)

                    # Calculate the reference PSF
                    # psf = RichardsWolfPSF(objective=obj, vacuum_wavelength=wavelength, pupil_function=1.0)
                    # reference_field_array = reference_psf(*self.grid)
                    from pyotf import otf
                    ref_psf = otf.HanserPSF(wl=self.vacuum_wavelength, na=obj.numerical_aperture, ni=obj.refractive_index,
                                            res=grid.step[-1], size=grid.shape[-1].item(), zres=grid.step[0], zsize=grid.shape[0].item(),
                                            vec_corr='total', condition='sine')
                    # ref_psf.apply_pupil(pupil_function(*pupil_grid)[-2])
                    fld6x = fld6y = ref_psf.PSFa.conj()[:, ::-1, :, :]  # Seems to propagate in the other direction than otf.SheppardPSF
                    # ref_psf.apply_pupil(pupil_function(*pupil_grid)[-1])
                    # fld6x = ref_psf.PSFa.conj()[:, ::-1, :, :]  # Seems to propagate in the other direction than otf.SheppardPSF
                    reference_field_array = fld6x[::2] * polarization[-1] + fld6y[1::2] * polarization[-2]

                    log.info(f'Calculating reference PSF for {desc}...')
                    npt.assert_almost_equal(np.linalg.norm(field_array), np.linalg.norm(reference_field_array),
                                            err_msg=f'Point-spread functions have different norm for {desc}.')
                    npt.assert_array_almost_equal(np.abs(field_array), np.abs(reference_field_array),
                                                  err_msg=f'Point-spread functions have different intensities for {desc}.')
                    npt.assert_array_almost_equal(field_array, reference_field_array,
                                                  err_msg=f'Point-spread functions are not equal for {desc}.')


if __name__ == '__main__':
    unittest.main()


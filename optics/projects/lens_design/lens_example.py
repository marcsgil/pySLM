from projects.lens_design import log

import numpy as np
import matplotlib.pyplot as plt

from optics.utils.ft import Grid
from optics.experimental.lens_design.optic import SingletLens, MultipletLens, CompoundOptic, OpticalSystem
from optics.experimental.lens_design.material import Material
from optics.experimental.lens_design.surface import PlanarSurface, SphericalSurface
from optics.experimental.lens_design.light import Wavefront

import torch

if __name__ == '__main__':
    glass = Material(refractive_index=1.5)

    detector_surface = PlanarSurface(position=200e-3, diameter=50e-3)

    singlet_lens_1 = SingletLens(material=glass,
                                 front_surface=PlanarSurface(),
                                 rear_surface=SphericalSurface(radius_of_curvature=-25e-3), thickness=5e-3, diameter=25e-3)
    singlet_lens_1.position = 50e-3
    singlet_lens_2 = SingletLens(material=glass,
                                 front_surface=SphericalSurface(radius_of_curvature=25e-3),
                                 rear_surface=PlanarSurface(), thickness=5e-3, diameter=25e-3)
    singlet_lens_2.position = 150e-3
    telescope = CompoundOptic(optics_and_materials=[singlet_lens_1, singlet_lens_2])
    # telescope = singlet_lens_1

    wavelength = 500e-9
    k0 = 2 * np.pi / wavelength
    E = [1.0, 0, 0]
    grid = Grid([1, 5, 1], extent=0.5, center=[0, 0, 1])
    direction = np.stack(np.broadcast_arrays(*grid), axis=2)[..., 0].swapaxes(0, 1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)

    # Define the source wave
    view_positions = [[0, 0, 5e-3], [0, 2.5e-3, 5e-3], [0, 5e-3, 5e-3]]
    # view_positions = view_positions[1]
    wave = Wavefront(E=E, k0=k0, direction=direction,  p=view_positions)

    lens_system = OpticalSystem(source=wave, optic=telescope, detector=detector_surface,
                                metric=lambda _: torch.std(_.propagate().p[..., :2]))

    wave = lens_system.propagate()
    log.info(f'System evaluation cost = {lens_system.evaluate()*1e3:0.6f}mm RMS error.')

    # lens_system.optimize()
    wave = lens_system.propagate()
    log.info(f'System evaluation cost = {lens_system.evaluate()*1e3:0.6f}mm RMS error.')

    plt.figure(figsize=(12, 8), frameon=False)
    ax = plt.gca()
    lens_system.plot(ax)
    wave.plot(ax)
    ax.set(xlabel='z  [m]', ylabel='y  [m]')

    plt.show()

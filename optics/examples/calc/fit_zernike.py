#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt

from examples.calc import log
from optics.utils import pol2cart
from optics.calc import zernike

if __name__ == '__main__':
    spherical = zernike.BasisPolynomial(n=4, m=0)
    # aberration = zernike.BasisPolynomial(n=0, m=0)  # piston
    # aberration = zernike.BasisPolynomial(n=1, m=-1)  # tilt
    # aberration = zernike.BasisPolynomial(n=2, m=0)  # defocus
    aberration = zernike.BasisPolynomial(n=3, m=3)  # trefoil

    nb_subdivisions = 100
    rng = np.linspace(-1, 1, nb_subdivisions)
    rho = np.linspace(0, 1, nb_subdivisions)[:, np.newaxis]
    phi = np.linspace(-np.pi, np.pi, nb_subdivisions + 1)[np.newaxis, :]
    y, x = pol2cart(rho, phi)

    fitted_coefficients = zernike.fit(z=aberration(rho, phi), rho=rho, phi=phi, order=15).coefficients

    log.info(f'A quick test with {spherical.name} aberration at rho = 0, 0.5, and 1: {spherical([0, 0.5, 1])}')
    log.info(f'The other aberration is {aberration.name}, ' +
             f'with standard Zernike coefficients: {", ".join([f"{c:0.0f}" for c in fitted_coefficients])}.')

    #
    # Display
    #
    fig = plt.figure(figsize=(10, 8))
    axs = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2, projection="3d")]

    axs[0].plot(rng, spherical(rng))
    axs[0].set(xlabel='$\\rho$', ylabel='$w_{2,0}$')
    axs[1].plot_surface(x, y, aberration(rho, phi))
    axs[1].set(xlabel='x', ylabel='y', zlabel=f'$Z_{{{aberration.n}^{aberration.m}}}$')

    plt.show(block=True)

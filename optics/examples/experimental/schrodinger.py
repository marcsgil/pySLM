"""
Solves the non-relativistic Schrödinger eigen-equation of the form (- h_bar^2/(2m) \nabla^2 + V)\psi = E\psi for the lowest real values E.
The potential V can be defined on regularly spaced 1D or 2D grids (no examples for 3D are included, the display code would need to change).
"""
import numpy as np
import scipy.sparse.linalg as spa
# import scipy.constants as const
from dataclasses import dataclass
import time
import pathlib
import matplotlib.pyplot as plt
import PIL
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'
plt.rcParams['ps.fonttype'] = 'truetype'

from optics.utils import ft
from optics.utils.ft import Grid
from optics.utils.display import complex2rgb, grid2extent, format_image_axes
from optics.utils.display import complex_color_legend
from examples import log

h_bar = 1  # const.hbar
m = 1  # const.m_e * const.m_p / (const.m_e + const.m_p)  # two-body reduced mass


# Diagnostic helper functions that calculate the L2 operator norm, condition number, and the infimum of the real part of Hermitian operators.
def norm(_):
    return np.abs(spa.eigsh(_, 1, which='LM')[0][0])


def norm_inv(_):
    return 1.0 / np.abs(spa.eigsh(_, 1, which='SM')[0][0])


def cond(_):
    return norm(_) * norm_inv(_)


def real(_):
    return spa.eigsh(_, 1, which='SR')[0][0]


if __name__ == '__main__':
    output_path = pathlib.Path(__file__).parent.absolute() / 'output'

    subplot_shape = [2, 3]  # number of rows and columns, respectively, in output plot.
    calc_diagnostics = False
    show_legends = False

    # grid = Grid(np.full(1, 1024), extent=10)
    grid = Grid(np.full(2, 128), extent=10)  # or 512
    problem_shape = (grid.size, grid.size)
    if calc_diagnostics:
        log.info(f'Calculation grid {grid.shape} requires matrices of {problem_shape}.')

    # Maps between n-dimensional 'fields' and vectors
    def a2v(_):
        return _.ravel()

    def v2a(_):
        return _.reshape(grid.shape)

    class Operator(spa.LinearOperator):
        """A class to represent linear operators that keep track of the number of evaluations."""
        def __init__(self, matvec, rmatvec=None):
            super().__init__(dtype=np.complex128, shape=problem_shape)
            self.__matvec_function = matvec
            self.__rmatvec_function = rmatvec if rmatvec is not None else matvec

        def _matvec(self, _):
            return self.__matvec_function(_)

        def _rmatvec(self, _):
            return self.__rmatvec_function(_)

        def _adjoint(self):
            return Operator(self.__rmatvec_function, self.__matvec_function)

    class HermitianOperator(Operator):
        """A class to allow use to represent functions as Hermitian matrices."""
        def __init__(self, matvec):
            super().__init__(matvec)

        def _adjoint(self):
            return self

    # Define some useful operators
    identity = HermitianOperator(lambda _: _)  # The identity operator
    op_ft = Operator(matvec=lambda _: a2v(ft.fftn(v2a(_))), rmatvec=lambda _: a2v(ft.ifftn(v2a(_))))  # Forward Discrete Fourier Transform
    op_ift = op_ft.adjoint()

    log.info('Defining the forward problem...')
    r = np.sqrt(sum(_**2 for _ in grid))
    k2 = sum(_**2 for _ in grid.k)

    # Laplacian term in Fourier space
    solve_in_fourier_space = False
    l_ft = k2 * h_bar**2 / (2 * m)

    # Harmonic potential
    # k = m  # to keep things simple
    # potential = (k/2) * r ** 2  # Should have E = (n + 1/2) h_bar omega with omega^2 = k/m

    # Infinity resonator
    min_extent = np.amin(grid.extent)
    r1 = np.sqrt((grid[0] / min_extent) ** 2 + (grid[1] / min_extent + 0.1) ** 2)
    r2 = np.sqrt((grid[0] / min_extent) ** 2 + (grid[1] / min_extent - 0.1) ** 2)
    potential = (np.abs(r1 - 0.1) > 0.01) * (np.abs(r2 - 0.1) > 0.01)
    potential = -10 * np.logical_not(potential).astype(np.float32)

    # # Finite square well (cylinder in 2D, sphere in 3D)
    # potential = -100.0 * (r < 0.5 * np.amax(r))  # Should have E = (n h / L)**2 / (8 * m)  # Ground state varies with infinity-approx

    # # Double square well
    # potential = -100.0 * (np.abs(r - 0.5 * np.amax(r)) < 0.25 * np.amax(r))

    # # Double harmonic (in 1D)
    # potential = 1.0 * np.abs(r - 0.5 * np.amax(r))**2

    # # Quartic, results in lowest states E0-Et and E0+Et, where Et scales with the barrier height.
    # r0 = np.amax(r) / 1.5
    # potential = 100.0 * ((r / r0)**4 - (r / r0)**2)
    # potential -= np.amin(potential)

    # # Random periodic
    # rng = np.random.RandomState(seed=1)
    # def lowpass(x):
    #     rf = np.sqrt(sum(_**2 for _ in grid.f))
    #     return ft.ifftn(ft.fftn(x) * (rf < np.amax(rf) / 2)).real
    # potential = 10.0 * lowpass(rng.randn(*grid.shape))

    # # Hydrogen, TODO: should have a pre-factor const.elementary_charge**2 / (4 * const.pi * const.epsilon_0)
    # potential = - 10.0 / np.maximum(r,  1e-6)
    # # Only for very large potential wells this works better in Fourier space.
    # # solve_in_fourier_space = True  # Doesn't help

    #
    # Solve the eigenvector problem
    #
    log.info('Normalizing the potential.')
    target_min_h = 0.5 * np.sqrt(np.amax(k2) * h_bar / (2 * m))  # todo: how much to add? Proportional to ||A||?
    potential_min = np.amin(potential) - target_min_h  # Make sure that the potential is positive so that the system is accretive (because Re[L] = 0).
    positive_potential = potential - potential_min
    # Find the potential bias that will minimize ||V||
    potential_center = 0.5 * (np.amin(positive_potential.real) + np.amax(positive_potential.real))  # The center
    # Determine what ||V|| should be scaled to by calculating S = |V| |H^{-1}|, with H = A_raw
    op_h = HermitianOperator(lambda _: a2v(ft.ifftn((k2 * h_bar / (2 * m) ) * ft.fftn(v2a(_))) + positive_potential * v2a(_)))
    if target_min_h > 0:
        norm_inv_h_estimate = 1 / target_min_h  # norm_inv(op_h)
        problem_scale = norm_inv_h_estimate * np.amax(np.abs(positive_potential - potential_center))
        target_norm_v = np.sqrt(problem_scale) / (np.sqrt(problem_scale) + np.sqrt(2))
    else:
        problem_scale = np.inf
        target_norm_v = 1.0
    log.info(f'S = |A^{-1}| |V| = {problem_scale:0.3f}, so |V| should be scaled to {target_norm_v:0.3f}.')

    if solve_in_fourier_space:
        log.info('Solving in Fourier space...')
        positive_potential, l_ft = l_ft, positive_potential

    potential_center = 0.5 * (np.amin(positive_potential.real) + np.amax(positive_potential.real))  # The center of the enclosing square
    scale = np.amax(np.abs(positive_potential - potential_center)) / target_norm_v
    log.info('scale = {:.1f}.'.format(scale))

    op_l = op_ift @ HermitianOperator(lambda _: a2v(l_ft + potential_center) * a2v(_) / scale) @ op_ft
    op_v = HermitianOperator(lambda _: a2v(positive_potential - potential_center) * a2v(_) / scale)
    op_a = op_l + op_v

    op_b = identity - op_v
    op_apb_inv = op_ift @ HermitianOperator(lambda _: _ * scale / a2v(l_ft + potential_center + scale)) @ op_ft  # L + 1

    alpha = 1.0  # Should be (close to) 1 for Hermitian problems
    prec = HermitianOperator(lambda _: alpha * op_b @ op_apb_inv @ _)
    prec_a = alpha * op_b @ (identity - op_apb_inv @ op_b)  # = prec @ op_a
    op_m = identity - prec_a

    if calc_diagnostics:
        # log.info('Checking (A+B)^-1(A+B)...')
        # cond_1 = cond(op_apb_inv @ (op_a + op_b))
        # if np.abs(cond_1 - 1) > 1e-3:
        #     log.error(f'cond((A+B)^-1(A+B)) = {cond_1:0.3f}, the inverse operator op_apb_inv is incorrect!')

        log.info('Evaluating pre-conditioning...')
        norm_v = norm(op_v)
        log.info(f'||V|| = {norm_v:0.6f}')
        if norm_v > 1:
            log.error(f'V is not a contraction! ||V|| > 1')
        min_real_a = real(op_a)
        if min_real_a < 0:
            log.error(f'The operator A = L + V is not accretive! (minimum value = {min_real_a})')
            log.error(f'(maximum value = {-real(-op_a)})')
            min_real_l = real(op_l)
            if min_real_l < 0:
                log.error(f'The operator L is not accretive! (minimum value = {min_real_l})')
            min_real_v = real(op_v)
            if min_real_v < 0:
                log.error(f'The operator V is not accretive! (minimum value = {min_real_v})')
        norm_m = norm(op_m)
        log.info(f'||M|| = {norm_m:0.3f}')
        if norm_m > 1:
            log.error('The preconditioned system will not converge because M is not a contraction!')
        cond_a = cond(op_a)
        cond_prec_a = cond(prec_a)
        log.info(f'cond(A) = {cond_a:0.3f}')
        log.info(f'cond(prec_A) = {cond_prec_a:0.3f}, a {cond_a / cond_prec_a:0.3f}x improvement by preconditioning.')

    @dataclass
    class NbEvals:
        A: int = 0
        prec: int = 0

    nb_evaluations = NbEvals()

    def fixed_point_iteration(a, b, x0=None, alpha=1.0, tol=1e-9, maxiter=1e4, callback=None, atol=None):
        if atol is not None:
            tol = atol / np.linalg.norm(b)

        dx = b
        if x0 is not None:
            dx -= a @ x0
        x = alpha * dx
        iter = 1
        while (maxiter is None or iter < maxiter) and tol < np.linalg.norm(dx):
            dx = b - a @ x
            x += alpha * dx
            iter += 1

        if maxiter is None or iter < maxiter:
            info = 0
        else:
            info = maxiter

        return x, info

    def op_a_inv_matvec(_):
        def evaluated_prec(_=None):
            nb_evaluations.prec += 1

        # psi, info = sla.cgs(op_a, _, x0=None, tol=1e-6, maxiter=None, M=None, callback=None, atol=None)
        evaluated_prec()
        # psi, info = sla.cg(prec_a, prec(_), x0=None, tol=1e-9, maxiter=None, M=None, callback=evaluated_prec, atol=None)
        # psi, info = sla.cgs(prec_a, prec(_), x0=None, tol=1e-9, maxiter=None, M=None, callback=evaluated_prec, atol=None)
        # psi, info = sla.bicgstab(prec_a, prec(_), x0=None, tol=1e-9, maxiter=None, M=None, callback=evaluated_prec, atol=None)
        psi, info = fixed_point_iteration(prec_a, prec(_), x0=None, tol=1e-9, maxiter=None, callback=evaluated_prec, atol=None)
        nb_evaluations.A += 1
        if info != 0:
            if info > 0:
                log.error(f'Inversion did not converge in {info} iterations!')
            else:
                log.error(f'Inversion failed, error {info}!')
        return psi

    op_a_inv = HermitianOperator(op_a_inv_matvec)

    #
    # The calculation of the eigenmodes and vectors happens now
    #
    nb_solutions = np.prod(subplot_shape) - 1
    log.info(f'Calculating {nb_solutions} eigenvectors corresponding to lowest energy states...')
    start_time = time.perf_counter()
    if isinstance(op_a_inv, HermitianOperator):
        energies, vectors = spa.eigsh(op_a_inv, k=nb_solutions, M=None, sigma=0,
                                      which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
                                      Minv=None, OPinv=op_a_inv, mode='normal')  # Hermitian (Lanczos) algorithm
        # energies, vectors = spa.lobpcg(A=op_a_inv, X=np.random.randn(op_a_inv.shape[1], nb_solutions),
        #                                maxiter=20, tol=0, largest=True, verbosityLevel=3)
    else:
        energies, vectors = spa.eigs(op_a_inv, k=nb_solutions, M=None, sigma=0,
                                     which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
                                     Minv=None, OPinv=op_a_inv)  # Non-Hermitian (Implicit restarted Arnoldi) algorithm

    # Undo the scaling
    energies = energies * scale + potential_min
    if solve_in_fourier_space:
        log.info('Converting solutions back to real space...')
        for _ in range(vectors.shape[1]):
            vectors[:, _] = a2v(ft.ifftn(v2a(vectors[:, _])))
        op_a = op_ift @ op_a @ op_ft

    log.info(f'Calculation time {time.perf_counter() - start_time:0.6f} s, by inverting A {nb_evaluations.A} times and evaluating the preconditioner {nb_evaluations.prec} times.')

    log.info('Sorting eigenvectors by ascending eigenvalue...')
    ascending_eigenvalue_indices = np.argsort(energies)
    energies = energies[ascending_eigenvalue_indices]
    vectors = vectors[:, ascending_eigenvalue_indices]
    log.info('Phase-normalizing results for display...')
    for _ in range(vectors.shape[1]):
        vector = vectors[:, _]
        current_phase = np.angle(vector[np.argmax(np.abs(vector))])
        vectors[:, _] = vector * np.exp(-1j * current_phase)

    log.info('Checking results...')
    op_h = op_a * scale + identity * potential_min  # Determine H of the non-scaled problem
    relative_errors = np.linalg.norm((op_h @ vectors) - (vectors * energies), axis=0) / np.linalg.norm(vectors * energies, axis=0)
    max_rel_error = np.amax(relative_errors)
    if max_rel_error > 1e-6:
        log.error(f'Maximum relative error: {max_rel_error}')

    log.info('Displaying results...')
    white_background = True
    fig, axs = plt.subplots(*subplot_shape, sharex='all', sharey='all' if grid.ndim > 1 else 'none',
                            frameon=False, figsize=tuple(3 * _ for _ in reversed(subplot_shape)))
    fig.tight_layout(pad=0.0)
    image_datas = []
    for ax_idx, ax in enumerate(axs.ravel()):
        result_idx = ax_idx - 1
        if ax_idx == 0:  # Display the potential
            title = f'V ($\\Delta$ {np.amax(potential) - np.amin(potential):0.1f}, c: {ft.ifftshift(potential).ravel()[0] - np.amin(potential):0.1f})' if show_legends else None
            if grid.ndim == 1:
                ax.fill_between(grid[0], potential, np.amin(potential) - (np.amax(potential) - np.amin(potential)) / 10,
                                linewidth=0, alpha=0.5, facecolor='#008080')
                if show_legends:
                    ax.set(title=title)
            else:
                # image_data = complex2rgb(potential, normalization=1, inverted=True)
                # ax.imshow(image_data, extent=grid2extent(grid))
                image_data = potential != potential[potential.shape[0]//2, potential.shape[0]//2]
                ax.imshow(image_data, extent=grid2extent(grid), cmap='gray')
                format_image_axes(ax, title=title, scale=grid.extent[1] / 5 * show_legends, white_background=white_background)
            if show_legends:
                # Add a color legend to the potential plot
                legend_size = 1/3 / np.array(subplot_shape)
                ax_legend = fig.add_axes([1/subplot_shape[1] - legend_size[1], 1 - legend_size[0], legend_size[1], legend_size[0]])
                complex_color_legend.draw(ax_legend)
        else:  # Display the ground states
            if result_idx < energies.size:
                energy = energies[result_idx]
                psi = v2a(vectors[:, result_idx])
                title = f'E = {energy:0.3f} ($\\times${energy / energies[0]:0.1f})' if show_legends else None
                if grid.ndim == 1:
                    ax.plot(grid[0], psi.real, linewidth=2)
                    ax.plot(grid[0], psi.imag, linewidth=2)
                    if show_legends:
                        ax.set(title=title)
                else:
                    image_data = complex2rgb(psi, normalization=1, inverted=white_background)
                    ax.imshow(image_data, extent=grid2extent(grid))
                    format_image_axes(ax, title=title, scale=grid.extent[1] / 5 * (ax_idx == axs.size-1) * show_legends,
                                      white_background=white_background)
            else:
                ax.set_visible(False)
        image_datas.append(image_data)

    log.info(f'complex E = {energies}')
    log.info('E = ' + ', '.join(f'{_:0.3f}' for _ in energies.real))

    plt.show(block=False)
    plt.pause(0.01)

    file_name = pathlib.Path(__file__).name[:-3]
    log.info(f'Saving to {output_path / file_name}...')
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
    fig.savefig(output_path / (file_name + '.png'), bbox_inches='tight', format='png', transparent=True)
    fig.savefig(output_path / (file_name + '.svg'), bbox_inches='tight', format='svg', transparent=True)
    fig.savefig(output_path / (file_name + '.pdf'), bbox_inches='tight', format='pdf', transparent=True)
    for _, image_data in enumerate(image_datas):
        PIL.Image.fromarray((image_data * 255.5).astype(np.uint8)).save(output_path / (file_name + f'_{_}.png'))

    log.info('Done.')

    plt.show(block=True)

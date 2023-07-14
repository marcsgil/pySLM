from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

from optics.utils import ft
from optics.utils.ft.czt import interp
from optics.utils.display import colormap, colorbar, complex2rgb, grid2extent
from optics import log


def background(shape):
    grid = ft.Grid(shape[:2])
    checker_board = sum((_//16) % 2 for _ in grid) % 2
    checker_board = np.repeat(checker_board[..., np.newaxis], 3, axis=-1).astype(float)
    return checker_board


def alpha_blend(*images):
    result = 0.0
    for _ in images:
        if _.shape[-1] > 3:
            alpha = np.clip(_[..., 3:], 0.0, 1.0)
            result *= 1.0 - alpha
            result += _[..., :3]
        else:
            result = _[..., :3]
    return result


def overlay(img):
    return alpha_blend(background(img.shape[:2]), img)


if __name__ == '__main__':
    show_phase = False
    up_sampling_factor = 1
    saturation = 4.0

    # input_path = Path.home() / 'Downloads/3d_scattering_833_E_conjugated.npy'
    # output_path = input_path.parent / (input_path.stem + '_slices.npz')
    #
    # log.info(f'Loading data from {input_path}...')
    folder = r"G:\My Drive\data\3d_scattering/"
    left_side = np.load(folder + "cube_corner0.npy", mmap_mode='r')
    right_side = np.load(folder + "cube_corner1.npy", mmap_mode='r')
    # input_path = r"G:\My Drive\data\3d_scattering" + r"\3d_scattering_833_E_conjugated.npy"
    # fld = np.load(input_path, mmap_mode='r')
    # fld = fld[0]  # Drop the polarization dimension
    # log.info(f'field.shape = {fld.shape}, taking slices through the center...')
    # slices = [fld.swapaxes(0, _)[fld.shape[_]//2].astype(np.complex64) for _ in range(fld.ndim)]
    slices = [left_side, right_side]


    # log.info(f'Saving slices to {output_path}...')
    # np.savez(output_path, slices=slices)

    log.info(f'Upsampling {len(slices)} slices by a factor of {up_sampling_factor}...')
    interpolated_slices = [interp(_, up_sampling_factor) for _ in slices]
    log.info(f'Normalizing {len(interpolated_slices)} slices...')
    # normalized_slices = [_ / np.amax(np.abs(_)) for _ in interpolated_slices]
    normalized_slices = [np.log(np.abs(_ ** 2)) for _ in interpolated_slices]

    log.info('Mapping colors...')
    scaled_slices = [_ / np.amax(_) for _ in normalized_slices]
    # scaled_slices = [(np.abs(_)**(1/3)) * np.exp(1j * np.angle(_)) for _ in normalized_slices]
    # scaled_slices = [np.log(_) for _ in scaled_slices]
    # scaled_slices = [_ / np.amax(_) for _ in scaled_slices]
    if show_phase:
        slices_rgba = [complex2rgb(_, normalization=saturation, alpha=1.0) for _ in scaled_slices]
    else:
        # cmap = mpl.cm.get_cmap('viridis')
        # cmap = colormap.InterpolatedColorMap('with_alpha', [(0, 0, 0, 0),
        #                                                  (1, 0, 0, 0.1),
        #                                                  (0.75, 0.75, 0.0, 0.2),
        #                                                  (0, 1, 0, 0.3),
        #                                                  (0, 0.75, 0.75, 0.4),
        #                                                  (0, 0, 1, 0.5),
        #                                                  (0.75, 0, 0.75, 0.6),
        #                                                  (1, 1, 1, 1)],
        #                                      points=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1])
        cmap = colormap.InterpolatedColorMap('with_alpha', [(0, 0, 0, 0),
                                                            (1, 0.25, 0.25, 0.4),
                                                            (1, 1, 0.25, 1),
                                                            (0.25, 1, 0.25, 1),
                                                            (0.25, 1, 1, 1),
                                                            (0.25, 0.25, 1, 1),
                                                            (1, 0.25, 1, 1),
                                                            (1, 1, 1, 1)],
                                             points=[0, *np.linspace(0.5, 0.83, 6), 1])
        slices_rgba = [cmap(np.abs(_)) for _ in scaled_slices]

    log.info('Blending with a checkerboard background for display...')
    blended_slices = [overlay(_) for _ in slices_rgba]  # Use alpha channel to color in a background

    log.info('Saving...')
    for _, img in enumerate(slices_rgba):
        # image_output_path = output_path.parent / (output_path.stem + ('_phase' if show_phase else '') + f'_us{up_sampling_factor:0.0f}_sat{saturation:0.0f}_{_}.png')
        # log.info(f'Saving slice image with alpha values to {image_output_path}...')
        out_folder = r"C:\Users\lvalantinas\Repositories\lab\code\python\optics\optics\experimental\output/"
        Image.fromarray((img * 255.5).astype(np.uint8), mode='RGBA').save(out_folder + f"field{_}.png")

    log.info('Displaying...')
    fig, axs = plt.subplots(2, len(slices))
    for _ in range(len(slices)):
        axs[0, _].imshow(slices_rgba[_][..., :3])
        axs[0, _].set(ylabel='xyz'[_ <= 1], xlabel='xyz'[(_ <= 1) + 1])
        axs[1, _].imshow(blended_slices[_])
        axs[1, _].set(ylabel='xyz'[_ <= 1], xlabel='xyz'[(_ <= 1) + 1])

    plt.draw_all()

    log.info('Done!')
    plt.show()

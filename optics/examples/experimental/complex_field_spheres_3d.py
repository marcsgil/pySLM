from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

from optics.utils import ft
from optics.utils.ft.czt import interp
from optics.utils.display import complex2rgb, grid2extent


if __name__ == "__main__":
    from examples.experimental import log

    up_sampling_factor = 6
    saturation = 2

    input_path = r"G:\My Drive\data\3d_scattering" + r"\3d_scattering_833_E_conjugated.npy"
    fld = np.load(input_path, mmap_mode='r')
    fld = fld[0]  # Drop the polarization dimension
    log.info(f'field.shape = {fld.shape}, taking slices through the center...')
    grid = ft.Grid(fld.shape, step=633e-9 / 3)
    centre_px = grid.shape // 2
    left_side = fld[centre_px[0], :, :]  # displays 2nd x 3rd axis
    right_side = fld[:, :, centre_px[2]]
    slices = [left_side, right_side]
    # slices = [fld.swapaxes(0, _)[fld.shape[_]//2].astype(np.complex64) for _ in range(fld.ndim)]

    interpolated_slices = [interp(_, up_sampling_factor) for _ in slices]
    log.info(f'Normalizing {len(interpolated_slices)} slices...')
    normalized_slices = [_ / np.amax(np.abs(_)) for _ in interpolated_slices]
    scaled_slices = [(np.abs(_)**(1/3)) * np.exp(1j * np.angle(_)) for _ in normalized_slices]

    scaled_centre_px = (centre_px * up_sampling_factor).astype(np.int16)
    scaled_slices[0] = scaled_slices[0][:, :scaled_centre_px[1]]
    scaled_slices[1] = np.swapaxes(scaled_slices[1][:scaled_centre_px[0], :][::-1], axis1=1, axis2=0)
    slices_rgba = [complex2rgb(_, normalization=saturation, inverted=False) for _ in scaled_slices]

    # for _, img in enumerate(slices_rgba):
    #     out_folder = r"output/"
    #     Image.fromarray((img * 255.5).astype(np.uint8), mode='RGBA').save(out_folder + f"complex_field{_}.png")

    sphere_file = np.load(r"output/intersecting_spheres.npz")
    left_intersecting_spheres = (sphere_file["left_intersecting_spheres"])
    left_intersecting_spheres[:, :2] = left_intersecting_spheres[:, :2][:, ::-1]
    right_intersecting_spheres = sphere_file["right_intersecting_spheres"]
    combined_intersecting_spheres = [left_intersecting_spheres, right_intersecting_spheres]

    extent_0 = [-grid.extent[2] / 2, 0, -grid.extent[1] / 2, grid.extent[1] / 2]
    extent_1 = [0, -grid.extent[0] / 2, -grid.extent[1] / 2, grid.extent[1] / 2]

    def make_circles():
        circles = [[], []]
        for _ in range(2):
            for circle_coord in combined_intersecting_spheres[_]:
                circles[_].append(plt.Circle(circle_coord[:2], circle_coord[-1], alpha=1, facecolor="none", edgecolor="white"))
        return circles

    def make_rectangle():
        rectangles = np.empty(2, dtype=object)
        side = np.max(grid.extent) / 10
        for _ in range(2):
            location = tuple(-side / 2 for _ in range(2))
            rectangles[_] = plt.Rectangle(location, side, side, alpha=1, facecolor="none", edgecolor="white")
        return rectangles

    # fig, axs = plt.subplots(1, 2, dpi=100)
    # # TODO: add a copy of individual field generation code
    # axs[0].imshow(slices_rgba[0], extent=extent_0)
    # axs[1].imshow(slices_rgba[1], extent=extent_1)
    # circles = make_circles()
    # for _ in range(2):
    #     for circle in circles[_]:
    #         axs[_].add_patch(circle)
    # plt.show(block=True)

    circles = make_circles()
    rectangles = make_rectangle()
    extents = [extent_0, extent_1]
    for idx in range(2):
        fig_shape = np.array(slices_rgba[idx].shape[:2])[::-1] / np.max(slices_rgba[idx].shape[:2]) * 16
        fig, axs = plt.subplots(dpi=400, figsize=fig_shape)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        axs.imshow(slices_rgba[idx], extent=extents[idx])

        axs.add_patch(rectangles[idx])
        for circle in circles[idx]:
            axs.add_patch(circle)

        for side in ["top", "right", "bottom", "left"]:
            axs.spines[side].set_visible(False)
        axs.axes.xaxis.set_visible(False)
        axs.axes.yaxis.set_visible(False)
        fig.savefig(f"output/complex_ring_corner{idx}.png")
        # plt.show()


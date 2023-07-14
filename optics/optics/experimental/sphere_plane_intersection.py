import numpy as np
import matplotlib.pyplot as plt

from optics.utils.ft import Grid
from optics.utils.display import grid2extent


def intersection(sphere_coordinates, plane_extent):
    sphere_radius = np.max(sphere_coordinates[:, -1])
    plane_dimidx = np.argmin(np.sum(np.abs(plane_extent), axis=1))

    spheres_within_plane = sphere_coordinates.copy()
    included_extent = plane_extent + np.array([-sphere_radius, sphere_radius])[np.newaxis, :]
    for axis in range(3):
        spheres_within_plane = spheres_within_plane[
            np.logical_and((included_extent[axis, 0] < spheres_within_plane[:, axis]),
                           (spheres_within_plane[:, axis] < included_extent[axis, 1]))]
    spheres_within_plane[:, -1] = np.sqrt(spheres_within_plane[:, -1] ** 2 - spheres_within_plane[:, plane_dimidx] ** 2)
    intersection_position_radii = np.concatenate((spheres_within_plane[:, :plane_dimidx],
                                                  spheres_within_plane[:, plane_dimidx + 1:]), axis=1)
    return intersection_position_radii


if __name__ == "__main__":
    # Testing
    sphere_position_radii = np.loadtxt(r"C:\Users\lvalantinas\Repositories\lab\code\python\blender\3d_scattering_cube" +
                                       r"\modified_positions_radii.csv",
                                       delimiter=',', skiprows=1, dtype=np.float32)
    grid = Grid(np.full(3, 833), step=633e-9/3)
    left_plane_extent = [[0, 0],
                         [-grid.extent[1] / 2, grid.extent[1] / 2],
                         [-grid.extent[2] / 2, 0]]
    left_plane_extent = np.asarray(left_plane_extent)
    right_plane_extent = [[-grid.extent[0] / 2, 0],
                          [-grid.extent[1] / 2, grid.extent[1] / 2],
                          [0, 0]]
    right_plane_extent = np.asarray(right_plane_extent)

    left_intersecting_spheres = intersection(sphere_position_radii, left_plane_extent)
    right_intersecting_spheres = intersection(sphere_position_radii, right_plane_extent)

    np.savez("output/intersecting_spheres.npz", left_intersecting_spheres=left_intersecting_spheres,
             right_intersecting_spheres=right_intersecting_spheres)
    nb_spheres = right_intersecting_spheres.shape[0]
    circles = np.empty(nb_spheres, dtype=object)
    for idx in range(nb_spheres):
        circles[idx] = plt.Circle(right_intersecting_spheres[idx, :2] * 1e6, right_intersecting_spheres[idx, -1] * 1e6, alpha=0.5, facecolor="none", edgecolor="black")

    background = np.random.randn(833, 833)

    fig, axs = plt.subplots()
    axs.imshow(background, extent=grid2extent(grid) * 1e6)
    for circle in circles:
        axs.add_patch(circle)

    plt.show()

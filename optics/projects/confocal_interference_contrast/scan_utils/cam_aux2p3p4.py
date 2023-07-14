import numpy as np
from scipy import ndimage
from optics.utils.roi import Roi


def roi4cam_aux(img, rad: int = 50):
    # TODO: write description
    img_centre = np.round(ndimage.center_of_mass(img)).astype(int)

    img_left, img_right = img[:, :img_centre[1]], img[:, img_centre[1]:]

    left_centre = np.round(ndimage.center_of_mass(img_left)).astype(int)
    right_centre = np.round(ndimage.center_of_mass(img_right)).astype(int)

    shape = [int(np.maximum(left_centre[0], right_centre[0]) - np.minimum(left_centre[0], right_centre[0]) + rad * 2),
             int(right_centre[1] - left_centre[1] + img_centre[1] + rad * 2)]

    img_centre_adjusted = (left_centre + right_centre) / 2
    img_centre_adjusted[1] += img_centre[1] / 2
    img_centre_adjusted = np.round(img_centre_adjusted).astype(int)

    return Roi(center=img_centre_adjusted, shape=shape)


def cam_aux2p3p4(img, rad: int = 50):
    """
    :param img: cam_aux image, which contains info on P3 and P4 intensity values
    :param rad: crop radius around the spot centre (in pixels)
    :return: P3 and P4 intensity arrays
    """

    roi = roi4cam_aux(img, rad)
    roi_slc = [slice(None)] * 2
    roi_slc[0] = slice(*[np.round(roi.center[0] + roi.shape[0] / 2 * _).astype(int) for _ in [-1, 1]])
    roi_slc[1] = slice(*[np.round(roi.center[1] + roi.shape[1] / 2 * _).astype(int) for _ in [-1, 1]])

    cropped_img = img[tuple(roi_slc)]

    p3, p4 = cropped_img[:, :cropped_img.shape[1] // 2], cropped_img[:, cropped_img.shape[1] // 2:]
    return p3, p4


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = np.zeros((500, 500))
    img[109, 103] = 1
    img[100, 250] = 4

    left, right = cam_aux2p3p4(img)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(left)
    axs[1].imshow(right)
    plt.show(block=True)

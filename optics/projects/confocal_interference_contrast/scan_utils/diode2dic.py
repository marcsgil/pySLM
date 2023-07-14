import numpy as np


def get_theta(p1, p2, p3, p4):
    theta = np.arccos((p1 - p2) / (2 * np.sqrt(p3 * p4)))
    return theta


def diode2dic(p1, p2, p3, p4):
    """
    :param p1: transmis
    :return: complex DIC image inferred from the 4 interference elements
    """
    p1, p2, p3, p4 = [np.mean(p_detector, axis=(-1, -2)) for p_detector in [p1, p2, p3, p4]]

    theta = get_theta(p1, p2, p3, p4)
    return np.exp(1j * theta)


if __name__ == "__main__":
    diode2dic(0)

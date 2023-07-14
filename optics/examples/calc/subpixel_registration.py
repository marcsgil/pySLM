import numpy as np

from optics.utils.ft.subpixel import register, roll

from examples.calc import log


if __name__ == '__main__':
    subject = np.zeros(shape=(3, 4))
    subject[0, 0] = 1
    subject[1, 1] = 1
    ref_img = np.zeros_like(subject, dtype=complex)
    ref_img[0, 0] = 1 / (1.5 + 1j)
    ref_img[1, 1] = 1 / (1.5 + 1j)
    ref_img = roll(ref_img, shift=(-0.20, -0.30))  # incorrect rounding of first dimension depends on fraction in second dim.

    registration = register(subject, ref_img, precision=0)
    log.info(f"0: {registration}")
    registration = register(subject, ref_img, precision=1)
    log.info(f"1: {registration}")
    registration = register(subject, ref_img, precision=1 / 2)
    log.info(f"2:  {registration}")
    registration = register(subject, ref_img, precision=1 / np.pi)
    log.info(f"pi: {registration}")
    registration = register(subject, ref_img)
    log.info(f"1/128: {registration}")
    registration = register(subject, ref_img, precision=1 / 1000)
    log.info(f"1000: {registration}")
    
    # 1-dimensional
    ref_img = np.zeros(4)
    ref_img[1] = 1
    ref_img[2] = 1 / 2
    ref_img = roll(ref_img, shift=-0.24)
    subject = np.zeros_like(ref_img)
    subject[2] = 1 / 3
    subject[3] = 1 / 6

    registration = register(subject, ref_img, precision=0)
    log.info(f"0: {registration}")
    registration = register(subject, ref_img, precision=1)
    log.info(f"1: {registration}")
    registration = register(subject, ref_img, precision=1 / 2)
    log.info(f"2:  {registration}")
    # registered = register(img1, img2, precision=1/2.01)
    # log.info(f"2.01:  {registered}")
    registration = register(subject, ref_img, precision=1 / np.pi)
    log.info(f"pi: {registration}")
    registration = register(subject, ref_img, precision=1 / 1000)
    log.info(f"1000: {registration}")


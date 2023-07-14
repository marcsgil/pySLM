import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM  # used by svglib
import os
import PIL
import PIL.Image
import PIL.ImageChops
from typing import Union, Optional, Sequence
import logging

__all__ = ['usaf1951', 'spokes', 'logo_uod', 'cloud',
           'boat', 'peppers']

array_like = Union[np.ndarray, Sequence, int]
log = logging.getLogger(__name__)


def __image_file_path(file_name: str=None):
    image_file_path = os.path.dirname(os.path.abspath(__file__))
    if file_name is not None:
        image_file_path = os.path.join(image_file_path, file_name)
    if not os.path.exists(image_file_path):
        raise FileNotFoundError(f'Image file {file_name} not found in {image_file_path}.')
    return image_file_path


def __to_shape(pil_image: PIL.Image, new_shape, scale: float=1) -> PIL.Image:
    if new_shape is not None:
        new_shape = np.asarray(new_shape).flatten()
        if new_shape.size == 1:
            new_shape = np.repeat(new_shape, repeats=2, axis=0)
        input_shape = np.asarray(pil_image.size)[::-1]
        new_shape = [(length if length is not None else (input_length / scale).astype(int))
                     for length, input_length in zip(new_shape, input_shape)]
        if np.any(input_shape != new_shape):
            new_image = PIL.Image.new(pil_image.mode, new_shape[::-1])
            box = ((np.asarray(new_shape) - input_shape) / 2).astype(int)[::-1]
            new_image.paste(im=pil_image, box=tuple(box))
            pil_image = new_image
    return pil_image


def __from_svg(file_name: str, shape=None, scale: float = 1.0) -> PIL.Image:
    """
    Rasterizes a Scalable Vector Graphics (svg) file.

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.
    :param scale: (optional) A scaling factor for the vector drawing. E.g. 0.50 shrinks the drawing with respect to
        the canvas, showing the background for half the width and height.
    :return: An ndarray of shape (height, width).
    """
    drawing = svg2rlg(__image_file_path(file_name))
    if drawing is None:
        raise FileNotFoundError(f'Could not read image file {__image_file_path(file_name)}.')
    file_shape = np.asarray([drawing.height, drawing.width])
    shape = np.asarray(shape).ravel()
    if shape.size < 2:
        shape = np.repeat(shape, repeats=2, axis=0)
    elif shape.size > 2:
        shape = shape[-2:]
    render_scale = [(scale * length / file_length if length is not None else np.inf)
                    for length, file_length in zip(shape, file_shape)]
    render_scale = np.amin(render_scale)
    if np.isinf(render_scale):
        render_scale = scale
    drawing.scale(sy=render_scale, sx=render_scale)
    drawing.height *= render_scale
    drawing.width *= render_scale
    # todo: Use drawing.translate() instead of pasting image over one of the right shape
    # renderPM doesn't seem to handle alpha channels, so render it twice and deduce the alpha value:
    pil_image = renderPM.drawToPIL(drawing, dpi=72, bg=0x000000)
    pil_image_white = renderPM.drawToPIL(drawing, dpi=72, bg=0xffffff)
    if not np.allclose(np.asarray(pil_image_white).ravel(), np.asarray(pil_image).ravel()):
        alpha = PIL.ImageChops.difference(pil_image_white, pil_image).convert('L')
        alpha = PIL.ImageChops.invert(alpha)
        pil_image.putalpha(alpha)

    return __to_shape(pil_image, shape, scale)


def __from_bitmap(file_name: str, shape=None) -> PIL.Image:
    pil_image = PIL.Image.open(__image_file_path(file_name))

    return __to_shape(pil_image, shape)


def usaf1951(shape: Optional[array_like] = None, scale: float = 1) -> PIL.Image:
    """
    The United States Air Force 1951 resolution test chart. A white-on-black image with sets of three bars at various
    resolutions. This is adapted from https://en.wikipedia.org/wiki/1951_USAF_resolution_test_chart

    The largest lines at the bottom right are 2/75th of the total width, and their center-to-center distance as well.
    The lines at the top right are 1/75th of the total width, and so is their center-to-center distance. The aspect
    ratio of all lines is 10x1.
    By default, the image width and height are 750 pixels, so the line width of the widest line at the bottom right is
    20 pixels and those at the top right are 10 pixels wide.
    When a shape is specified with a different width and height, the smallest of the two is used for scaling.
    E.g. :code:`usaf1951(shape=(75, 100))` results in an image of shape :code:`(75, 100) == (height, width)` where the largest bars, at
    the bottom right, have a width of 2 pixels, and those at the top right one of 1 pixel. I.e. the scale is the same
    as that for :code:`usaf1951(shape=(75, 75)) == usaf1951(shape=75)`

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.

    :param scale: (optional) A scaling factor for the vector drawing. E.g. 0.50 shrinks the drawing with respect to
        the canvas, showing the background for half the width and height.

    :return: An ndarray of shape (height, width). By default, this is 750x750 pixels.

    """
    return __from_svg('usaf1951.svgz', shape=shape, scale=scale).convert(mode='L')  # Convert to monochrome


def spokes(shape=None, scale: float=1) -> PIL.Image:
    """
    The spokes target.
    https://en.wikipedia.org/wiki/Siemens_star

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.
    :param scale: (optional) A scaling factor for the vector drawing. E.g. 0.50 shrinks the drawing with respect to
    the canvas, showing the background for half the width and height.
    :return: An ndarray of shape (height, width).
    """
    # todo: Produce this using e.g. matplotlib, without going theough an svg file.
    return __from_svg('spokes.svgz', shape=shape, scale=scale).convert(mode='L')  # Convert to monochrome


def logo_uod(shape=None, scale: float=1) -> PIL.Image:
    """
    The logo of the University of Dundee.

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.
    :param scale: (optional) A scaling factor for the vector drawing. E.g. 0.50 shrinks the drawing with respect to
        the canvas, showing the background for half the width and height.
    :return: An ndarray of shape (height, width).
    """
    return __from_svg('logo_uod.svgz', shape=shape, scale=scale)


def cloud(shape=None, scale: float=1) -> PIL.Image:
    """
    The simple drawing of a blue cloud.

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.
    :param scale: (optional) A scaling factor for the vector drawing. E.g. 0.50 shrinks the drawing with respect to
        the canvas, showing the background for half the width and height.
    :return: An ndarray of shape (height, width).
    """
    return __from_svg('cloud.svg', shape=shape, scale=scale)


def boat(shape=None) -> PIL.Image:
    """
    The 'boat' reference grayscale image.

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.
    :return: An ndarray of shape (height, width).
    """
    return __from_bitmap('boat.png', shape=shape)


def peppers(shape=None) -> PIL.Image:
    """
    The 'peppers' reference color image.
    Source: http://sipi.usc.edu/database/database.php?volume=misc
    Alternative source: http://imageprocessingplace.com/root_files_V3/image_databases.htm

    :param shape: The desired output shape (height, width) in pixels. Default: None, the original resolution.
    :return: An ndarray of shape (height, width).
    """
    return __from_bitmap('peppers.png', shape=shape)


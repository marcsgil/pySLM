from __future__ import annotations
import numpy as np
import collections.abc as col
from typing import Union, Iterable, Sequence
import logging

from optics.utils.ft import Grid

log = logging.getLogger(__name__)

coord_type = Union[int, float]
array_type = Union[np.ndarray, Sequence, float, int]


class Roi(col.Sequence):
    """
    A class that represents a rectangular area with its sides aligned with the Cartesian axes.
    Objects of this class can be used to represent regions of interest.
    """
    def __init__(self, top=None, left=None, height=None, width=None, top_left=None, shape=None,
                 bottom_right=None, bottom=None, right=None, center=None, dtype=None):
        """
        Construct a region-of-interest object.

        :param top: The top position. Default: 0.
        :param left: The left position. Default: 0.
        :param height: The height of the region-of-interest. Default: 0.
        :param width: The width of the region-of-interest. Default: 0.
        :param top_left: A sequence of two elements: (top, left).
        :param shape: A sequence of two elements: (height, width).
        :param bottom_right: A sequence of two elements: (bottom, right).
        :param bottom: The bottom of the region of interest.
        :param right: The right-hand-side of the region of interest.
        :param center: The center as a sequence of two elements: (vertical, horizontal).
        :param dtype: The `dtype` of this region of interest. Default int, unless some of the previous arguments is a float.
        """
        self.__top_left = np.zeros(2, dtype=int)
        self.__shape = np.zeros(2, dtype=int)  # The dtype of shape will determine the self.dtype property
        self(top=top, left=left, height=height, width=width, top_left=top_left, shape=shape, bottom_right=bottom_right,
             bottom=bottom, right=right, center=center, dtype=dtype)

    def __call__(self, top=None, left=None, height=None, width=None, top_left=None, shape=None,
                 bottom_right=None, bottom=None, right=None, center=None, dtype=None) -> Roi:
        """
        Update the current Roi object's values.

        :param top: The top position. Default: 0.
        :param left: The left position. Default: 0.
        :param height: The height of the region-of-interest. Default: 0.
        :param width: The width of the region-of-interest. Default: 0.
        :param top_left: A sequence of two elements: (top, left).
        :param shape: A sequence of two elements: (height, width).
        :param bottom_right: A sequence of two elements: (bottom, right).
        :param bottom: The bottom of the region of interest.
        :param right: The right-hand-side of the region of interest.
        :param center: The center as a sequence of two elements: (vertical, horizontal).
        :param dtype: The dtype of this region of interest. Default int, unless some of the previous arguments is a float.
        :return: A reference to this Roi.
        """
        if isinstance(top, Roi):
            top_left = top.top_left
            shape = top.shape
            top = None
        elif isinstance(top, Iterable):
            seq = top
            if len(top) == 4:
                top, left, height, width = seq
            elif len(top) == 2:
                top_left = seq
                if isinstance(left, col.Sequence) and len(left) == 2:
                    shape = left
                    left = None
                else:
                    log.error(f"If the first argument, top, is a 2-vector, so must be the second, not a {len(left)}-vector")
            else:
                log.error(f"The first argument, top, must be a multidimensional, 2-vector, or a 4-vector, not a {len(seq)}-vector")
            del seq
        if height is None:
            if top is not None and bottom is not None:
                height = bottom - top
            else:
                height = self.height
        if width is None:
            if left is not None and right is not None:
                width = right - left
            else:
                width = self.width
        if shape is None:
            shape = (height, width)
            if top_left is not None:
                top_left = np.asarray(top_left).flatten()
                if bottom_right is not None:
                    shape = np.asarray(bottom_right).ravel() - top_left
                elif center is not None:
                    shape = (np.asarray(center).ravel() - top_left) * 2
            elif bottom_right is not None and center is not None:
                shape = (np.asarray(bottom_right).ravel() - np.asarray(center).ravel()) * 2

        # At this point the shape should be known
        shape = np.asarray(shape).ravel()  # Make sure it is a numpy vector

        if top_left is None:
            if bottom_right is not None:
                top_left = np.asarray(bottom_right).ravel() - shape
            elif center is not None:
                center = np.asarray(center).ravel()
                if dtype is None:
                    dtype = (shape[0] + center[0]).dtype
                half_shape = (shape / 2).astype(dtype)
                top_left = center.astype(dtype) - half_shape
            else:
                if top is None:
                    if bottom is None:
                        top = self.top
                    else:
                        top = shape[0] - bottom
                if left is None:
                    if right is None:
                        left = self.left
                    else:
                        left = shape[1] - right
                top_left = (top, left)

        top_left = np.array(top_left).flatten()

        if dtype is None:
            dtype = (top_left[0] + shape[0]).dtype

        self.__top_left = np.array(top_left, dtype=dtype)
        self.__shape = np.array(shape, dtype=dtype)
        # The dtype of shape determines the self.dtype property

        return self

    def __len__(self) -> int:
        """
        The length of the equivalent tuple(top, left, height, width)
        :return: 4
        """
        return 4

    def __getitem__(self, idx: Union[int, col.Sequence[int]]) -> Union[coord_type, col.Sequence[coord_type]]:
        """
        Returns an element of the equivalent tuple
        :param idx: the index in the tuple
        :return: Element idx in (top, left, height, width)
        """
        return (self.top, self.left, self.height, self.width)[idx]

    def __setitem__(self, idx: Union[int, col.Sequence[int]],
                    new_value: Union[coord_type, col.Sequence[coord_type]]):
        """
        Sets an element of the equivalent tuple
        :param idx: the index in the tuple
        :return: Element idx in (top, left, height, width)
        """
        # Convert to sequence of setter methods
        setters = (self.__class__.top.setter, self.__class__.left.setter,
                   self.__class__.height.setter, self.__class__.width.setter)[idx]
        # Call setter methods as required
        for setter, val in zip(setters, new_value):
            setter(val)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(top_left={self.top_left}, shape={self.shape}, dtype={self.dtype})"

    @property
    def __dict__(self):
        return {'top_left': self.top_left.tolist(), 'shape': self.shape.tolist()}

    @__dict__.setter
    def __dict__(self, roi_dict: dict):
        self.top_left = roi_dict['top_left']
        self.shape = roi_dict['shape']
        if 'dtype' in roi_dict:
            self.dtype = roi_dict['dtype']

    @property
    def top_left(self) -> np.ndarray:
        """The top-left corner of the region-of-interest. The coordinates are listed in that order."""
        return self.__top_left

    @top_left.setter
    def top_left(self, top_left=None, top=None, left=None):
        """Moves the region-of-interest so that the top-left edge is as indicated, without changing its shape."""
        if top_left is None:
            top_left = np.array((top, left), dtype=self.dtype)

        self.__top_left = np.array(top_left, dtype=self.dtype).flatten()

    @property
    def shape(self) -> np.ndarray:
        """The shape [height, width] of the region-of-interest."""
        if not hasattr(self, f'_{self.__class__.__name__}__shape'):  # This is required for the json-deserialization
            self.__shape = np.zeros(2, dtype=int)
        return self.__shape

    @shape.setter
    def shape(self, shape=None, height=None, width=None):
        """Set the shape [height, width] of the region-of-interest."""
        if shape is None:
            shape = np.array((height, width), dtype=self.dtype)

        self.__shape = np.array(shape, dtype=self.dtype).flatten()

    @property
    def size(self) -> int:
        """The number of pixels in this region of interest."""
        return int(np.prod(self.shape))

    @property
    def top(self) -> coord_type:
        """The top coordinate of the region of interest."""
        return self.__top_left[0]

    @top.setter
    def top(self, top: coord_type):
        """Moves the region-of-interest so that the top edge is as indicated, without changing its shape."""
        self.__top_left[0] = np.array(top, dtype=self.dtype)

    @property
    def left(self) -> coord_type:
        """The left-side of the region-of-interest."""
        return self.__top_left[1]

    @left.setter
    def left(self, left: coord_type):
        """Moves the region-of-interest so that the left edge is as indicated, without changing its shape."""
        self.__top_left[1] = np.array(left, dtype=self.dtype)
        
    @property
    def bottom_right(self) -> np.ndarray:
        """The bottom-right corner of the region-of-interest."""
        return self.__top_left + self.__shape
    
    @bottom_right.setter
    def bottom_right(self, new_bottom_right: array_type):
        """
        Moves the region-of-interest so that the bottom-right edge is as indicated, without changing its shape.
        :param new_bottom_right: a 2-tuple or 2-vector indicating the bottom and right position, respectively.
        """
        self.bottom = new_bottom_right[0]
        self.right = new_bottom_right[1]

    @property
    def bottom(self) -> coord_type:
        """Moves the region of interest so that the bottom side is as specified."""
        return self.__top_left[0] + self.shape[0]

    @bottom.setter
    def bottom(self, bottom: coord_type):
        """Moves the region-of-interest so that the bottom edge is as indicated, without changing its shape."""
        self.__top_left[0] = np.array(bottom, dtype=self.dtype) - self.shape[0]

    @property
    def right(self) -> coord_type:
        """Moves the region of interest so that the right side is as specified."""
        return self.__top_left[1] + self.shape[1]

    @right.setter
    def right(self, right: coord_type):
        """Moves the region-of-interest so that the right edge is as indicated, without changing its shape."""
        self.__top_left[1] = np.array(right, dtype=self.dtype) - self.shape[1]

    @property
    def height(self) -> coord_type:
        """The height of the region of interest."""
        return self.__shape[0]

    @height.setter
    def height(self, height: coord_type):
        """Set the height of the region of interest."""
        self.__shape[0] = height

    @property
    def width(self) -> coord_type:
        """The width of the region of interest."""
        return self.__shape[1]

    @width.setter
    def width(self, width: coord_type):
        """Set the width of the region of interest."""
        self.__shape[1] = width

    @property
    def center(self) -> np.ndarray:
        """The center position of the region-of-interest."""
        return np.array(self.__top_left + self.shape / 2.0, dtype=self.dtype)

    @center.setter
    def center(self, pos: array_type):
        """Moves the region-of-interest so that the center is as indicated, without changing its shape."""
        self.__top_left = np.array(pos - self.shape / 2.0, dtype=self.dtype).flatten()

    @property
    def dtype(self):
        """The dtype of the coodinates, either an int or float"""
        return self.shape.dtype

    @dtype.setter
    def dtype(self, new_dtype):
        """The dtype of the coodinates, either an int or float"""
        self.__shape = np.array(self.shape, dtype=new_dtype)
        self.__top_left = np.array(self.top_left, dtype=new_dtype)

    def __mod__(self, clipping_roi: Roi) -> Roi:
        """
        Returns a new rectangle that is clipped by the other rectangle.
        This object is not altered.

        Usage: clipped_roi = roi % clipping_roi

        :param clipping_roi: The clipping rectangle.
        :return: The newly clipped rectangle, never larger than roi.
        """
        # clip top left
        new_top_left = np.maximum(self.top_left, clipping_roi.top_left)
        new_shape = self.bottom_right - new_top_left
        # clip bottom right
        new_shape = np.minimum(new_top_left + new_shape, clipping_roi.bottom_right) - new_top_left

        return Roi(top_left=new_top_left, shape=new_shape)

    def __mul__(self, reference: Roi) -> Roi:
        """
        Returns a new Roi of which the right-hand rectangle is the relative with respect to this object.
        The original object is not altered.

        ::

            #######################################    #######################################
            #                                     #    #                                     #
            #                                     #    #                                     #
            #    ###########################      #    #                                     #
            #    # reference               #      #    #                                     #
            #    #                         #      #    #                                     #
            #    #    ##################   #      #    #         ##################          #
            #    #    #     self       #   #      # => #         # self*reference #          #
            #    #    ##################   #      #    #         ##################          #
            #    #                         #      #    #                                     #
            #    ###########################      #    #                                     #
            #                                     #    #                                     #
            #######################################    #######################################

        Usage: absolute_roi = roi * relative_roi

        Note that this multiplication is non-commutative! The shape is always that of the current object.

        :param reference: The region-of-interest to which the current Roi is referenced.
        :return: The absolute region-of-interest.
        """

        return Roi(top_left=self.top_left + reference.top_left, shape=self.shape)

    def __neg__(self) -> Roi:
        """Inverts the top-left and bottom-right corners"""
        return Roi(bottom_right=self.top_left, shape=-self.shape)

    def __truediv__(self, reference: Roi):
        """
        Returns a new Roi that is the relative of the first with respect to the second in the fraction.
        This object is not altered.

        ::

            #######################################
            #                                     #
            #                                     #
            #    ###########################      #    ###########################
            #    # reference               #      #    #                         #
            #    #                         #      #    #                         #
            #    #    ##################   #      #    #    ##################   #
            #    #    #     self       #   #      # => #    # self/reference #   #
            #    #    ##################   #      #    #    ##################   #
            #    #                         #      #    #                         #
            #    ###########################      #    ###########################
            #                                     #
            #######################################

        Usage: relative_roi = roi / absolute_roi

        :param reference: The second rectangle in the fraction.
        :return: The relative rectangle.
        """
        # if np.any(self.top_left < absolute.top_left) or np.any(self.bottom_right > absolute.bottom_right):
        #     log.debug('First rectangle in fraction does not fit within the absolute rectangle!')

        return Roi(top_left=self.top_left - reference.top_left, shape=self.shape)

    def __add__(self, other: Roi) -> Roi:
        """
        Returns a new rectangle that overlaps both rectangles in the sum.
        This object is not altered.

        :param other: The other rectangle.
        :return: The overlapping rectangle.
        """
        # extend top left
        new_top_left = np.minimum(self.top_left, other.top_left)
        new_shape = self.bottom_right - new_top_left
        # extend bottom right
        new_shape = np.maximum(new_top_left + new_shape, other.bottom_right) - new_top_left

        return Roi(top_left=new_top_left, shape=new_shape)

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(top_left={self.top_left!r}, shape={self.shape!r}, dtype={self.dtype!r})")

    def __eq__(self, other: Roi) -> bool:
        return np.all(self.top_left == other.top_left) and np.all(self.shape == other.shape)

    def astype(self, new_dtype) -> Roi:
        return Roi(top_left=np.array(self.top_left, dtype=new_dtype), shape=np.array(self.shape, dtype=new_dtype))

    @property
    def grid(self) -> Grid:
        """
        :return: The vertical and horizontal index ranges, respectively. All indexes start at 0.
        """
        return Grid(shape=self.shape, first=self.top_left)

    @property
    def convex(self) -> Roi:
        """
        A new Rect that has a non-negative shape. This object is not altered.
        I.e. left is on the left side of right, and top is on the top of bottom.
        """
        if self.shape[0] > 0:
            bottom, top = self.bottom, self.top
        else:
            top, bottom = self.bottom, self.top
        if self.shape[1] > 0:
            right, left = self.right, self.left
        else:
            left, right = self.right, self.left

        return Roi(top=top, bottom=bottom, left=left, right=right)


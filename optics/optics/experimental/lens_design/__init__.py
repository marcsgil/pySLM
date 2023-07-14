import numpy as np
from typing import Union, Sequence
from numbers import Complex
import logging

log = logging.getLogger(__name__)

# array_type = np.ndarray
# array_like = Union[array_type, Sequence, Complex]
# Param = array_type

import torch

array_type = torch.Tensor
array_like = Union[array_type, Sequence, Complex]
Param = torch.nn.Parameter
einsum = torch.einsum
norm = torch.norm

dtype_r = torch.float32
dtype_c = torch.complex64


def asarray_r(arr: array_like, dtype=dtype_r) -> array_type:
    if isinstance(arr, array_type):
        if torch.is_complex(arr):
            arr = arr.real
        if arr.dtype != dtype:
            arr = arr.to(dtype=dtype)
    else:
        arr = np.asarray(arr).real
        return array_type(arr).to(dtype=dtype)

    if not arr.requires_grad:
        arr.requires_grad = True
    return arr


def asarray_c(arr: array_like, dtype=dtype_c) -> array_type:
    return asarray_r(arr, dtype)


def asnumpy(arr: array_type) -> np.ndarray:
    if isinstance(arr, array_type):
        return arr.detach().numpy()
    else:
        return np.asarray(arr)


def copy_array(arr: array_type) -> array_type:
    return arr.detach().clone()


array_type.copy = copy_array


# class Param(np.ndarray):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.__grad = np.zeros_like(self)
#
#     @property
#     def grad(self) -> np.ndarray:
#         return self.__grad
#
#     @grad.setter
#     def grad(self, new_grad):

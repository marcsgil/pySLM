"""
This submodule provides classes for abstract vector manipulation. It consists of ``Vector``s, ``BasisTensor``s,
and ``Space``s.

* ``Vector``s are superpositions of ``BasisTensor``s

* ``BasisTensor``s can consist of multiple ``Dimension``s. These are singleton ``BasisTensor``s.

* ``Space``s are an indexible collection of ``Dimension``s.

Usage:

::

    x, y, z = Space(3)

    s = Space()
    x, y, z = s[:3]

    s = Space(range(3))
    x, y, z = s

    s = Space('xyz')
    x, y, z = s[:3]

    s = Space(name='s')
    x, y, z = s[:3]

    s = Space('xyz', name='s')
    x, y, z = s[:3]

    v = 3 * x - 2 * y + z
    print(v)
    print(v.dual)

    v = 5 * x * y * z
    print(v)
    print(v.dual)


TODO: Implement inner products to reduce BasisTensors.
"""
from __future__ import annotations

from abc import ABC
from typing import Union, Sequence, TypeVar, Generic, Optional
from copy import copy, deepcopy
from optics.utils.display import subsup

__all__ = ['Space']

ElementType = TypeVar('ElementType')
IndexType = TypeVar('IndexType')


ScalarType = Union[int, float, complex]


class Space(Generic[IndexType]):
    """Represents a space that can have multiple dimensions."""
    def __init__(self, indices: Union[Sequence[IndexType], int, None] = None, name: Optional[str] = None):
        if isinstance(indices, int):
            indices = range(indices)
        self.__indices = indices
        self.__name = name
        self.__dual = None

    @property
    def indices(self) -> Optional[Sequence[IndexType]]:
        return self.__indices

    @property
    def name(self) -> str:
        return self.__name

    @property
    def dual(self) -> Space:
        if self.__dual is None:
            self.__dual = DualSpace(self)
        return self.__dual

    def __call_(self, index: IndexType) -> Dimension[IndexType]:
        """Returns a basis vector in a given dimension."""
        assert index in self.indices, f"Index {index} is not in {self.indices}."
        return Dimension[IndexType](self, index)

    def __getitem__(self, item: Union[int, slice]) -> Dimension[IndexType]:
        """Returns a basis vector in a given dimension."""
        if isinstance(item, int):
            index = self.indices[item] if self.indices is not None else item
            return Dimension[IndexType](self, index)
        else:
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else start + 1
            return [self[_] for _ in range(start, stop, item.step if item.step is not None else 1)]

    def __invert__(self) -> Space:
        """The reciprocal of this object using the ~ unary operator."""
        return self.dual

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        arg_str = ", ".join(f"{k.removeprefix(f'_{self.__class__.__name__}__')}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({arg_str})"


class DualSpace(Space):
    def __init__(self, space: Space):
        super().__init__(indices=space.indices, name=space.name)
        self.__space: Space = space

    @property
    def dual(self) -> Space:
        return self.__space


class Arithmetic(ABC, Generic[ElementType]):
    """
    Defines *, /, +, -, ~, as well as in-place operations.

    Subclasses should implement __iadd__ or __add__, and __imul__ or __mul__.
    """
    def __iadd__(self, other: ElementType) -> ElementType:
        raise NotImplementedError

    def __imul__(self, other: ElementType) -> ElementType:
        raise NotImplementedError

    def __add__(self, other: ElementType) -> ElementType:
        result = copy(self)
        result += other
        return result

    def __mul__(self, other: ElementType) -> ElementType:
        result = copy(self)
        result *= other
        return result

    def dual(self) -> Space:
        raise NotImplementedError

    def __invert__(self) -> ElementType:
        """The reciprocal of this object using the ~ unary operator."""
        return self.dual

    def __copy__(self):
        """Called when using copy.copy()."""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Called when using copy.deepcopy()."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __radd__(self, other: ElementType) -> ElementType:
        summation = copy(other)
        summation += self
        return summation

    def __rmul__(self, other: ElementType) -> ElementType:
        if isinstance(other, ScalarType):
            return self * other
        else:
            return other * self

    def __neg__(self) -> ElementType:
        """The additive inverse of this."""
        return self * (-1)

    def __isub__(self, other: ElementType) -> ElementType:
        self.__iadd__(-other)
        return self

    def __sub__(self, other: ElementType) -> ElementType:
        return self + (-other)

    def __rsub__(self, other: ElementType) -> ElementType:
        return other - self

    def __itruediv__(self, other: Union[ElementType, ScalarType]) -> ElementType:
        if isinstance(other, type(self)):  # Must be ElementType
            other_inv = ~other
        elif isinstance(other, ScalarType):
            other_inv = 1 / other
        else:
            raise TypeError(f"Division of {self} by {other} of unknown type {type(other)}.")
        self.__imul__(other_inv)
        return self

    def __truediv__(self, other: Union[ElementType, ScalarType]) -> ElementType:
        if isinstance(other, type(self)):  # Must be ElementType
            other_inv = ~other
        elif isinstance(other, ScalarType):
            other_inv = 1 / other
        else:
            raise TypeError(f"Division of {self} by {other} of unknown type {type(other)}.")
        return self * other_inv

    def __rtruediv__(self, other: Union[ElementType, ScalarType]) -> ElementType:
        return (~self) / other

    def __idiv__(self, other: Union[ElementType, ScalarType]) -> ElementType:
        return self.__itruediv__(other)

    def __div__(self, other: Union[ElementType, ScalarType]) -> ElementType:
        return self / other

    def __rdiv__(self, other: Union[ElementType, ScalarType]) -> ElementType:
        return other / self

    def __repr__(self) -> str:
        arg_str = ", ".join(f"{k.removeprefix(f'_{self.__class__.__name__}__')}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({arg_str})"

    def __eq__(self, other: ElementType) -> bool:
        return hash(self) == hash(other)


class BasisTensor(Arithmetic):
    """Represents a basis vector or tensor that can be used in a linear combination."""
    def __init__(self, spaces: Sequence[Space] = tuple(), indices: Sequence[IndexType] = tuple()):
        self.__spaces: Sequence[Space] = spaces
        self.__indices: Sequence[IndexType] = indices

    @property
    def spaces(self) -> Sequence[Space]:
        return self.__spaces

    @property
    def indices(self) -> Sequence[IndexType]:
        return self.__indices

    @property
    def factors(self) -> Sequence[Dimension]:
        return [Dimension(s, _) for s, _ in zip(self.spaces, self.indices)]

    def __iadd__(self, other: Union[BasisTensor, Vector, ScalarType]) -> BasisTensor:
        raise ValueError(f"Tried to do {self} += {other}, which would change the type from a BasisTensor to a Vector.")

    def __add__(self, other: Union[BasisTensor, Vector, ScalarType]) -> Vector:
        if isinstance(other, BasisTensor):
            return Term(self) + Term(other)
        elif isinstance(other, Vector):
            return Term(self) + other
        elif isinstance(other, ScalarType):
            return Term(self) + Scalar(other)
        else:
            raise TypeError(f"Tried to add {self} and {other}.")

    def __imul__(self, other: BasisTensor) -> BasisTensor:
        if not isinstance(other, BasisTensor):
            raise ValueError(f"Tried to do {self} *= {other} with type {type(other)}. In place multiplication requires a BasisTensor.")
        self.__spaces += other.spaces
        self.__indices += other.indices

        # spaces = []
        # indices = []
        # for _, (s, ind) in enumerate(zip(self.__spaces, self.__indices)):
        #     if s.dual not in self.__spaces[:_]:
        #         spaces.append(s)
        #         indices.append(ind)
        #     else:
        #         dual_idx = spaces.index(s.dual, 0, _)
        #         spaces.pop(dual_idx)
        #         indices.pop(dual_idx)
        #
        # self.__spaces = spaces
        # self.__indices = indices

        return self

    def __mul__(self, other: Union[BasisTensor, Vector, ScalarType]) -> Union[BasisTensor, Vector]:
        if isinstance(other, ScalarType):
            return Term(basis_tensor=self, scalar=other)
        elif isinstance(other, Vector):
            return Term(self) * other
        else:
            assert isinstance(other, BasisTensor), f"Tried to multiply {self} with {other} of unknown type {type(other)}."
            return BasisTensor(spaces=[*self.spaces, *other.spaces], indices=[*self.indices, *other.indices])

    @property
    def dual(self) -> BasisTensor:
        spaces_inv = [~_ for _ in self.spaces]
        return BasisTensor(spaces=spaces_inv[::-1], indices=self.indices[::-1])

    def __str__(self) -> str:
        return ''.join(str(_) for _ in self.factors)

    def __hash__(self) -> int:
        return hash((tuple(self.spaces), tuple(self.indices)))


class Dimension(BasisTensor, Generic[IndexType]):
    """A class to represent a single dimension with a given index in a certain space."""

    def __init__(self, space: Space, index: IndexType):
        super().__init__([space], [index])

    @property
    def space(self) -> Space:
        return self.spaces[0]

    @property
    def index(self) -> IndexType:
        return self.indices[0]

    def __mul__(self, other: Union[BasisTensor, Vector, ScalarType]) -> Union[BasisTensor, Vector]:
        if isinstance(other, ScalarType):
            return Term(basis_tensor=self, scalar=other)
        elif isinstance(other, Vector):
            return Vector(basis_tensors=[self, *other.basis_tensors], scalars=[1, *other.scalars])
        elif isinstance(other, BasisTensor):
            return BasisTensor(spaces=[self.space, *other.spaces], indices=[self.index, *other.indices])
        else:
            raise TypeError(f"Tried to multiply {self} with {other} of unknown type {type(other)}.")

    def __str__(self) -> str:
        if not isinstance(self.space, DualSpace):
            if self.space.name is not None:
                return f"|{self.space.name}{subsup.superscript(self.index)}〉"
            else:
                return f"|{self.index}〉"
        else:
            if self.space.name is not None:
                return f"〈{self.space.name}{subsup.subscript(self.index)}|"
            else:
                return f"〈{self.index}|"


class Vector(Arithmetic):
    """A class to represent a generic vector, any potential super-position."""
    def __init__(self, basis_tensors: Sequence[BasisTensor] = tuple(), scalars: Sequence[ScalarType] = tuple()):
        # Combine duplicates
        self.__basis_tensors = []
        self.__scalars = []
        for t, s in zip(basis_tensors, scalars):
            if t not in self.__basis_tensors:
                self.__scalars.append(s)
                self.__basis_tensors.append(t)
            else:
                self.__scalars[self.__basis_tensors.index(t)] += s

        # Drop any terms that are 0
        self.__basis_tensors = [t for t, s in zip(self.__basis_tensors, self.__scalars) if s != 0]
        self.__scalars = [s for s in self.__scalars if s != 0]

    @property
    def scalars(self) -> Sequence[ScalarType]:
        return self.__scalars

    @property
    def basis_tensors(self) -> Sequence[BasisTensor]:
        return self.__basis_tensors

    @property
    def terms(self) -> Sequence[Term]:
        return [Term(c, scalar=s) for c, s in zip(self.basis_tensors, self.scalars)]

    def __iadd__(self, other: Union[Vector, BasisTensor, ScalarType]) -> Vector:
        if isinstance(other, BasisTensor):
            other = Term(other)
        elif isinstance(other, ScalarType):
            other = Scalar(other)
        for ot, os in zip(other.basis_tensors, other.scalars):
            for _, t in enumerate(self.basis_tensors):
                if ot == t:
                    self.__scalars[_] += os
                    break
            else:
                self.__basis_tensors.append(ot)
                self.__scalars.append(os)
        # Drop any terms that became 0
        self.__basis_tensors = [t for t, s in zip(self.__basis_tensors, self.__scalars) if s != 0]
        self.__scalars = [s for s in self.__scalars if s != 0]
        return self

    def __imul__(self, other: Union[Vector, BasisTensor, ScalarType]) -> Vector:
        if isinstance(other, BasisTensor):
            other = Term(other)
        elif isinstance(other, ScalarType):
            other = Scalar(other)
        product_basis_tensors = []
        product_scalars = []
        for c, s in zip(self.basis_tensors, self.scalars):
            for oc, os in zip(other.basis_tensors, other.scalars):
                product_basis_tensors.append(c * oc)
                product_scalars.append(s * os)
        self.__basis_tensors = product_basis_tensors
        self.__scalars = product_scalars
        return self

    @property
    def dual(self) -> Vector:
        return Vector(basis_tensors=[~_ for _ in self.__basis_tensors], scalars=[1 / _ for _ in self.__scalars])

    def __str__(self) -> str:
        def format_term(s: ScalarType, t: BasisTensor) -> str:
            if s.imag == 0 and s == int(s):
                s = int(s)
            if len(t.factors) > 0:
                if s == 1:
                    return f"{t}"
                elif s == -1:
                    return f"-{t}"
                else:
                    return f"{s}{t}"
            else:
                return f"{s}"
        desc = '+'.join(f"{format_term(_, t)}" for _, t in zip(self.scalars, self.basis_tensors))
        desc = desc.removeprefix('+').replace('+-', '-')
        return desc


class Term(Vector):
    """A class that represents a vector with a single, potentially scaled, component."""
    def __init__(self, basis_tensor: BasisTensor, scalar: ScalarType = 1.0):
        super().__init__(basis_tensors=[basis_tensor], scalars=[scalar])

    @property
    def scalar(self) -> ScalarType:
        return self.scalars[0]

    @property
    def basis_tensor(self) -> BasisTensor:
        return self.basis_tensors[0]

    def __iadd__(self, other: Union[Vector, BasisTensor, ScalarType]) -> Vector:
        # raise NotImplementedError
        return self + other

    def __imul__(self, other: Union[Vector, BasisTensor, ScalarType]) -> Vector:
        # raise NotImplementedError
        return self * other

    def __add__(self, other: Union[Vector, BasisTensor, ScalarType]) -> Vector:
        if isinstance(other, BasisTensor):
            other = Term(other)
        elif isinstance(other, ScalarType):
            other = Scalar(other)
        return Vector(basis_tensors=[self.basis_tensor, *other.basis_tensors], scalars=[self.scalar, *other.scalars])

    def __mul__(self, other: Union[Vector, ScalarType]) -> Vector:
        if isinstance(other, BasisTensor):
            other = Term(other)
        elif isinstance(other, ScalarType):
            if other != 0:
                other = Scalar(other)
            else:
                other = Vector()
        product_tensors = []
        product_scalars = []
        for oc, os in zip(other.basis_tensors, other.scalars):
            product_tensors.append(self.basis_tensor * oc)
            product_scalars.append(self.scalar * os)
        return Vector(basis_tensors=product_tensors, scalars=product_scalars)


class Scalar(Vector):
    """A class that represents scalar terms, without a vector component."""
    def __init__(self, scalar: ScalarType = 1.0):
        super().__init__(basis_tensors=[BasisTensor()], scalars=[scalar])


if __name__ == '__main__':
    from optics.experimental import log

    x, y, z = Space('xyz')

    s1 = Space()
    x, y, z = s1[:3]

    v = 1 * x - 2 * x + 3 * x + 4 * y + 2j * x
    v = 3 * x / 4 + y * 1j - 0.25 * x

    v = x * y * z

    # v = x.dual * x  #todo: should be 1 or -1, depending on the Space

    log.info(f"{v}: {repr(v)}")

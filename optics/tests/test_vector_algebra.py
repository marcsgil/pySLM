import unittest
import numpy.testing as npt

from optics.utils.linalg.vector_algebra import Space
from optics.utils.display import subsup

import numpy as np


class TestBeamSection(unittest.TestCase):
    def setUp(self):
        self.spaces = [Space(), Space(3), Space('xyz'), Space(name='s'), Space(3, 's')]
        self.indices = [None, range(3), 'xyz', None, range(3)]
        self.names = [None, None, None, 's', 's']

    def test_space(self):
        for s, ind, n in zip(self.spaces, self.indices, self.names):
            npt.assert_array_equal(s.indices, ind)
            npt.assert_array_equal(s.dual.indices, ind)
            npt.assert_equal(s.name, n)
            npt.assert_equal(s.dual.name, n)

    def test_basis(self):
        for s, ind, n in zip(self.spaces, self.indices, self.names):
            x, y, z = s[:3]
            dims = [x, y, z]

            if ind is None:
                ind = range(len(dims))
            for _ in range(len(dims)):
                npt.assert_equal(dims[_].index, ind[_])
                npt.assert_equal(s[_].index, ind[_])
                npt.assert_equal(s[_].space, s)
                npt.assert_equal(s.dual[_].space, s.dual)

            if s.name is None:
                npt.assert_equal(str(x), f'|{x.index}〉')
                npt.assert_equal(str(y), f'|{y.index}〉')
                npt.assert_equal(str(z), f'|{z.index}〉')
            else:
                npt.assert_equal(str(x), f'|{s.name}{subsup.superscript(0)}〉')
                npt.assert_equal(str(y), f'|{s.name}{subsup.superscript(1)}〉')
                npt.assert_equal(str(z), f'|{s.name}{subsup.superscript(2)}〉')
                npt.assert_equal(str(x.dual), f'〈{s.name}{subsup.subscript(0)}|')
                npt.assert_equal(str(y.dual), f'〈{s.name}{subsup.subscript(1)}|')
                npt.assert_equal(str(z.dual), f'〈{s.name}{subsup.subscript(2)}|')

    def test_scalar_mul(self):
        for s in self.spaces:
            x, y, z = s[:3]

            p = 2 * x
            npt.assert_equal(str(p), f"2{x}")
            p = -2 * x
            npt.assert_equal(str(p), f"-2{x}")
            p = 2.0 * x
            npt.assert_equal(str(p), f"2{x}")
            p = 2.0 * x * 2
            npt.assert_equal(str(p), f"4{x}")
            p = 2.5j * x
            npt.assert_equal(str(p), f"2.5j{x}")
            p = (1.5 + 2.5j) * x
            npt.assert_equal(str(p), f"(1.5+2.5j){x}")

            p = 1 * x
            npt.assert_equal(str(p), f"{x}")
            p = x * 1
            npt.assert_equal(str(p), f"{x}")
            p = 0 * x
            npt.assert_equal(str(p), '')
            p = x * 0
            npt.assert_equal(str(p), '')

    def test_tensor_mul(self):
        for s in self.spaces:
            x, y, z = s[:3]

            p = x * y
            npt.assert_equal(str(p), f"{x}{y}")
            p = x * y * z
            npt.assert_equal(str(p), f"{x}{y}{z}")
            p = x * (y * z)
            npt.assert_equal(str(p), f"{x}{y}{z}")

            p = 2 * x * y
            npt.assert_equal(str(p), f"2{x}{y}")
            p = x * 2 * y * z
            npt.assert_equal(str(p), f"2{x}{y}{z}")
            p = x * (y * z) * 3
            npt.assert_equal(str(p), f"3{x}{y}{z}")

    def test_div(self):
        for s in self.spaces:
            x, y, z = s[:3]
            p = x / 2
            npt.assert_equal(str(p), f"0.5{x}")
            p = 2 * x / 2
            npt.assert_equal(str(p), f"{x}")
            p = 2 * x / 2 * y
            npt.assert_equal(str(p), f"{x}{y}")

    def test_add(self):
        for s in self.spaces:
            x, y, z = s[:3]

            res = x + y
            npt.assert_equal(str(res), f"{x}+{y}")
            res = x + y + z
            npt.assert_equal(str(res), f"{x}+{y}+{z}")
            res = x + (y + z)
            npt.assert_equal(str(res), f"{x}+{y}+{z}")
            res = x + 2
            npt.assert_equal(str(res), f"{x}+2")
            res = x + 1
            npt.assert_equal(str(res), f"{x}+1")
            res = x + 0
            npt.assert_equal(str(res), f"{x}")
            res = x + 0 * x
            npt.assert_equal(str(res), f"{x}")
            res = x + 0 * y
            npt.assert_equal(str(res), f"{x}")

    def test_sub(self):
        for s in self.spaces:
            x, y, z = s[:3]

            res = -x
            npt.assert_equal(str(res), f"-{x}")
            res = x - y
            npt.assert_equal(str(res), f"{x}-{y}")
            res = x + y - z
            npt.assert_equal(str(res), f"{x}+{y}-{z}")
            res = x - y + z
            npt.assert_equal(str(res), f"{x}-{y}+{z}")
            res = x + (y - z)
            npt.assert_equal(str(res), f"{x}+{y}-{z}")
            res = x - (y + z)
            npt.assert_equal(str(res), f"{x}-{y}-{z}")
            res = x - 2
            npt.assert_equal(str(res), f"{x}-2")
            res = x - 1
            npt.assert_equal(str(res), f"{x}-1")

    def test_superposition(self):
        for s in self.spaces:
            x, y, z = s[:3]

            res = 3 * x + 2 * x
            npt.assert_equal(str(res), f"5{x}")
            res = 3 * x - 2 * x
            npt.assert_equal(str(res), f"{x}")
            res = 3 * x - 5 * x
            npt.assert_equal(str(res), f"-2{x}")
            res = 3 * x - 4 * x
            npt.assert_equal(str(res), f"-{x}")
            res = 3 * x - 3 * x
            npt.assert_equal(str(res), '')
            res = - 4 * x + 3 * x
            npt.assert_equal(str(res), f"-{x}")
            res = 4 * x + y
            npt.assert_equal(str(res), f"4{x}+{y}")
            res = 4 * x - y
            npt.assert_equal(str(res), f"4{x}-{y}")
            res = 4 * x - 2 * y
            npt.assert_equal(str(res), f"4{x}-2{y}")
            res = 4 * x + (- 2 * y)
            npt.assert_equal(str(res), f"4{x}-2{y}")
            res = 4 * x + ((-2) * y)
            npt.assert_equal(str(res), f"4{x}-2{y}")

            p = (4 * x) * (2 * y)
            npt.assert_equal(str(p), f"8{x}{y}")
            p = 4 * x * 2 * y
            npt.assert_equal(str(p), f"8{x}{y}")
            p = 4 * x * y * 2
            npt.assert_equal(str(p), f"8{x}{y}")
            p = (4 * x + 1 * y) * (2 * x + y)
            npt.assert_equal(str(p), f"8{x}{x}+4{x}{y}+2{y}{x}+{y}{y}")

    def test_dual(self):
        for s in self.spaces:
            x, y, z = s[:3]
            res = (3 * x + 2 * x).dual
            npt.assert_equal(str(res), f"0.2{x.dual}")
            res = (5 * x + 2 * y).dual
            npt.assert_equal(str(res), f"0.2{x.dual}+0.5{y.dual}")
            res = (5 * x + 2j * y).dual
            npt.assert_equal(str(res), f"0.2{x.dual}-0.5j{y.dual}")
            res = (5 * x * 2 * y).dual
            npt.assert_equal(str(res), f"0.1{y.dual}{x.dual}")

            x, y, z = s.dual[:3]

            res = 3 * x + 2 * x
            npt.assert_equal(str(res), f"5{x}")
            res = 3 * x - 2 * x
            npt.assert_equal(str(res), f"{x}")
            res = 3 * x - 5 * x
            npt.assert_equal(str(res), f"-2{x}")
            res = 3 * x - 4 * x
            npt.assert_equal(str(res), f"-{x}")
            res = 3 * x - 3 * x
            npt.assert_equal(str(res), '')
            res = - 4 * x + 3 * x
            npt.assert_equal(str(res), f"-{x}")
            res = 4 * x + y
            npt.assert_equal(str(res), f"4{x}+{y}")
            res = 4 * x - y
            npt.assert_equal(str(res), f"4{x}-{y}")
            res = 4 * x - 2 * y
            npt.assert_equal(str(res), f"4{x}-2{y}")
            res = 4 * x + (- 2 * y)
            npt.assert_equal(str(res), f"4{x}-2{y}")
            res = 4 * x + ((-2) * y)
            npt.assert_equal(str(res), f"4{x}-2{y}")

            p = (4 * x) * (2 * y)
            npt.assert_equal(str(p), f"8{x}{y}")
            p = 4 * x * 2 * y
            npt.assert_equal(str(p), f"8{x}{y}")
            p = 4 * x * y * 2
            npt.assert_equal(str(p), f"8{x}{y}")
            p = (4 * x + 1 * y) * (2 * x + y)
            npt.assert_equal(str(p), f"8{x}{x}+4{x}{y}+2{y}{x}+{y}{y}")

    def test_inplace(self):
        for s in self.spaces:
            x, y, z = s[:3]

            res = 2 * x + 3 * y
            res += 2 * x
            npt.assert_equal(str(res), f"4{x}+3{y}")
            res = 2 * x + 3 * y
            res += 3
            npt.assert_equal(str(res), f"2{x}+3{y}+3")
            res = 2 * x + 3 * y
            res += 3 * y
            npt.assert_equal(str(res), f"2{x}+6{y}")
            res = 2 * x - 3 * y
            res += 3 * y
            npt.assert_equal(str(res), f"2{x}")
            res = 2 * x - 3 * y + z
            res += 3 * y
            npt.assert_equal(str(res), f"2{x}+{z}")
            res = 2 * x - 3 * y
            res += 3 * y + z
            npt.assert_equal(str(res), f"2{x}+{z}")

            res = 2 * x
            res += 2 * x
            npt.assert_equal(str(res), f"4{x}")
            res = 2 * x
            res += 3
            npt.assert_equal(str(res), f"2{x}+3")
            res = 2 * x
            res += 3 * y
            npt.assert_equal(str(res), f"2{x}+3{y}")


if __name__ == '__main__':
    unittest.main()


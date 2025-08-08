# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest

import iris.coords
from iris.coords import Cell


class TestCells:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cell1 = iris.coords.Cell(3, [2, 4])
        self.cell2 = iris.coords.Cell(360.0, [350.0, 370.0])

    def test_cell_from_coord(self):
        Cell = iris.coords.Cell
        coord = iris.coords.AuxCoord(np.arange(4) * 1.5, long_name="test", units="1")
        assert Cell(point=0.0, bound=None) == coord.cell(0)
        assert Cell(point=1.5, bound=None) == coord.cell(1)
        assert Cell(point=4.5, bound=None) == coord.cell(-1)
        assert Cell(point=3.0, bound=None) == coord.cell(-2)

        # put bounds on the regular coordinate
        coord.guess_bounds()
        assert Cell(point=0.0, bound=(-0.75, 0.75)) == coord.cell(0)
        assert Cell(point=1.5, bound=(0.75, 2.25)) == coord.cell(1)
        assert Cell(point=4.5, bound=(3.75, 5.25)) == coord.cell(-1)
        assert Cell(point=3.0, bound=(2.25, 3.75)) == coord.cell(-2)
        assert Cell(point=4.5, bound=(3.75, 5.25)) == coord.cell(slice(-1, None))

    def test_cell_from_multidim_coord(self):
        Cell = iris.coords.Cell
        coord = iris.coords.AuxCoord(
            points=np.arange(12).reshape(3, 4),
            long_name="test",
            units="1",
            bounds=np.arange(48).reshape(3, 4, 4),
        )
        with pytest.raises(IndexError, match="did not uniquely identify"):
            coord.cell(0)
        assert Cell(point=3, bound=(12, 13, 14, 15)) == coord.cell((0, 3))

    def test_mod(self):
        # Check that applying the mod function is not modifying the original
        c = self.cell1 % 3
        assert c != self.cell1

        c = self.cell2 % 360
        assert str(c) == "Cell(point=0.0, bound=(350.0, 10.0))"

        c = self.cell2 % 359.13
        assert (
            str(c)
            == "Cell(point=0.8700000000000045, bound=(350.0, 10.870000000000005))"
        )

    def test_contains_point(self):
        c = iris.coords.Cell(359.5, (359.49951, 359.5004))
        assert c.contains_point(359.49951)

    def test_pointless(self):
        with pytest.raises(ValueError, match="Point must be defined"):
            iris.coords.Cell(None, (359.49951, 359.5004))

    def test_add(self):
        # Check that applying the mod function is not modifying the original
        c = self.cell1 + 3
        assert c != self.cell1

        c = self.cell2 + 360
        assert str(c) == "Cell(point=720.0, bound=(710.0, 730.0))"

        c = self.cell2 + 359.13
        assert str(c) == "Cell(point=719.13, bound=(709.13, 729.13))"

    def test_in(self):
        c = iris.coords.Cell(4, None)
        assert c not in [3, 5]
        assert c in [3, 4]

        c = iris.coords.Cell(4, [4, 5])
        assert c not in [3, 6]
        assert c in [3, 4]
        assert c in [3, 5]

        c = iris.coords.Cell(4, [4, 5])
        c1 = iris.coords.Cell(5, [4, 5])
        c2 = iris.coords.Cell(4, [3, 6])

        assert c in [3, c]
        assert c not in [3, c1]
        assert c not in [3, c2]

    def test_coord_equality(self):
        self.d = iris.coords.Cell(1.9, None)
        assert self.d == 1.9
        assert not self.d == [1.5, 1.9]
        assert not self.d != 1.9
        assert self.d >= 1.9
        assert self.d <= 1.9
        assert not self.d > 1.9
        assert not self.d < 1.9
        assert self.d not in [1.5, 3.5]
        assert self.d in [1.5, 1.9]

        assert self.d != 1
        assert not self.d == 1
        assert not self.d >= 2
        assert not self.d <= 1
        assert self.d > 1
        assert self.d < 2

        # Ensure the Cell's operators return NotImplemented.
        class Terry:
            pass

        assert self.d.__eq__(Terry()) == NotImplemented
        assert self.d.__ne__(Terry()) == NotImplemented

    def test_numpy_int_equality(self):
        dtypes = (np.int_, np.int16, np.int32, np.int64)
        for dtype in dtypes:
            val = dtype(3)
            cell = iris.coords.Cell(val, None)
            assert cell == val

    def test_numpy_float_equality(self):
        dtypes = (
            np.float64,
            np.float16,
            np.float32,
            np.float64,
            np.float128,
            np.double,
        )
        for dtype in dtypes:
            val = dtype(3.2)
            cell = iris.coords.Cell(val, None)
            assert cell == val, dtype

    def test_coord_bounds_cmp(self):
        self.e = iris.coords.Cell(0.7, [1.1, 1.9])
        assert self.e == 1.6
        assert not self.e != 1.6
        assert self.e >= 1.9
        assert self.e <= 1.9
        assert not self.e > 1.9
        assert not self.e < 1.9

        assert self.e not in [1.0, 3.5]
        assert self.e in [1.5, 1.9]
        assert self.e != 1
        assert not self.e == 1
        assert not self.e >= 2
        assert not self.e <= 1
        assert self.e > 1
        assert self.e < 2

    def test_cell_cell_cmp(self):
        self.e = iris.coords.Cell(1)
        self.f = iris.coords.Cell(1)

        assert self.e == self.f
        assert hash(self.e) == hash(self.f)

        self.e = iris.coords.Cell(1)
        self.f = iris.coords.Cell(1, [0, 2])

        assert not self.e == self.f
        assert hash(self.e) != hash(self.f)

        self.e = iris.coords.Cell(1, [0, 2])
        self.f = iris.coords.Cell(1, [0, 2])

        assert self.e == self.f
        assert hash(self.e) == hash(self.f)

        self.e = iris.coords.Cell(1, [0, 2])
        self.f = iris.coords.Cell(1, [2, 0])

        assert self.e == self.f
        assert hash(self.e) == hash(self.f)

        self.e = iris.coords.Cell(0.7, [1.1, 1.9])
        self.f = iris.coords.Cell(0.8, [1.1, 1.9])

        assert not self.e == self.f
        assert hash(self.e) != hash(self.f)
        assert not self.e > self.f
        assert self.e <= self.f
        assert self.f >= self.e
        assert not self.f < self.e

        self.e = iris.coords.Cell(0.9, [2, 2.1])
        self.f = iris.coords.Cell(0.8, [1.1, 1.9])

        assert self.e > self.f
        assert not self.e <= self.f
        assert not self.f >= self.e
        assert self.f < self.e

    def test_cmp_contig(self):
        # Test cells that share an edge
        a = iris.coords.Cell(point=1054440.0, bound=(1054080.0, 1054800.0))
        b = iris.coords.Cell(point=1055160.0, bound=(1054800.0, 1055520.0))
        assert a < b
        assert a <= b
        assert not a == b
        assert not a >= b
        assert not a > b

    def test_overlap_order(self):
        # Test cells that overlap still sort correctly.
        cells = [
            Cell(point=375804.0, bound=(375792.0, 375816.0)),
            Cell(point=375672.0, bound=(375660.0, 375684.0)),
            Cell(point=375792.0, bound=(375780.0, 375804.0)),
            Cell(point=375960.0, bound=(375948.0, 375972.0)),
        ]
        sorted_cells = [
            Cell(point=375672.0, bound=(375660.0, 375684.0)),
            Cell(point=375792.0, bound=(375780.0, 375804.0)),
            Cell(point=375804.0, bound=(375792.0, 375816.0)),
            Cell(point=375960.0, bound=(375948.0, 375972.0)),
        ]
        assert sorted(cells) == sorted_cells

    def _check_permutations(self, a, b, a_lt_b, a_le_b, a_eq_b):
        assert (a < b) == a_lt_b
        assert (a <= b) == a_le_b
        assert (a == b) == a_eq_b

        assert (a > b) != a_le_b
        assert (a >= b) != a_lt_b
        assert (a != b) != a_eq_b

    def _check_all_permutations(self, a, b, a_lt_b, a_le_b, a_eq_b):
        self._check_permutations(a, b, a_lt_b, a_le_b, a_eq_b)
        self._check_permutations(b, a, not a_le_b, not a_lt_b, a_eq_b)

    def test_comparison_numeric(self):
        # Check what happens when you compare a simple number with a
        # point-only Cell.
        self._check_permutations(9, Cell(10), True, True, False)
        self._check_permutations(10, Cell(10), False, True, True)
        self._check_permutations(11, Cell(10), False, False, False)

    def test_comparison_numeric_with_bounds(self):
        # Check what happens when you compare a simple number with a
        # point-and-bound Cell.
        self._check_permutations(7, Cell(10, [8, 12]), True, True, False)
        self._check_permutations(8, Cell(10, [8, 12]), False, True, True)
        self._check_permutations(9, Cell(10, [8, 12]), False, True, True)
        self._check_permutations(10, Cell(10, [8, 12]), False, True, True)
        self._check_permutations(11, Cell(10, [8, 12]), False, True, True)
        self._check_permutations(12, Cell(10, [8, 12]), False, True, True)
        self._check_permutations(13, Cell(10, [8, 12]), False, False, False)

# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris._merge.ProtoCube` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import warnings

import mock
import numpy as np
import numpy.ma as ma

import iris
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.exceptions import MergeError
from iris.coords import DimCoord, AuxCoord
from iris.unit import Unit


class Test_register__cube_sig(tests.IrisTest):
    # Test potential registration failures.

    def setUp(self):
        self.cube1 = iris.cube.Cube([1, 2, 3], standard_name="air_temperature",
                                    units='K', attributes={"mint": "thin"})

    def check_fail(self, cube2, substrs):
        if isinstance(substrs, basestring):
            substrs = [substrs]

        proto_cube = iris._merge.ProtoCube(self.cube1)
        with self.assertRaises(MergeError) as arc:
            proto_cube.register(cube2, or_fail=True)

#         print arc.exception.message
        self.assertTrue("cube differences:" in arc.exception.message,
                        arc.exception.message)
        for substr in substrs:
            self.assertTrue(substr in arc.exception.message,
                            arc.exception.message)

    def test_defn_standard_name(self):
        cube2 = self.cube1.copy()
        cube2.standard_name = "air_pressure"
        self.check_fail(cube2, "cube.standard_name")

    def test_defn_long_name(self):
        self.cube1.rename("Arthur")

        cube2 = self.cube1.copy()
        cube2.rename("Belling")
        self.check_fail(cube2, "cube.long_name")

    def test_defn_var_name(self):
        self.cube1.standard_name = None
        self.cube1.var_name = "Arthur"

        cube2 = self.cube1.copy()
        cube2.var_name = "Nudge"
        self.check_fail(cube2, "cube.var_name")

    def test_defn_units(self):
        cube2 = self.cube1.copy()
        cube2.units = 'C'
        self.check_fail(cube2, "cube.units")

    def test_defn_attributes_unequal(self):
        cube2 = self.cube1.copy()
        cube2.attributes['mint'] = 'waffer-thin'
        self.check_fail(cube2, "cube.attributes")

    def test_defn_attributes_superset(self):
        cube2 = self.cube1.copy()
        cube2.attributes['stuffed'] = 'yes'
        self.check_fail(cube2, "cube.attributes")

    def test_defn_cell_method(self):
        cube2 = self.cube1.copy()
        cube2.add_cell_method(iris.coords.CellMethod('monty', ('python',)))
        self.check_fail(cube2, "cube.cell_methods")

    def test_data_shape(self):
        cube2 = self.cube1[1:]
        self.check_fail(cube2, "data_shape:")

    def test_data_type(self):
        cube2 = self.cube1.copy(data=self.cube1.data.astype(np.int8))
        self.check_fail(cube2, "data_type:")

    def test_mdi(self):
        cube2 = self.cube1.copy(data=ma.array(self.cube1.data))
        cube2.data.fill_value = 12345
        self.check_fail(cube2, "mdi:")

    def test_noise(self):
        # Test a massive set of all defn diffs to make sure it's not noise.
        self.cube1.var_name = "Arthur"

        cube2 = self.cube1[1:]
        cube2.data = cube2.data.astype(np.int8)
        cube2.data = ma.array(cube2.data)
        cube2.data.fill_value = 12345
        cube2.standard_name = "air_pressure"
        cube2.var_name = "Nudge"
        cube2.attributes['stuffed'] = 'yes'
        cube2.attributes['mint'] = 'waffer-thin'
        cube2.add_cell_method(iris.coords.CellMethod('monty', ('python',)))

        # Check the actual message, so we've got a readable reference text.
        proto_cube = iris._merge.ProtoCube(self.cube1)
        with self.assertRaises(MergeError) as arc:
            proto_cube.register(cube2, or_fail=True)

        # pending #884
        self.assertString(
            arc.exception.message,
            ["unit", "_merge", "ProtoCube", "register__cube_sig", "noise.txt"])


class Test_register__coord_sig(tests.IrisTest):

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3, 3)))

    def check_fail(self, cube2, substrs, verbose=False):
        if isinstance(substrs, basestring):
            substrs = [substrs]

        proto_cube = iris._merge.ProtoCube(self.cube1)
        with self.assertRaises(MergeError) as arc:
            proto_cube.register(cube2, or_fail=True, fail_verbose=verbose)

#         print arc.exception.message
        self.assertTrue("coord differences:" in arc.exception.message,
                        arc.exception.message)
        for substr in substrs:
            self.assertTrue(substr in arc.exception.message,
                            arc.exception.message)

    def test_scalar_defns_one_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_aux_coord(DimCoord([1], standard_name="latitude"))
        self.check_fail(cube2, ["scalar coord: latitude"])

    def test_scalar_defns_both_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_aux_coord(DimCoord([1], standard_name="latitude"))
        self.cube1.add_aux_coord(DimCoord([1], standard_name="longitude"))
        self.check_fail(
            cube2,
            ["scalar coord: latitude", "scalar coord: longitude"])

    def test_vector_dim_coords_and_dims_one_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.check_fail(cube2, ["dim coord: latitude"])

    def test_vector_dim_coords_and_dims_both_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="longitude"), 0)
        self.check_fail(cube2, ["dim coord: latitude"])

    def test_vector_aux_coords_and_dims_one_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_aux_coord(DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.check_fail(cube2, ["aux coord: latitude"])

    def test_vector_aux_coords_and_dims_both_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_aux_coord(DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.cube1.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="longitude"), 0)
        self.check_fail(cube2, ["aux coord: latitude"])

    def test_factory_defns_one_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.check_fail(cube2, ["factory", "different"])

    def test_factory_defns_both_extra(self):
        cube2 = self.cube1.copy()
        cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.cube1.add_aux_factory(mock.MagicMock(spec=HybridPressureFactory))
        self.check_fail(cube2, ["factory", "different"])

    def test_noise(self):
        cube2 = self.cube1.copy()

        # scalar
        cube2.add_aux_coord(DimCoord([1], long_name="liff"))
        cube2.add_aux_coord(DimCoord([1], long_name="life"))
        cube2.add_aux_coord(DimCoord([1], long_name="like"))

        self.cube1.add_aux_coord(DimCoord([1], var_name="ming"))
        self.cube1.add_aux_coord(DimCoord([1], var_name="mong"))
        self.cube1.add_aux_coord(DimCoord([1], var_name="moog"))

        # aux
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="latitude"), 0)
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="longitude"), 1)
        cube2.add_dim_coord(DimCoord([1, 2, 3], standard_name="altitude"), 2)

        self.cube1.add_dim_coord(DimCoord([1, 2, 3], long_name="equinimity"),
                                 0)
        self.cube1.add_dim_coord(DimCoord([1, 2, 3], long_name="equinomity"),
                                 1)
        self.cube1.add_dim_coord(DimCoord([1, 2, 3], long_name="equinumity"),
                                 2)

        # dim
        cube2.add_aux_coord(DimCoord([1, 2, 3], var_name="one"), 0)
        cube2.add_aux_coord(DimCoord([1, 2, 3], var_name="two"), 1)
        cube2.add_aux_coord(DimCoord([1, 2, 3], var_name="three"), 2)

        self.cube1.add_aux_coord(DimCoord([1, 2, 3], long_name="ay"), 0)
        self.cube1.add_aux_coord(DimCoord([1, 2, 3], long_name="bee"), 1)
        self.cube1.add_aux_coord(DimCoord([1, 2, 3], long_name="cee"), 2)

        # factory
        cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.cube1.add_aux_factory(mock.MagicMock(spec=HybridPressureFactory))

        # Check the actual message, so we've got a readable reference text.
        proto_cube = iris._merge.ProtoCube(self.cube1)
        with self.assertRaises(MergeError) as arc:
            proto_cube.register(cube2, or_fail=True)

        # pending #884
        self.assertString(
            arc.exception.message,
            ["unit", "_merge", "ProtoCube", "register__coord_sig",
             "noise.txt"])


if __name__ == "__main__":
    tests.main()

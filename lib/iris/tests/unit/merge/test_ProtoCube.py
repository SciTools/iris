# (C) British Crown Copyright 2013 - 2014, Met Office
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

import mock
import numpy as np
import numpy.ma as ma

import iris
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.exceptions import MergeError

from iris.coords import DimCoord, AuxCoord
from iris.unit import Unit


class _MergeTest(object):
    # A mixin test class for common test methods implementation.

    # used by check routine: inheritors must implement it
    _mergetest_type = NotImplementedError

    def check_merge_fails_with_message(self):
        proto_cube = iris._merge.ProtoCube(self.cube1)
        with self.assertRaises(MergeError) as arc:
            proto_cube.register(self.cube2, error_on_mismatch=True)
        return str(arc.exception)

    def check_fail(self, *substrs):
        if isinstance(substrs, basestring):
            substrs = [substrs]
        msg = self.check_merge_fails_with_message()
        for substr in substrs:
            self.assertIn(substr, msg)


class Test_register__CubeSig(_MergeTest, tests.IrisTest):
    # Test potential registration failures.

    _mergetest_type = 'cube'

    def setUp(self):
        self.cube1 = iris.cube.Cube([1, 2, 3], standard_name="air_temperature",
                                    units='K', attributes={"mint": "thin"})
        self.cube2 = self.cube1.copy()

    def test_defn_standard_name(self):
        self.cube2.standard_name = "air_pressure"
        self.check_fail('cube.standard_name', 'air_pressure',
                        'air_temperature')

    def test_defn_long_name(self):
        self.cube1.rename("Arthur")
        self.cube2 = self.cube1.copy()
        self.cube2.rename("Belling")
        self.check_fail('cube.long_name', 'Arthur', 'Belling')

    def test_defn_var_name(self):
        self.cube1.standard_name = None
        self.cube1.var_name = "Arthur"
        self.cube2 = self.cube1.copy()
        self.cube2.var_name = "Nudge"
        self.check_fail('cube.var_name', 'Arthur', 'Nudge')

    def test_defn_units(self):
        self.cube2.units = 'C'
        self.check_fail('cube.units', 'C', 'K')

    def test_defn_attributes_unequal(self):
        self.cube2.attributes['mint'] = 'waffer-thin'
        self.check_fail('cube.attributes', 'mint')

    def test_defn_attributes_superset(self):
        self.cube2.attributes['stuffed'] = 'yes'
        self.check_fail('cube.attributes', 'keys', 'stuffed')

    def test_defn_attributes_multidiff(self):
        self.cube1.attributes['tom'] = 1
        self.cube1.attributes['dick'] = 2
        self.cube1.attributes['harry'] = 3
        self.cube2 = self.cube1.copy()
        self.cube2.attributes['mint'] = 'diddle'
        self.cube2.attributes['dick'] = 'fiddle'
        self.check_fail('cube.attributes', 'dick', 'mint')

    def test_defn_cell_method(self):
        self.cube2.add_cell_method(
            iris.coords.CellMethod('monty', ('python',)))
        self.check_fail('cube.cell_methods')

    def test_data_shape(self):
        self.cube2 = self.cube1[1:]
        self.check_fail('cube.shape', '(2,)', '(3,)')

    def test_data_type(self):
        self.cube2.data = self.cube1.data.astype(np.int8)
        self.check_fail('data dtype', 'int64', 'int8')

    def test_fill_value(self):
        self.cube2.data = ma.array(self.cube2.data)
        self.cube2.data.fill_value = 12345
        self.check_fail('fill_value', '12345', '999999')

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
        self.cube2 = cube2
        msg = self.check_merge_fails_with_message()

        # pending #884
        self.assertString(msg, self.result_path(ext='txt'))


class Test_register__CoordSig_general(_MergeTest, tests.IrisTest):

    _mergetest_type = 'coord'

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3, 3)))
        self.cube2 = self.cube1.copy()

    def test_scalar_defns_one_extra(self):
        self.cube2.add_aux_coord(DimCoord([1], standard_name="latitude"))
        self.check_fail('aux_coords (scalar)', 'latitude')

    def test_scalar_defns_both_extra(self):
        self.cube2.add_aux_coord(DimCoord([1], standard_name="latitude"))
        self.cube1.add_aux_coord(DimCoord([1], standard_name="longitude"))
        self.check_fail('aux_coords (scalar)', 'latitude', 'longitude')

    def test_vector_dim_coords_and_dims_one_extra(self):
        self.cube2.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.check_fail('dim_coords', 'latitude')

    def test_vector_dim_coords_and_dims_both_extra(self):
        self.cube2.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], standard_name="longitude"), 0)
        self.check_fail('dim_coords', 'latitude', 'longitude')

    def test_vector_aux_coords_and_dims_one_extra(self):
        self.cube2.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.check_fail('aux_coords (non-scalar)', 'latitude')

    def test_vector_aux_coords_and_dims_both_extra(self):
        self.cube2.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="latitude"), 0)
        self.cube1.add_aux_coord(
            DimCoord([1, 2, 3], standard_name="longitude"), 0)
        self.check_fail('aux_coords (non-scalar)', 'latitude', 'longitude')

    def test_factory_defns_one_extra(self):
        self.cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.check_fail("cube.aux_factories", "differ")

    def test_factory_defns_both_extra(self):
        self.cube2.add_aux_factory(mock.MagicMock(spec=HybridHeightFactory))
        self.cube1.add_aux_factory(mock.MagicMock(spec=HybridPressureFactory))
        self.check_fail("cube.aux_factories", "differ")

    def test_noise(self):
        cube2 = self.cube2

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

        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], long_name="equinimity"), 0)
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], long_name="equinomity"), 1)
        self.cube1.add_dim_coord(
            DimCoord([1, 2, 3], long_name="equinumity"), 2)

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
        self.cube2 = cube2
        msg = self.check_merge_fails_with_message()

        # pending #884
        self.assertString(msg, self.result_path(ext='txt'))


class _MergeTest_coordprops(_MergeTest):
    # A mixin test class for common coordinate properties tests.

    # This must be implemented by inheritors.
    _mergetest_type = NotImplementedError

    def test_nochange(self):
        # this should simply succeed..
        proto_cube = iris._merge.ProtoCube(self.cube1)
        proto_cube.register(self.cube2, error_on_mismatch=True)

    def _props_fail(self, *terms):
        self.check_fail(self._mergetest_type, self.test_coord.name(), *terms)

    def test_standard_name(self):
        self.test_coord.standard_name = 'soil_temperature'
        self._props_fail('air_temperature', 'soil_temperature')

    def test_long_name(self):
        self.test_coord.long_name = 'alternate_name'
        self._props_fail('air_temperature')

    def test_var_name(self):
        self.test_coord.var_name = 'alternate_name'
        self._props_fail('air_temperature')

    def test_units(self):
        self.test_coord.units = 'm'
        self._props_fail('air_temperature')

    def test_attrs_unequal(self):
        self.test_coord.attributes['att_a'] = 99
        self._props_fail('air_temperature')

    def test_attrs_set(self):
        self.test_coord.attributes['att_extra'] = 101
        self._props_fail('air_temperature')

    def test_coord_system(self):
        self.test_coord.coord_system = mock.Mock()
        self._props_fail('air_temperature')


class Test_register__CoordSig_scalar(_MergeTest_coordprops, tests.IrisTest):

    _mergetest_type = 'aux_coords (scalar)'

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3, 3)))
        self.cube1.add_aux_coord(DimCoord(
            [1],
            standard_name="air_temperature",
            long_name='eg_scalar',
            var_name='t1',
            units='K',
            attributes={'att_a': 1, 'att_b': 2},
            coord_system=None))
        self.test_coord = self.cube1.coord('air_temperature')
        self.cube2 = self.cube1.copy()


class _MergeTest_coordprops_vect(_MergeTest_coordprops):
    # A derived mixin test class.
    # Adds extra props test for aux+dim coords (test points, bounds + dims)
    _mergetest_type = NotImplementedError
    _coord_typename = NotImplementedError

    def test_points(self):
        self.test_coord.points = self.test_coord.points + 1.0
        self.check_fail(self._mergetest_type, 'air_temperature')

    def test_bounds(self):
        self.test_coord.bounds = self.test_coord.bounds + 1.0
        self.check_fail(self._mergetest_type, 'air_temperature')

    def test_dims(self):
        self.cube2.remove_coord(self.test_coord)
        cube2_add_method = getattr(self.cube2, 'add_'+self._coord_typename)
        cube2_add_method(self.test_coord, (1,))
        self.check_fail(self._mergetest_type, 'mapping')


class Test_register__CoordSig_dim(_MergeTest_coordprops_vect, tests.IrisTest):

    _mergetest_type = 'dim_coords'
    _coord_typename = 'dim_coord'

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3)))
        self.cube1.add_dim_coord(DimCoord(
            [15, 25, 35],
            bounds=[[10, 20], [20, 30], [30, 40]],
            standard_name="air_temperature",
            long_name='eg_scalar',
            var_name='t1',
            units='K',
            attributes={'att_a': 1, 'att_b': 2},
            coord_system=None),
            (0,))
        self.test_coord = self.cube1.coord('air_temperature')
        self.cube2 = self.cube1.copy()

    def test_circular(self):
        # Extra failure mode that only applies to dim coords
        self.test_coord.circular = True
        self.check_fail(self._mergetest_type, 'air_temperature')


class Test_register__CoordSig_aux(_MergeTest_coordprops_vect, tests.IrisTest):

    _mergetest_type = 'aux_coords (non-scalar)'
    _coord_typename = 'aux_coord'

    def setUp(self):
        self.cube1 = iris.cube.Cube(np.zeros((3, 3)))
        self.cube1.add_aux_coord(AuxCoord(
            [65, 45, 85],
            bounds=[[60, 70], [40, 50], [80, 90]],
            standard_name="air_temperature",
            long_name='eg_scalar',
            var_name='t1',
            units='K',
            attributes={'att_a': 1, 'att_b': 2},
            coord_system=None),
            (0,))
        self.test_coord = self.cube1.coord('air_temperature')
        self.cube2 = self.cube1.copy()


if __name__ == "__main__":
    tests.main()

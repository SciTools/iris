# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_and_add_dimension_coordinate`."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from unittest import mock
import warnings

import numpy as np
import pytest

from iris.coord_systems import RotatedGeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import CannotAddError
from iris.fileformats._nc_load_rules.helpers import build_and_add_dimension_coordinate
from iris.loading import LOAD_PROBLEMS


def _make_bounds_var(bounds, dimensions, units):
    bounds = np.array(bounds)
    cf_data = mock.Mock(spec=[])
    # we want to mock the absence of flag attributes to helpers.get_attr_units
    # see https://docs.python.org/3/library/unittest.mock.html#deleting-attributes
    del cf_data.flag_values
    del cf_data.flag_masks
    del cf_data.flag_meanings
    result = mock.Mock(
        dimensions=dimensions,
        cf_name="wibble_bnds",
        cf_data=cf_data,
        units=units,
        calendar=None,
        shape=bounds.shape,
        size=np.prod(bounds.shape),
        dtype=bounds.dtype,
        __getitem__=lambda self, key: bounds[key],
    )
    delattr(result, "_data_array")
    return result


class RulesTestMixin:
    def setUp(self):
        # Create dummy pyke engine.
        self.engine = mock.Mock(
            cube=mock.Mock(),
            cf_var=mock.Mock(dimensions=("foo", "bar")),
            filename="DUMMY",
            cube_parts=dict(coordinates=[]),
        )

        # Create patch for deferred loading that prevents attempted
        # file access. This assumes that self.cf_coord_var and
        # self.cf_bounds_var are defined in the test case.
        def patched__getitem__(proxy_self, keys):
            for var in (self.cf_coord_var, self.cf_bounds_var):
                if proxy_self.variable_name == var.cf_name:
                    return var[keys]
            raise RuntimeError()

        self.deferred_load_patch = mock.patch(
            "iris.fileformats.netcdf.NetCDFDataProxy.__getitem__",
            new=patched__getitem__,
        )

        # Patch the helper function that retrieves the bounds cf variable.
        # This avoids the need for setting up further mocking of cf objects.
        self.use_climatology_bounds = False  # Set this when you need to.

        def get_cf_bounds_var(coord_var):
            return self.cf_bounds_var, self.use_climatology_bounds

        self.get_cf_bounds_var_patch = mock.patch(
            "iris.fileformats._nc_load_rules.helpers.get_cf_bounds_var",
            new=get_cf_bounds_var,
        )


class TestCoordConstruction(tests.IrisTest, RulesTestMixin):
    def setUp(self):
        # Call parent setUp explicitly, because of how unittests work.
        RulesTestMixin.setUp(self)

        bounds = np.arange(12).reshape(6, 2)
        dimensions = ("x", "nv")
        units = "days since 1970-01-01"
        self.cf_bounds_var = _make_bounds_var(bounds, dimensions, units)
        self.bounds = bounds

        # test_dimcoord_not_added() and test_auxcoord_not_added have been
        #  written in pytest-style, but the rest of the class is pending
        #  migration. Defining self.monkeypatch (not the
        #  typical practice in pure pytest) allows this transitional state.
        self.monkeypatch = pytest.MonkeyPatch()

    def _set_cf_coord_var(self, points):
        self.cf_coord_var = mock.Mock(
            dimensions=("foo",),
            cf_name="wibble",
            cf_data=mock.Mock(spec=[]),
            standard_name=None,
            long_name="wibble",
            units="days since 1970-01-01",
            calendar=None,
            shape=points.shape,
            size=np.prod(points.shape),
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key],
            cf_attrs=lambda: [("foo", "a"), ("bar", "b")],
        )
        delattr(self.cf_coord_var, "_data_array")

    def check_case_dim_coord_construction(self, climatology=False):
        # Test a generic dimension coordinate, with or without
        # a climatological coord.
        self.use_climatology_bounds = climatology
        self._set_cf_coord_var(np.arange(6))

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
            climatological=climatology,
        )

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

    def test_dim_coord_construction(self):
        self.check_case_dim_coord_construction(climatology=False)

    def test_dim_coord_construction__climatology(self):
        self.check_case_dim_coord_construction(climatology=True)

    def test_dim_coord_construction_masked_array(self):
        self._set_cf_coord_var(
            np.ma.array(
                np.arange(6),
                mask=[True, False, False, False, False, False],
                fill_value=-999,
            )
        )

        expected_coord = DimCoord(
            np.array([-999, 1, 2, 3, 4, 5]),
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
        )

        with warnings.catch_warnings(record=True) as w:
            # Asserts must lie within context manager because of deferred
            # loading.
            with self.deferred_load_patch, self.get_cf_bounds_var_patch:
                build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

                # Test that expected coord is built and added to cube.
                self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

            # Assert warning is raised
            assert len(w) == 1
            assert "Gracefully filling" in w[0].message.args[0]

    def test_dim_coord_construction_masked_array_mask_does_nothing(self):
        self._set_cf_coord_var(
            np.ma.array(
                np.arange(6),
                mask=False,
            )
        )

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
        )

        with warnings.catch_warnings(record=True) as w:
            # Asserts must lie within context manager because of deferred
            # loading.
            with self.deferred_load_patch, self.get_cf_bounds_var_patch:
                build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

                # Test that expected coord is built and added to cube.
                self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

            # Assert no warning is raised
            assert len(w) == 0

    def test_dim_coord_construction_masked_bounds_mask_does_nothing(self):
        self.bounds = np.ma.array(np.arange(12).reshape(6, 2), mask=False)
        self._set_cf_coord_var(np.arange(6))

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
        )

        with warnings.catch_warnings(record=True) as w:
            # Asserts must lie within context manager because of deferred
            # loading.
            with self.deferred_load_patch, self.get_cf_bounds_var_patch:
                build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

                # Test that expected coord is built and added to cube.
                self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

            # Assert no warning is raised
            assert len(w) == 0

    def test_with_coord_system(self):
        self._set_cf_coord_var(np.arange(6))
        coord_system = RotatedGeogCS(
            grid_north_pole_latitude=45.0, grid_north_pole_longitude=45.0
        )

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
            coord_system=coord_system,
        )

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(
                self.engine, self.cf_coord_var, coord_system=coord_system
            )

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

    def test_bad_coord_system(self):
        self._set_cf_coord_var(np.arange(6))
        coord_system = RotatedGeogCS(
            grid_north_pole_latitude=45.0, grid_north_pole_longitude=45.0
        )

        def mock_setter(self, value):
            # Currently coord_system is not validated during setting, but we
            #  want to ensure that any problems _would_ be handled, so fake
            #  an error.
            if value is not None:
                raise ValueError("test_bad_coord_system")
            else:
                self._metadata_manager.coord_system = value

        with mock.patch.object(
            DimCoord,
            "coord_system",
            new=property(DimCoord.coord_system.fget, mock_setter),
        ):
            with self.deferred_load_patch, self.get_cf_bounds_var_patch:
                build_and_add_dimension_coordinate(
                    self.engine, self.cf_coord_var, coord_system=coord_system
                )
                load_problem = LOAD_PROBLEMS.problems[-1]
                self.assertIn(
                    "test_bad_coord_system",
                    "".join(load_problem.stack_trace.format()),
                )

    def test_aux_coord_construction(self):
        # Use non monotonically increasing coordinates to force aux coord
        # construction.
        self._set_cf_coord_var(np.array([1, 3, 2, 4, 6, 5]))

        expected_coord = AuxCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=self.bounds,
        )

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_aux_coord.assert_called_with(expected_coord, [0])
            load_problem = LOAD_PROBLEMS.problems[-1]
            self.assertIn(
                "creating 'wibble' auxiliary coordinate instead",
                "".join(load_problem.stack_trace.format()),
            )
            self.assertTrue(load_problem.handled)

    def test_dimcoord_not_added(self):
        # Confirm that the coord will be skipped if a CannotAddError is raised
        #  when attempting to add.
        def mock_add_dim_coord(_, __):
            raise CannotAddError("foo")

        with self.monkeypatch.context() as m:
            m.setattr(self.engine.cube, "add_dim_coord", mock_add_dim_coord)

            self._set_cf_coord_var(np.arange(6))

            with self.deferred_load_patch, self.get_cf_bounds_var_patch:
                build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert load_problem.stack_trace.exc_type is CannotAddError
        assert isinstance(load_problem.loaded, DimCoord)
        assert [type(i[0]) for i in self.engine.cube_parts["coordinates"]] == [AuxCoord]

    def test_auxcoord_not_added(self):
        # Confirm that a gracefully-created auxiliary coord will also be
        #  skipped if a CannotAddError is raised when attempting to add.
        def mock_add_aux_coord(_, __):
            raise CannotAddError("foo")

        with self.monkeypatch.context() as m:
            m.setattr(self.engine.cube, "add_aux_coord", mock_add_aux_coord)

            self._set_cf_coord_var(np.array([1, 3, 2, 4, 6, 5]))

            with self.deferred_load_patch, self.get_cf_bounds_var_patch:
                build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert load_problem.stack_trace.exc_type is CannotAddError
        assert self.engine.cube_parts["coordinates"] == []

    def test_unhandlable_error(self):
        # Confirm that the code can redirect an error to LOAD_PROBLEMS even
        #  when there is no specific handling code for it.
        with self.monkeypatch.context() as m:
            m.setattr(self.engine, "cube", "foo")
            n_problems = len(LOAD_PROBLEMS.problems)
            self._set_cf_coord_var(np.array([1, 3, 2, 4, 6, 5]))
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)
            self.assertTrue(len(LOAD_PROBLEMS.problems) > n_problems)

        assert self.engine.cube_parts["coordinates"] == []

    def test_problem_destination(self):
        # Confirm that the destination of the problem is set correctly.
        with self.monkeypatch.context() as m:
            m.setattr(self.engine, "cube", "foo")
            self._set_cf_coord_var(np.array([1, 3, 2, 4, 6, 5]))
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

            destination = LOAD_PROBLEMS.problems[-1].destination
            assert destination.iris_class is Cube
            assert destination.identifier == self.engine.cf_var.cf_name

        assert self.engine.cube_parts["coordinates"] == []


class TestBoundsVertexDim(tests.IrisTest, RulesTestMixin):
    def setUp(self):
        # Call parent setUp explicitly, because of how unittests work.
        RulesTestMixin.setUp(self)
        # Create test coordinate cf variable.
        points = np.arange(6)
        self.cf_coord_var = mock.Mock(
            dimensions=("foo",),
            cf_name="wibble",
            standard_name=None,
            long_name="wibble",
            cf_data=mock.Mock(spec=[]),
            units="km",
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key],
        )

    def test_slowest_varying_vertex_dim__normalise_bounds(self):
        # Create the bounds cf variable.
        bounds = np.arange(12).reshape(2, 6) * 1000
        dimensions = ("nv", "foo")
        units = "m"
        self.cf_bounds_var = _make_bounds_var(bounds, dimensions, units)

        # Expected bounds on the resulting coordinate should be rolled so that
        # the vertex dimension is at the end.
        expected_bounds = bounds.transpose() / 1000
        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=expected_bounds,
        )

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

            # Test that engine.cube_parts container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.cube_parts["coordinates"], expected_list)

    def test_fastest_varying_vertex_dim__normalise_bounds(self):
        bounds = np.arange(12).reshape(6, 2) * 1000
        dimensions = ("foo", "nv")
        units = "m"
        self.cf_bounds_var = _make_bounds_var(bounds, dimensions, units)

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=bounds / 1000,
        )

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

            # Test that engine.cube_parts container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.cube_parts["coordinates"], expected_list)

    def test_fastest_with_different_dim_names__normalise_bounds(self):
        # Despite the dimension names 'x' differing from the coord's
        # which is 'foo' (as permitted by the cf spec),
        # this should still work because the vertex dim is the fastest varying.
        bounds = np.arange(12).reshape(6, 2) * 1000
        dimensions = ("x", "nv")
        units = "m"
        self.cf_bounds_var = _make_bounds_var(bounds, dimensions, units)

        expected_coord = DimCoord(
            self.cf_coord_var[:],
            long_name=self.cf_coord_var.long_name,
            var_name=self.cf_coord_var.cf_name,
            units=self.cf_coord_var.units,
            bounds=bounds / 1000,
        )

        # Asserts must lie within context manager because of deferred loading.
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(self.engine, self.cf_coord_var)

            # Test that expected coord is built and added to cube.
            self.engine.cube.add_dim_coord.assert_called_with(expected_coord, 0)

            # Test that engine.cube_parts container is correctly populated.
            expected_list = [(expected_coord, self.cf_coord_var.cf_name)]
            self.assertEqual(self.engine.cube_parts["coordinates"], expected_list)


class TestCircular(tests.IrisTest, RulesTestMixin):
    # Test the rules logic for marking a coordinate "circular".
    def setUp(self):
        # Call parent setUp explicitly, because of how unittests work.
        RulesTestMixin.setUp(self)
        self.cf_bounds_var = None

    def _make_vars(self, points, bounds=None, units="degrees"):
        points = np.array(points)
        self.cf_coord_var = mock.MagicMock(
            dimensions=("foo",),
            cf_name="wibble",
            standard_name=None,
            long_name="wibble",
            cf_data=mock.Mock(spec=[]),
            units=units,
            shape=points.shape,
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key],
        )
        if bounds:
            bounds = np.array(bounds).reshape(self.cf_coord_var.shape + (2,))
            dimensions = ("x", "nv")
            self.cf_bounds_var = _make_bounds_var(bounds, dimensions, units)

    def _check_circular(self, circular, *args, **kwargs):
        if "coord_name" in kwargs:
            coord_name = kwargs.pop("coord_name")
        else:
            coord_name = "longitude"
        self._make_vars(*args, **kwargs)
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(
                self.engine, self.cf_coord_var, coord_name=coord_name
            )
            self.assertEqual(self.engine.cube.add_dim_coord.call_count, 1)
            coord, dims = self.engine.cube.add_dim_coord.call_args[0]
        self.assertEqual(coord.circular, circular)

    def check_circular(self, *args, **kwargs):
        self._check_circular(True, *args, **kwargs)

    def check_noncircular(self, *args, **kwargs):
        self._check_circular(False, *args, **kwargs)

    def test_single_zero_noncircular(self):
        self.check_noncircular([0.0])

    def test_single_lt_modulus_noncircular(self):
        self.check_noncircular([-1.0])

    def test_single_eq_modulus_circular(self):
        self.check_circular([360.0])

    def test_single_gt_modulus_circular(self):
        self.check_circular([361.0])

    def test_single_bounded_noncircular(self):
        self.check_noncircular([180.0], bounds=[90.0, 240.0])

    def test_single_bounded_circular(self):
        self.check_circular([180.0], bounds=[90.0, 450.0])

    def test_multiple_unbounded_circular(self):
        self.check_circular([0.0, 90.0, 180.0, 270.0])

    def test_non_angle_noncircular(self):
        points = [0.0, 90.0, 180.0, 270.0]
        self.check_noncircular(points, units="m")

    def test_non_longitude_noncircular(self):
        points = [0.0, 90.0, 180.0, 270.0]
        self.check_noncircular(points, coord_name="depth")

    def test_multiple_unbounded_irregular_noncircular(self):
        self.check_noncircular([0.0, 90.0, 189.999, 270.0])

    def test_multiple_unbounded_offset_circular(self):
        self.check_circular([45.0, 135.0, 225.0, 315.0])

    def test_multiple_unbounded_shortrange_circular(self):
        self.check_circular([0.0, 90.0, 180.0, 269.9999])

    def test_multiple_bounded_circular(self):
        self.check_circular(
            [0.0, 120.3, 240.0],
            bounds=[[-45.0, 50.0], [100.0, 175.0], [200.0, 315.0]],
        )

    def test_multiple_bounded_noncircular(self):
        self.check_noncircular(
            [0.0, 120.3, 240.0],
            bounds=[[-45.0, 50.0], [100.0, 175.0], [200.0, 355.0]],
        )


class TestCircularScalar(tests.IrisTest, RulesTestMixin):
    def setUp(self):
        RulesTestMixin.setUp(self)

    def _make_vars(self, bounds):
        # Create cf vars for the coordinate and its bounds.
        # Note that for a scalar the shape of the array from
        # the cf var is (), rather than (1,).
        points = np.array([0.0])
        units = "degrees"
        self.cf_coord_var = mock.Mock(
            dimensions=(),
            cf_name="wibble",
            standard_name=None,
            long_name="wibble",
            units=units,
            cf_data=mock.Mock(spec=[]),
            shape=(),
            dtype=points.dtype,
            __getitem__=lambda self, key: points[key],
        )

        bounds = np.array(bounds)
        dimensions = ("bnds",)
        self.cf_bounds_var = _make_bounds_var(bounds, dimensions, units)

    def _assert_circular(self, value):
        with self.deferred_load_patch, self.get_cf_bounds_var_patch:
            build_and_add_dimension_coordinate(
                self.engine, self.cf_coord_var, coord_name="longitude"
            )
            self.assertEqual(self.engine.cube.add_aux_coord.call_count, 1)
            coord, dims = self.engine.cube.add_aux_coord.call_args[0]
        self.assertEqual(coord.circular, value)

    def test_two_bounds_noncircular(self):
        self._make_vars([0.0, 180.0])
        self._assert_circular(False)

    def test_two_bounds_circular(self):
        self._make_vars([0.0, 360.0])
        self._assert_circular(True)

    def test_two_bounds_circular_decreasing(self):
        self._make_vars([360.0, 0.0])
        self._assert_circular(True)

    def test_two_bounds_circular_alt(self):
        self._make_vars([-180.0, 180.0])
        self._assert_circular(True)

    def test_two_bounds_circular_alt_decreasing(self):
        self._make_vars([180.0, -180.0])
        self._assert_circular(True)

    def test_four_bounds(self):
        self._make_vars([0.0, 10.0, 20.0, 30.0])
        self._assert_circular(False)


if __name__ == "__main__":
    tests.main()

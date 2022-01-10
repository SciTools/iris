# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.netcdf._load_aux_factory` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock
import warnings

import numpy as np

from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import _load_aux_factory


class TestAtmosphereHybridSigmaPressureCoordinate(tests.IrisTest):
    def setUp(self):
        standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
        self.requires = dict(formula_type=standard_name)
        self.ap = mock.MagicMock(units="units")
        self.ps = mock.MagicMock(units="units")
        coordinates = [(mock.sentinel.b, "b"), (self.ps, "ps")]
        self.cube_parts = dict(coordinates=coordinates)
        self.engine = mock.Mock(
            requires=self.requires, cube_parts=self.cube_parts
        )
        self.cube = mock.create_autospec(Cube, spec_set=True, instance=True)
        # Patch out the check_dependencies functionality.
        func = "iris.aux_factory.HybridPressureFactory._check_dependencies"
        patcher = mock.patch(func)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_formula_terms_ap(self):
        self.cube_parts["coordinates"].append((self.ap, "ap"))
        self.requires["formula_terms"] = dict(ap="ap", b="b", ps="ps")
        _load_aux_factory(self.engine, self.cube)
        # Check cube.add_aux_coord method.
        self.assertEqual(self.cube.add_aux_coord.call_count, 0)
        # Check cube.add_aux_factory method.
        self.assertEqual(self.cube.add_aux_factory.call_count, 1)
        args, _ = self.cube.add_aux_factory.call_args
        self.assertEqual(len(args), 1)
        factory = args[0]
        self.assertEqual(factory.delta, self.ap)
        self.assertEqual(factory.sigma, mock.sentinel.b)
        self.assertEqual(factory.surface_air_pressure, self.ps)

    def test_formula_terms_a_p0(self):
        coord_a = DimCoord(np.arange(5), units="1")
        coord_p0 = DimCoord(10, units="Pa")
        coord_expected = DimCoord(
            np.arange(5) * 10,
            units="Pa",
            long_name="vertical pressure",
            var_name="ap",
        )
        self.cube_parts["coordinates"].extend(
            [(coord_a, "a"), (coord_p0, "p0")]
        )
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps", p0="p0")
        _load_aux_factory(self.engine, self.cube)
        # Check cube.coord_dims method.
        self.assertEqual(self.cube.coord_dims.call_count, 1)
        args, _ = self.cube.coord_dims.call_args
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], coord_a)
        # Check cube.add_aux_coord method.
        self.assertEqual(self.cube.add_aux_coord.call_count, 1)
        args, _ = self.cube.add_aux_coord.call_args
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], coord_expected)
        self.assertIsInstance(args[1], mock.Mock)
        # Check cube.add_aux_factory method.
        self.assertEqual(self.cube.add_aux_factory.call_count, 1)
        args, _ = self.cube.add_aux_factory.call_args
        self.assertEqual(len(args), 1)
        factory = args[0]
        self.assertEqual(factory.delta, coord_expected)
        self.assertEqual(factory.sigma, mock.sentinel.b)
        self.assertEqual(factory.surface_air_pressure, self.ps)

    def test_formula_terms_a_p0__promote_a_units_unknown_to_dimensionless(
        self,
    ):
        coord_a = DimCoord(np.arange(5), units="unknown")
        coord_p0 = DimCoord(10, units="Pa")
        coord_expected = DimCoord(
            np.arange(5) * 10,
            units="Pa",
            long_name="vertical pressure",
            var_name="ap",
        )
        self.cube_parts["coordinates"].extend(
            [(coord_a, "a"), (coord_p0, "p0")]
        )
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps", p0="p0")
        _load_aux_factory(self.engine, self.cube)
        # Check cube.coord_dims method.
        self.assertEqual(self.cube.coord_dims.call_count, 1)
        args, _ = self.cube.coord_dims.call_args
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], coord_a)
        self.assertEqual("1", args[0].units)
        # Check cube.add_aux_coord method.
        self.assertEqual(self.cube.add_aux_coord.call_count, 1)
        args, _ = self.cube.add_aux_coord.call_args
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], coord_expected)
        self.assertIsInstance(args[1], mock.Mock)
        # Check cube.add_aux_factory method.
        self.assertEqual(self.cube.add_aux_factory.call_count, 1)
        args, _ = self.cube.add_aux_factory.call_args
        self.assertEqual(len(args), 1)
        factory = args[0]
        self.assertEqual(factory.delta, coord_expected)
        self.assertEqual(factory.sigma, mock.sentinel.b)
        self.assertEqual(factory.surface_air_pressure, self.ps)

    def test_formula_terms_p0_non_scalar(self):
        coord_p0 = DimCoord(np.arange(5))
        self.cube_parts["coordinates"].append((coord_p0, "p0"))
        self.requires["formula_terms"] = dict(p0="p0")
        with self.assertRaises(ValueError):
            _load_aux_factory(self.engine, self.cube)

    def test_formula_terms_p0_bounded(self):
        coord_a = DimCoord(np.arange(5))
        coord_p0 = DimCoord(1, bounds=[0, 2], var_name="p0")
        self.cube_parts["coordinates"].extend(
            [(coord_a, "a"), (coord_p0, "p0")]
        )
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps", p0="p0")
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            _load_aux_factory(self.engine, self.cube)
            self.assertEqual(len(warn), 1)
            msg = (
                "Ignoring atmosphere hybrid sigma pressure scalar "
                "coordinate {!r} bounds.".format(coord_p0.name())
            )
            self.assertEqual(msg, str(warn[0].message))

    def _check_no_delta(self):
        # Check cube.add_aux_coord method.
        self.assertEqual(self.cube.add_aux_coord.call_count, 0)
        # Check cube.add_aux_factory method.
        self.assertEqual(self.cube.add_aux_factory.call_count, 1)
        args, _ = self.cube.add_aux_factory.call_args
        self.assertEqual(len(args), 1)
        factory = args[0]
        # Check that the factory has no delta term
        self.assertEqual(factory.delta, None)
        self.assertEqual(factory.sigma, mock.sentinel.b)
        self.assertEqual(factory.surface_air_pressure, self.ps)

    def test_formula_terms_ap_missing_coords(self):
        self.requires["formula_terms"] = dict(ap="ap", b="b", ps="ps")
        with mock.patch("warnings.warn") as warn:
            _load_aux_factory(self.engine, self.cube)
        warn.assert_called_once_with(
            "Unable to find coordinate for variable " "'ap'"
        )
        self._check_no_delta()

    def test_formula_terms_no_delta_terms(self):
        self.requires["formula_terms"] = dict(b="b", ps="ps")
        _load_aux_factory(self.engine, self.cube)
        self._check_no_delta()

    def test_formula_terms_no_p0_term(self):
        coord_a = DimCoord(np.arange(5), units="Pa")
        self.cube_parts["coordinates"].append((coord_a, "a"))
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps")
        _load_aux_factory(self.engine, self.cube)
        self._check_no_delta()

    def test_formula_terms_no_a_term(self):
        coord_p0 = DimCoord(10, units="1")
        self.cube_parts["coordinates"].append((coord_p0, "p0"))
        self.requires["formula_terms"] = dict(a="p0", b="b", ps="ps")
        _load_aux_factory(self.engine, self.cube)
        self._check_no_delta()


if __name__ == "__main__":
    tests.main()

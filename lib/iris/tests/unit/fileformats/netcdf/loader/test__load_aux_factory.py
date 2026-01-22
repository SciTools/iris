# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.netcdf._load_aux_factory` function."""

import warnings

import numpy as np
import pytest

from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf.loader import _load_aux_factory
from iris.tests.unit.fileformats import MockerMixin
from iris.warnings import IrisFactoryCoordNotFoundWarning


class TestAtmosphereHybridSigmaPressureCoordinate(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
        self.requires = dict(formula_type=standard_name)
        self.ap = mocker.MagicMock(units="units")
        self.ps = mocker.MagicMock(units="units")
        coordinates = [(mocker.sentinel.b, "b"), (self.ps, "ps")]
        self.cube_parts = dict(coordinates=coordinates)
        self.engine = mocker.Mock(requires=self.requires, cube_parts=self.cube_parts)
        self.cube = mocker.create_autospec(Cube, spec_set=True, instance=True)
        # Patch out the check_dependencies functionality.
        func = "iris.aux_factory.HybridPressureFactory._check_dependencies"
        _ = mocker.patch(func)

    def test_formula_terms_ap(self, mocker):
        self.cube_parts["coordinates"].append((self.ap, "ap"))
        self.requires["formula_terms"] = dict(ap="ap", b="b", ps="ps")
        _load_aux_factory(self.engine, self.cube)
        # Check cube.add_aux_coord method.
        assert self.cube.add_aux_coord.call_count == 0
        # Check cube.add_aux_factory method.
        assert self.cube.add_aux_factory.call_count == 1
        args, _ = self.cube.add_aux_factory.call_args
        assert len(args) == 1
        factory = args[0]
        assert factory.delta == self.ap
        assert factory.sigma == mocker.sentinel.b
        assert factory.surface_air_pressure == self.ps

    def test_formula_terms_a_p0(self, mocker):
        coord_a = DimCoord(np.arange(5), units="1")
        coord_p0 = DimCoord(10, units="Pa")
        coord_expected = DimCoord(
            np.arange(5) * 10,
            units="Pa",
            long_name="vertical pressure",
            var_name="ap",
        )
        self.cube_parts["coordinates"].extend([(coord_a, "a"), (coord_p0, "p0")])
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps", p0="p0")
        _load_aux_factory(self.engine, self.cube)
        # Check cube.coord_dims method.
        assert self.cube.coord_dims.call_count == 1
        args, _ = self.cube.coord_dims.call_args
        assert len(args) == 1
        assert args[0] is coord_a
        # Check cube.add_aux_coord method.
        assert self.cube.add_aux_coord.call_count == 1
        args, _ = self.cube.add_aux_coord.call_args
        assert len(args) == 2
        assert args[0] == coord_expected
        assert isinstance(args[1], mocker.Mock)
        # Check cube.add_aux_factory method.
        assert self.cube.add_aux_factory.call_count == 1
        args, _ = self.cube.add_aux_factory.call_args
        assert len(args) == 1
        factory = args[0]
        assert factory.delta == coord_expected
        assert factory.sigma == mocker.sentinel.b
        assert factory.surface_air_pressure == self.ps

    def test_formula_terms_a_p0__promote_a_units_unknown_to_dimensionless(self, mocker):
        coord_a = DimCoord(np.arange(5), units="unknown")
        coord_p0 = DimCoord(10, units="Pa")
        coord_expected = DimCoord(
            np.arange(5) * 10,
            units="Pa",
            long_name="vertical pressure",
            var_name="ap",
        )
        self.cube_parts["coordinates"].extend([(coord_a, "a"), (coord_p0, "p0")])
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps", p0="p0")
        _load_aux_factory(self.engine, self.cube)
        # Check cube.coord_dims method.
        assert self.cube.coord_dims.call_count == 1
        args, _ = self.cube.coord_dims.call_args
        assert len(args) == 1
        assert args[0] is coord_a
        assert "1" == args[0].units
        # Check cube.add_aux_coord method.
        assert self.cube.add_aux_coord.call_count == 1
        args, _ = self.cube.add_aux_coord.call_args
        assert len(args) == 2
        assert args[0] == coord_expected
        assert isinstance(args[1], mocker.Mock)
        # Check cube.add_aux_factory method.
        assert self.cube.add_aux_factory.call_count == 1
        args, _ = self.cube.add_aux_factory.call_args
        assert len(args) == 1
        factory = args[0]
        assert factory.delta == coord_expected
        assert factory.sigma == mocker.sentinel.b
        assert factory.surface_air_pressure == self.ps

    def test_formula_terms_p0_non_scalar(self):
        coord_p0 = DimCoord(np.arange(5))
        self.cube_parts["coordinates"].append((coord_p0, "p0"))
        self.requires["formula_terms"] = dict(p0="p0")
        with pytest.raises(ValueError):
            _load_aux_factory(self.engine, self.cube)

    def test_formula_terms_p0_bounded(self):
        coord_a = DimCoord(np.arange(5))
        coord_p0 = DimCoord(1, bounds=[0, 2], var_name="p0")
        self.cube_parts["coordinates"].extend([(coord_a, "a"), (coord_p0, "p0")])
        self.requires["formula_terms"] = dict(a="a", b="b", ps="ps", p0="p0")
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            _load_aux_factory(self.engine, self.cube)
            assert len(warn) == 1
            msg = (
                "Ignoring atmosphere hybrid sigma pressure scalar "
                "coordinate {!r} bounds.".format(coord_p0.name())
            )
            assert msg == str(warn[0].message)

    def _check_no_delta(self):
        # Check cube.add_aux_coord method.
        assert self.cube.add_aux_coord.call_count == 0
        # Check cube.add_aux_factory method.
        assert self.cube.add_aux_factory.call_count == 1
        args, _ = self.cube.add_aux_factory.call_args
        assert len(args) == 1
        factory = args[0]
        # Check that the factory has no delta term
        assert factory.delta == None
        assert factory.sigma == self.mocker.sentinel.b
        assert factory.surface_air_pressure == self.ps

    def test_formula_terms_ap_missing_coords(self):
        self.requires["formula_terms"] = dict(ap="ap", b="b", ps="ps")
        with pytest.warns(
            IrisFactoryCoordNotFoundWarning,
            match="Unable to find coordinate for variable 'ap'",
        ) as warn:
            _load_aux_factory(self.engine, self.cube)
        assert len(warn) == 1
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

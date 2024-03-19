# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.HybridPressureFactory` class.

"""

from unittest.mock import Mock

import cf_units
import numpy as np
import pytest

import iris
from iris.aux_factory import HybridPressureFactory


def create_default_sample_parts(self):
    self.delta = Mock(units=cf_units.Unit("Pa"), nbounds=0)
    self.sigma = Mock(units=cf_units.Unit("1"), nbounds=0)
    self.surface_air_pressure = Mock(units=cf_units.Unit("Pa"), nbounds=0)
    self.factory = HybridPressureFactory(
        delta=self.delta,
        sigma=self.sigma,
        surface_air_pressure=self.surface_air_pressure,
    )


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        create_default_sample_parts(self)

    def test_insufficient_coords(self):
        with pytest.raises(ValueError):
            HybridPressureFactory()
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=None, sigma=self.sigma, surface_air_pressure=None
            )
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=None,
                sigma=None,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_incompatible_delta_units(self):
        self.delta.units = cf_units.Unit("m")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=self.delta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_incompatible_sigma_units(self):
        self.sigma.units = cf_units.Unit("Pa")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=self.delta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_incompatible_surface_air_pressure_units(self):
        self.surface_air_pressure.units = cf_units.Unit("unknown")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=self.delta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_different_pressure_units(self):
        self.delta.units = cf_units.Unit("hPa")
        self.surface_air_pressure.units = cf_units.Unit("Pa")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=self.delta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_too_many_delta_bounds(self):
        self.delta.nbounds = 4
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=self.delta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_too_many_sigma_bounds(self):
        self.sigma.nbounds = 4
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=self.delta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_factory_metadata(self):
        factory = HybridPressureFactory(
            delta=self.delta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        assert factory.standard_name == "air_pressure"
        assert factory.long_name is None
        assert factory.var_name is None
        assert factory.units == self.delta.units
        assert factory.units == self.surface_air_pressure.units
        assert factory.coord_system is None
        assert factory.attributes == {}

    def test_promote_sigma_units_unknown_to_dimensionless(self):
        sigma = Mock(units=cf_units.Unit("unknown"), nbounds=0)
        factory = HybridPressureFactory(
            delta=self.delta,
            sigma=sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        assert factory.dependencies["sigma"].units == "1"


class Test_dependencies:
    @pytest.fixture(autouse=True)
    def _setup(self):
        create_default_sample_parts(self)

    def test_value(self):
        kwargs = dict(
            delta=self.delta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        factory = HybridPressureFactory(**kwargs)
        assert factory.dependencies == kwargs


class Test_make_coord:
    @staticmethod
    def coords_dims_func(coord):
        mapping = dict(level_pressure=(0,), sigma=(0,), surface_air_pressure=(1, 2))
        return mapping[coord.name()]

    @pytest.fixture(autouse=True)
    def _setup(self):
        # Create standard data objects for coord testing
        self.delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name="level_pressure", units="Pa"
        )
        self.sigma = iris.coords.DimCoord([1.0, 0.9, 0.8], long_name="sigma", units="1")
        self.surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), "surface_air_pressure", units="Pa"
        )

    def test_points_only(self):
        # Determine expected coord by manually broadcasting coord points
        # knowing the dimension mapping.
        delta_pts = self.delta.points[..., np.newaxis, np.newaxis]
        sigma_pts = self.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = self.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            delta=self.delta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord

    def test_none_delta(self):
        delta_pts = 0
        sigma_pts = self.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = self.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord

    def test_none_sigma(self):
        delta_pts = self.delta.points[..., np.newaxis, np.newaxis]
        sigma_pts = 0
        surf_pts = self.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            delta=self.delta,
            surface_air_pressure=self.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord

    def test_none_surface_air_pressure(self):
        # Note absence of broadcasting as multidimensional coord
        # is not present.
        expected_points = self.delta.points
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(delta=self.delta, sigma=self.sigma)
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord

    def test_with_bounds(self):
        self.delta.guess_bounds(0)
        self.sigma.guess_bounds(0.5)
        # Determine expected coord by manually broadcasting coord points
        # and bounds based on the dimension mapping.
        delta_pts = self.delta.points[..., np.newaxis, np.newaxis]
        sigma_pts = self.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = self.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        delta_vals = self.delta.bounds.reshape(3, 1, 1, 2)
        sigma_vals = self.sigma.bounds.reshape(3, 1, 1, 2)
        surf_vals = self.surface_air_pressure.points.reshape(1, 2, 2, 1)
        expected_bounds = delta_vals + sigma_vals * surf_vals
        expected_coord = iris.coords.AuxCoord(
            expected_points,
            standard_name="air_pressure",
            units="Pa",
            bounds=expected_bounds,
        )
        factory = HybridPressureFactory(
            delta=self.delta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self):
        create_default_sample_parts(self)

    def test_good_delta(self):
        new_delta_coord = Mock(units=cf_units.Unit("Pa"), nbounds=0)
        self.factory.update(self.delta, new_delta_coord)
        assert self.factory.delta is new_delta_coord

    def test_bad_delta(self):
        new_delta_coord = Mock(units=cf_units.Unit("1"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.delta, new_delta_coord)

    def test_alternative_bad_delta(self):
        new_delta_coord = Mock(units=cf_units.Unit("Pa"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.delta, new_delta_coord)

    def test_good_surface_air_pressure(self):
        new_surface_p_coord = Mock(units=cf_units.Unit("Pa"), nbounds=0)
        self.factory.update(self.surface_air_pressure, new_surface_p_coord)
        assert self.factory.surface_air_pressure is new_surface_p_coord

    def test_bad_surface_air_pressure(self):
        new_surface_p_coord = Mock(units=cf_units.Unit("km"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.surface_air_pressure, new_surface_p_coord)

    def test_non_dependency(self):
        old_coord = Mock()
        new_coord = Mock()
        orig_dependencies = self.factory.dependencies
        self.factory.update(old_coord, new_coord)
        assert self.factory.dependencies == orig_dependencies

    def test_none_delta(self):
        self.factory.update(self.delta, None)
        assert self.factory.delta is None

    def test_none_sigma(self):
        self.factory.update(self.sigma, None)
        assert self.factory.sigma is None

    def test_insufficient_coords(self):
        self.factory.update(self.delta, None)
        with pytest.raises(ValueError):
            self.factory.update(self.surface_air_pressure, None)

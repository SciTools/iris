# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.HybridPressureFactory` class.

"""
from dataclasses import dataclass

import cf_units
import numpy as np
import pytest

import iris
from iris.aux_factory import HybridPressureFactory


@dataclass
class SampleParts:
    delta = None
    sigma = None
    surface_air_pressure = None


@pytest.fixture()
def sample_parts(mocker):
    data = SampleParts()
    data.delta = mocker.Mock(units=cf_units.Unit("Pa"), nbounds=0)
    data.sigma = mocker.Mock(units=cf_units.Unit("1"), nbounds=0)
    data.surface_air_pressure = mocker.Mock(units=cf_units.Unit("Pa"), nbounds=0)
    return data


class Test___init__:
    def test_insufficient_coords(self, sample_parts):
        with pytest.raises(ValueError):
            HybridPressureFactory()
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=None, sigma=sample_parts.sigma, surface_air_pressure=None
            )
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=None,
                sigma=None,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_incompatible_delta_units(self, sample_parts):
        sample_parts.delta.units = cf_units.Unit("m")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=sample_parts.delta,
                sigma=sample_parts.sigma,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_incompatible_sigma_units(self, sample_parts):
        sample_parts.sigma.units = cf_units.Unit("Pa")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=sample_parts.delta,
                sigma=sample_parts.sigma,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_incompatible_surface_air_pressure_units(self, sample_parts):
        sample_parts.surface_air_pressure.units = cf_units.Unit("unknown")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=sample_parts.delta,
                sigma=sample_parts.sigma,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_different_pressure_units(self, sample_parts):
        sample_parts.delta.units = cf_units.Unit("hPa")
        sample_parts.surface_air_pressure.units = cf_units.Unit("Pa")
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=sample_parts.delta,
                sigma=sample_parts.sigma,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_too_many_delta_bounds(self, sample_parts):
        sample_parts.delta.nbounds = 4
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=sample_parts.delta,
                sigma=sample_parts.sigma,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_too_many_sigma_bounds(self, sample_parts):
        sample_parts.sigma.nbounds = 4
        with pytest.raises(ValueError):
            HybridPressureFactory(
                delta=sample_parts.delta,
                sigma=sample_parts.sigma,
                surface_air_pressure=sample_parts.surface_air_pressure,
            )

    def test_factory_metadata(self, sample_parts):
        factory = HybridPressureFactory(
            delta=sample_parts.delta,
            sigma=sample_parts.sigma,
            surface_air_pressure=sample_parts.surface_air_pressure,
        )
        assert factory.standard_name == "air_pressure"
        assert factory.long_name is None
        assert factory.var_name is None
        assert factory.units == sample_parts.delta.units
        assert factory.units == sample_parts.surface_air_pressure.units
        assert factory.coord_system is None
        assert factory.attributes == {}

    def test_promote_sigma_units_unknown_to_dimensionless(self, sample_parts, mocker):
        sigma = mocker.Mock(units=cf_units.Unit("unknown"), nbounds=0)
        factory = HybridPressureFactory(
            delta=sample_parts.delta,
            sigma=sigma,
            surface_air_pressure=sample_parts.surface_air_pressure,
        )
        assert "1" == factory.dependencies["sigma"].units


class Test_dependencies:
    def test_value(self, sample_parts):
        kwargs = dict(
            delta=sample_parts.delta,
            sigma=sample_parts.sigma,
            surface_air_pressure=sample_parts.surface_air_pressure,
        )
        factory = HybridPressureFactory(**kwargs)
        assert factory.dependencies == kwargs


class Test_make_coord:
    @staticmethod
    def coords_dims_func(coord):
        mapping = dict(level_pressure=(0,), sigma=(0,), surface_air_pressure=(1, 2))
        return mapping[coord.name()]

    @pytest.fixture()
    def coord_parts(self):
        # A different standard 'SampleParts' makeup for coord testing
        parts = SampleParts()
        parts.delta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name="level_pressure", units="Pa"
        )
        parts.sigma = iris.coords.DimCoord(
            [1.0, 0.9, 0.8], long_name="sigma", units="1"
        )
        parts.surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), "surface_air_pressure", units="Pa"
        )
        return parts

    def test_points_only(self, coord_parts):
        # Determine expected coord by manually broadcasting coord points
        # knowing the dimension mapping.
        delta_pts = coord_parts.delta.points[..., np.newaxis, np.newaxis]
        sigma_pts = coord_parts.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = coord_parts.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            delta=coord_parts.delta,
            sigma=coord_parts.sigma,
            surface_air_pressure=coord_parts.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert expected_coord == derived_coord

    def test_none_delta(self, coord_parts):
        delta_pts = 0
        sigma_pts = coord_parts.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = coord_parts.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            sigma=coord_parts.sigma,
            surface_air_pressure=coord_parts.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert expected_coord == derived_coord

    def test_none_sigma(self, coord_parts):
        delta_pts = coord_parts.delta.points[..., np.newaxis, np.newaxis]
        sigma_pts = 0
        surf_pts = coord_parts.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            delta=coord_parts.delta,
            surface_air_pressure=coord_parts.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert expected_coord == derived_coord

    def test_none_surface_air_pressure(self, coord_parts):
        # Note absence of broadcasting as multidimensional coord
        # is not present.
        expected_points = coord_parts.delta.points
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridPressureFactory(
            delta=coord_parts.delta, sigma=coord_parts.sigma
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert expected_coord == derived_coord

    def test_with_bounds(self, coord_parts):
        coord_parts.delta.guess_bounds(0)
        coord_parts.sigma.guess_bounds(0.5)
        # Determine expected coord by manually broadcasting coord points
        # and bounds based on the dimension mapping.
        delta_pts = coord_parts.delta.points[..., np.newaxis, np.newaxis]
        sigma_pts = coord_parts.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = coord_parts.surface_air_pressure.points[np.newaxis, ...]
        expected_points = delta_pts + sigma_pts * surf_pts
        delta_vals = coord_parts.delta.bounds.reshape(3, 1, 1, 2)
        sigma_vals = coord_parts.sigma.bounds.reshape(3, 1, 1, 2)
        surf_vals = coord_parts.surface_air_pressure.points.reshape(1, 2, 2, 1)
        expected_bounds = delta_vals + sigma_vals * surf_vals
        expected_coord = iris.coords.AuxCoord(
            expected_points,
            standard_name="air_pressure",
            units="Pa",
            bounds=expected_bounds,
        )
        factory = HybridPressureFactory(
            delta=coord_parts.delta,
            sigma=coord_parts.sigma,
            surface_air_pressure=coord_parts.surface_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert expected_coord == derived_coord


class Test_update:
    @pytest.fixture()
    def update_parts(self, sample_parts):
        sample_parts.factory = HybridPressureFactory(
            delta=sample_parts.delta,
            sigma=sample_parts.sigma,
            surface_air_pressure=sample_parts.surface_air_pressure,
        )
        return sample_parts

    def test_good_delta(self, update_parts, mocker):
        new_delta_coord = mocker.Mock(units=cf_units.Unit("Pa"), nbounds=0)
        update_parts.factory.update(update_parts.delta, new_delta_coord)
        assert update_parts.factory.delta is new_delta_coord

    def test_bad_delta(self, update_parts, mocker):
        new_delta_coord = mocker.Mock(units=cf_units.Unit("1"), nbounds=0)
        with pytest.raises(ValueError):
            update_parts.factory.update(update_parts.delta, new_delta_coord)

    def test_alternative_bad_delta(self, update_parts, mocker):
        new_delta_coord = mocker.Mock(units=cf_units.Unit("Pa"), nbounds=4)
        with pytest.raises(ValueError):
            update_parts.factory.update(update_parts.delta, new_delta_coord)

    def test_good_surface_air_pressure(self, update_parts, mocker):
        new_surface_p_coord = mocker.Mock(units=cf_units.Unit("Pa"), nbounds=0)
        update_parts.factory.update(
            update_parts.surface_air_pressure, new_surface_p_coord
        )
        assert update_parts.factory.surface_air_pressure is new_surface_p_coord

    def test_bad_surface_air_pressure(self, update_parts, mocker):
        new_surface_p_coord = mocker.Mock(units=cf_units.Unit("km"), nbounds=0)
        with pytest.raises(ValueError):
            update_parts.factory.update(
                update_parts.surface_air_pressure, new_surface_p_coord
            )

    def test_non_dependency(self, update_parts, mocker):
        old_coord = mocker.Mock()
        new_coord = mocker.Mock()
        orig_dependencies = update_parts.factory.dependencies
        update_parts.factory.update(old_coord, new_coord)
        assert orig_dependencies == update_parts.factory.dependencies

    def test_none_delta(self, update_parts):
        update_parts.factory.update(update_parts.delta, None)
        assert update_parts.factory.delta is None

    def test_none_sigma(self, update_parts):
        update_parts.factory.update(update_parts.sigma, None)
        assert update_parts.factory.sigma is None

    def test_insufficient_coords(self, update_parts):
        update_parts.factory.update(update_parts.delta, None)
        with pytest.raises(ValueError):
            update_parts.factory.update(update_parts.surface_air_pressure, None)

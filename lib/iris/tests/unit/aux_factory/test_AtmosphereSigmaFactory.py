# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.AtmosphereSigmaFactory` class.

"""

import re

from cf_units import Unit
import numpy as np
import pytest

from iris.aux_factory import AtmosphereSigmaFactory
from iris.coords import AuxCoord, DimCoord


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        self.sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        self.surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        self.kwargs = dict(
            pressure_at_top=self.pressure_at_top,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )

    def test_insufficient_coordinates_no_args(self):
        msg = "Unable to construct atmosphere sigma coordinate factory due to insufficient source coordinates"
        with pytest.raises(ValueError, match=msg):
            AtmosphereSigmaFactory()

    def test_insufficient_coordinates_no_ptop(self):
        msg = "Unable to construct atmosphere sigma coordinate factory due to insufficient source coordinates"
        with pytest.raises(ValueError, match=msg):
            AtmosphereSigmaFactory(
                pressure_at_top=None,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_insufficient_coordinates_no_sigma(self):
        msg = "Unable to construct atmosphere sigma coordinate factory due to insufficient source coordinates"
        with pytest.raises(ValueError, match=msg):
            AtmosphereSigmaFactory(
                pressure_at_top=self.pressure_at_top,
                sigma=None,
                surface_air_pressure=self.surface_air_pressure,
            )

    def test_insufficient_coordinates_no_ps(self):
        msg = "Unable to construct atmosphere sigma coordinate factory due to insufficient source coordinates"
        with pytest.raises(ValueError, match=msg):
            AtmosphereSigmaFactory(
                pressure_at_top=self.pressure_at_top,
                sigma=self.sigma,
                surface_air_pressure=None,
            )

    def test_ptop_shapes(self):
        for shape in [(), (1,)]:
            self.pressure_at_top.shape = shape
            AtmosphereSigmaFactory(**self.kwargs)

    def test_ptop_invalid_shapes(self):
        for shape in [(2,), (1, 1)]:
            msg = re.escape(
                f"Expected scalar 'pressure_at_top' coordinate, got shape {shape}"
            )
            self.pressure_at_top.shape = shape
            with pytest.raises(ValueError, match=msg):
                AtmosphereSigmaFactory(**self.kwargs)

    def test_sigma_bounds(self):
        for n_bounds in [0, 2]:
            self.sigma.nbounds = n_bounds
            AtmosphereSigmaFactory(**self.kwargs)

    def test_sigma_invalid_bounds(self):
        for n_bounds in [-1, 1, 3]:
            msg = f"Invalid 'sigma' coordinate: must have either 0 or 2 bounds, got {n_bounds}"
            self.sigma.nbounds = n_bounds
            with pytest.raises(ValueError, match=msg):
                AtmosphereSigmaFactory(**self.kwargs)

    def test_sigma_units(self):
        for units in ["1", "unknown", None]:
            self.sigma.units = Unit(units)
            AtmosphereSigmaFactory(**self.kwargs)

    def test_sigma_invalid_units(self):
        for units in ["Pa", "m"]:
            self.sigma.units = Unit(units)
            msg = f"Invalid units: 'sigma' must be dimensionless, got '{units}'"
            with pytest.raises(ValueError, match=msg):
                AtmosphereSigmaFactory(**self.kwargs)

    def test_ptop_ps_units(self):
        for units in [("Pa", "Pa")]:
            self.pressure_at_top.units = Unit(units[0])
            self.surface_air_pressure.units = Unit(units[1])
            AtmosphereSigmaFactory(**self.kwargs)

    def test_ptop_ps_invalid_units(self):
        for units in [("Pa", "1"), ("1", "Pa"), ("bar", "Pa"), ("Pa", "hPa")]:
            msg = f"Incompatible units: 'pressure_at_top' and 'surface_air_pressure' must have the same units, got '{units[0]}' and '{units[1]}'"
            self.pressure_at_top.units = Unit(units[0])
            self.surface_air_pressure.units = Unit(units[1])
            with pytest.raises(ValueError, match=msg):
                AtmosphereSigmaFactory(**self.kwargs)

    def test_ptop_units(self):
        for units in ["Pa", "bar", "mbar", "hPa"]:
            self.pressure_at_top.units = Unit(units)
            self.surface_air_pressure.units = Unit(units)
            AtmosphereSigmaFactory(**self.kwargs)

    def test_ptop_invalid_units(self):
        for units in ["1", "m", "kg", None]:
            msg = "Invalid units: 'pressure_at_top' and 'surface_air_pressure' must have units of pressure"
            self.pressure_at_top.units = Unit(units)
            self.surface_air_pressure.units = Unit(units)
            with pytest.raises(ValueError, match=msg):
                AtmosphereSigmaFactory(**self.kwargs)


class Test_dependencies:
    @pytest.fixture
    def sample_kwargs(self, mocker):
        pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        kwargs = dict(
            pressure_at_top=pressure_at_top,
            sigma=sigma,
            surface_air_pressure=surface_air_pressure,
        )
        return kwargs

    def test_values(self, sample_kwargs):
        factory = AtmosphereSigmaFactory(**sample_kwargs)
        assert factory.dependencies == sample_kwargs


class Test__derive:
    def test_function_scalar(self):
        assert AtmosphereSigmaFactory._calculate_array(0, 0, 0) == 0
        assert AtmosphereSigmaFactory._calculate_array(3, 0, 0) == 3
        assert AtmosphereSigmaFactory._calculate_array(0, 5, 0) == 0
        assert AtmosphereSigmaFactory._calculate_array(0, 0, 7) == 0
        assert AtmosphereSigmaFactory._calculate_array(3, 5, 0) == -12
        assert AtmosphereSigmaFactory._calculate_array(3, 0, 7) == 3
        assert AtmosphereSigmaFactory._calculate_array(0, 5, 7) == 35
        assert AtmosphereSigmaFactory._calculate_array(3, 5, 7) == 23

    def test_function_array(self):
        ptop = 3
        sigma = np.array([2, 4])
        ps = np.arange(4).reshape(2, 2)
        np.testing.assert_equal(
            AtmosphereSigmaFactory._calculate_array(ptop, sigma, ps),
            [[-3, -5], [1, 3]],
        )


class Test_make_coord:
    @staticmethod
    def coord_dims(coord):
        mapping = dict(
            pressure_at_top=(),
            sigma=(0,),
            surface_air_pressure=(1, 2),
        )
        return mapping[coord.name()]

    @staticmethod
    def derive(pressure_at_top, sigma, surface_air_pressure, coord=True):
        result = pressure_at_top + sigma * (surface_air_pressure - pressure_at_top)
        if coord:
            name = "air_pressure"
            result = AuxCoord(
                result,
                standard_name=name,
                units="Pa",
            )
        return result

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pressure_at_top = AuxCoord(
            [3.0],
            long_name="pressure_at_top",
            units="Pa",
        )
        self.sigma = DimCoord(
            [1.0, 0.4, 0.1],
            bounds=[[1.0, 0.6], [0.6, 0.2], [0.2, 0.0]],
            long_name="sigma",
            units="1",
        )
        self.surface_air_pressure = AuxCoord(
            [[-1.0, 2.0], [1.0, 3.0]],
            long_name="surface_air_pressure",
            units="Pa",
        )
        self.kwargs = dict(
            pressure_at_top=self.pressure_at_top,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )

    def test_derived_coord(self):
        # Broadcast expected points given the known dimensional mapping
        pressure_at_top = self.pressure_at_top.points[0]
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        surface_air_pressure = self.surface_air_pressure.points[np.newaxis, ...]

        # Calculate the expected result

        expected_coord = self.derive(pressure_at_top, sigma, surface_air_pressure)

        # Calculate the actual result
        factory = AtmosphereSigmaFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)

        # Check bounds
        expected_bounds = [
            [[[-1.0, 0.6], [2.0, 2.4]], [[1.0, 1.8], [3.0, 3.0]]],
            [[[0.6, 2.2], [2.4, 2.8]], [[1.8, 2.6], [3.0, 3.0]]],
            [[[2.2, 3.0], [2.8, 3.0]], [[2.6, 3.0], [3.0, 3.0]]],
        ]
        np.testing.assert_allclose(coord.bounds, expected_bounds)
        coord.bounds = None

        # Check points and metadata
        assert coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        self.sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        self.surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        self.kwargs = dict(
            pressure_at_top=self.pressure_at_top,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
        )
        self.factory = AtmosphereSigmaFactory(**self.kwargs)

    def test_pressure_at_top(self, mocker):
        new_pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        self.factory.update(self.pressure_at_top, new_pressure_at_top)
        assert self.factory.pressure_at_top is new_pressure_at_top

    def test_pressure_at_top_wrong_shape(self, mocker):
        new_pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=(2,))
        msg = re.escape(
            "Failed to update dependencies. Expected scalar 'pressure_at_top' coordinate, got shape (2,)"
        )
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.pressure_at_top, new_pressure_at_top)

    def test_sigma(self, mocker):
        new_sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        self.factory.update(self.sigma, new_sigma)
        assert self.factory.sigma is new_sigma

    def test_sigma_too_many_bounds(self, mocker):
        new_sigma = mocker.Mock(units=Unit("1"), nbounds=4)
        msg = "Failed to update dependencies. Invalid 'sigma' coordinate: must have either 0 or 2 bounds, got 4"
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.sigma, new_sigma)

    def test_sigma_incompatible_units(self, mocker):
        new_sigma = mocker.Mock(units=Unit("Pa"), nbounds=0)
        msg = "Failed to update dependencies. Invalid units: 'sigma' must be dimensionless, got 'Pa'"
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.sigma, new_sigma)

    def test_surface_air_pressure(self, mocker):
        new_surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        self.factory.update(self.surface_air_pressure, new_surface_air_pressure)
        assert self.factory.surface_air_pressure is new_surface_air_pressure

    def test_surface_air_pressure_incompatible_units(self, mocker):
        new_surface_air_pressure = mocker.Mock(units=Unit("mbar"), nbounds=0)
        msg = "Failed to update dependencies. Incompatible units: 'pressure_at_top' and 'surface_air_pressure' must have the same units, got 'Pa' and 'mbar'"
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.surface_air_pressure, new_surface_air_pressure)

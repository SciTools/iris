# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.AtmosphereSigmaFactory` class.

"""
from dataclasses import dataclass

from cf_units import Unit
import numpy as np
import pytest

from iris.aux_factory import AtmosphereSigmaFactory
from iris.coords import AuxCoord, DimCoord


class Test___init__:
    @pytest.fixture()
    def sample_args(self, mocker):
        @dataclass
        class SampleParts:
            pressure_at_top = None
            sigma = None
            surface_air_pressure = None
            kwargs = None

        sample = SampleParts()
        sample.pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        sample.sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        sample.surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        sample.kwargs = dict(
            pressure_at_top=sample.pressure_at_top,
            sigma=sample.sigma,
            surface_air_pressure=sample.surface_air_pressure,
        )
        return sample

    def test_insufficient_coordinates_no_args(self):
        with pytest.raises(ValueError):
            AtmosphereSigmaFactory()

    def test_insufficient_coordinates_no_ptop(self, sample_args):
        with pytest.raises(ValueError):
            AtmosphereSigmaFactory(
                pressure_at_top=None,
                sigma=sample_args.sigma,
                surface_air_pressure=sample_args.surface_air_pressure,
            )

    def test_insufficient_coordinates_no_sigma(self, sample_args):
        with pytest.raises(ValueError):
            AtmosphereSigmaFactory(
                pressure_at_top=sample_args.pressure_at_top,
                sigma=None,
                surface_air_pressure=sample_args.surface_air_pressure,
            )

    def test_insufficient_coordinates_no_ps(self, sample_args):
        with pytest.raises(ValueError):
            AtmosphereSigmaFactory(
                pressure_at_top=sample_args.pressure_at_top,
                sigma=sample_args.sigma,
                surface_air_pressure=None,
            )

    def test_ptop_shapes(self, sample_args):
        for shape in [(), (1,)]:
            sample_args.pressure_at_top.shape = shape
            AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_ptop_invalid_shapes(self, sample_args):
        for shape in [(2,), (1, 1)]:
            sample_args.pressure_at_top.shape = shape
            with pytest.raises(ValueError):
                AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_sigma_bounds(self, sample_args):
        for n_bounds in [0, 2]:
            sample_args.sigma.nbounds = n_bounds
            AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_sigma_invalid_bounds(self, sample_args):
        for n_bounds in [-1, 1, 3]:
            sample_args.sigma.nbounds = n_bounds
            with pytest.raises(ValueError):
                AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_sigma_units(self, sample_args):
        for units in ["1", "unknown", None]:
            sample_args.sigma.units = Unit(units)
            AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_sigma_invalid_units(self, sample_args):
        for units in ["Pa", "m"]:
            sample_args.sigma.units = Unit(units)
            with pytest.raises(ValueError):
                AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_ptop_ps_units(self, sample_args):
        for units in [("Pa", "Pa")]:
            sample_args.pressure_at_top.units = Unit(units[0])
            sample_args.surface_air_pressure.units = Unit(units[1])
            AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_ptop_ps_invalid_units(self, sample_args):
        for units in [("Pa", "1"), ("1", "Pa"), ("bar", "Pa"), ("Pa", "hPa")]:
            sample_args.pressure_at_top.units = Unit(units[0])
            sample_args.surface_air_pressure.units = Unit(units[1])
            with pytest.raises(ValueError):
                AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_ptop_units(self, sample_args):
        for units in ["Pa", "bar", "mbar", "hPa"]:
            sample_args.pressure_at_top.units = Unit(units)
            sample_args.surface_air_pressure.units = Unit(units)
            AtmosphereSigmaFactory(**sample_args.kwargs)

    def test_ptop_invalid_units(self, sample_args):
        for units in ["1", "m", "kg", None]:
            sample_args.pressure_at_top.units = Unit(units)
            sample_args.surface_air_pressure.units = Unit(units)
            with pytest.raises(ValueError):
                AtmosphereSigmaFactory(**sample_args.kwargs)


class Test_dependencies:
    @pytest.fixture()
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
        assert AtmosphereSigmaFactory._derive(0, 0, 0) == 0
        assert AtmosphereSigmaFactory._derive(3, 0, 0) == 3
        assert AtmosphereSigmaFactory._derive(0, 5, 0) == 0
        assert AtmosphereSigmaFactory._derive(0, 0, 7) == 0
        assert AtmosphereSigmaFactory._derive(3, 5, 0) == -12
        assert AtmosphereSigmaFactory._derive(3, 0, 7) == 3
        assert AtmosphereSigmaFactory._derive(0, 5, 7) == 35
        assert AtmosphereSigmaFactory._derive(3, 5, 7) == 23

    def test_function_array(self):
        ptop = 3
        sigma = np.array([2, 4])
        ps = np.arange(4).reshape(2, 2)
        np.testing.assert_equal(
            AtmosphereSigmaFactory._derive(ptop, sigma, ps),
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

    @pytest.fixture()
    def sample_parts(self):
        @dataclass
        class SampleParts:
            pressure_at_top = None
            sigma = None
            surface_air_pressure = None
            kwargs = None

        parts = SampleParts()
        parts.pressure_at_top = AuxCoord(
            [3.0],
            long_name="pressure_at_top",
            units="Pa",
        )
        parts.sigma = DimCoord(
            [1.0, 0.4, 0.1],
            bounds=[[1.0, 0.6], [0.6, 0.2], [0.2, 0.0]],
            long_name="sigma",
            units="1",
        )
        parts.surface_air_pressure = AuxCoord(
            [[-1.0, 2.0], [1.0, 3.0]],
            long_name="surface_air_pressure",
            units="Pa",
        )
        parts.kwargs = dict(
            pressure_at_top=parts.pressure_at_top,
            sigma=parts.sigma,
            surface_air_pressure=parts.surface_air_pressure,
        )
        return parts

    def test_derived_coord(self, sample_parts):
        # Broadcast expected points given the known dimensional mapping
        pressure_at_top = sample_parts.pressure_at_top.points[0]
        sigma = sample_parts.sigma.points[..., np.newaxis, np.newaxis]
        surface_air_pressure = sample_parts.surface_air_pressure.points[np.newaxis, ...]

        # Calculate the expected result

        expected_coord = self.derive(pressure_at_top, sigma, surface_air_pressure)

        # Calculate the actual result
        factory = AtmosphereSigmaFactory(**sample_parts.kwargs)
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
    @pytest.fixture()
    def sample_data(self, mocker):
        @dataclass
        class SampleData:
            pressure_at_top = None
            sigma = None
            surface_air_pressure = None
            kwargs = None
            factory = None

        data = SampleData()
        data.pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        data.sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        data.surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        data.kwargs = dict(
            pressure_at_top=data.pressure_at_top,
            sigma=data.sigma,
            surface_air_pressure=data.surface_air_pressure,
        )
        data.factory = AtmosphereSigmaFactory(**data.kwargs)
        return data

    def test_pressure_at_top(self, sample_data, mocker):
        new_pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=())
        sample_data.factory.update(sample_data.pressure_at_top, new_pressure_at_top)
        assert sample_data.factory.pressure_at_top is new_pressure_at_top

    def test_pressure_at_top_wrong_shape(self, sample_data, mocker):
        new_pressure_at_top = mocker.Mock(units=Unit("Pa"), nbounds=0, shape=(2,))
        with pytest.raises(ValueError):
            sample_data.factory.update(sample_data.pressure_at_top, new_pressure_at_top)

    def test_sigma(self, sample_data, mocker):
        new_sigma = mocker.Mock(units=Unit("1"), nbounds=0)
        sample_data.factory.update(sample_data.sigma, new_sigma)
        assert sample_data.factory.sigma is new_sigma

    def test_sigma_too_many_bounds(self, sample_data, mocker):
        new_sigma = mocker.Mock(units=Unit("1"), nbounds=4)
        with pytest.raises(ValueError):
            sample_data.factory.update(sample_data.sigma, new_sigma)

    def test_sigma_incompatible_units(self, sample_data, mocker):
        new_sigma = mocker.Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            sample_data.factory.update(sample_data.sigma, new_sigma)

    def test_surface_air_pressure(self, sample_data, mocker):
        new_surface_air_pressure = mocker.Mock(units=Unit("Pa"), nbounds=0)
        sample_data.factory.update(
            sample_data.surface_air_pressure, new_surface_air_pressure
        )
        assert sample_data.factory.surface_air_pressure is new_surface_air_pressure

    def test_surface_air_pressure_incompatible_units(self, sample_data, mocker):
        new_surface_air_pressure = mocker.Mock(units=Unit("mbar"), nbounds=0)
        with pytest.raises(ValueError):
            sample_data.factory.update(
                sample_data.surface_air_pressure, new_surface_air_pressure
            )

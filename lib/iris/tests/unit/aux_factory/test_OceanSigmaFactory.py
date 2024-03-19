# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.OceanSigmaFactory` class.

"""

from unittest.mock import Mock

from cf_units import Unit
import numpy as np
import pytest

from iris.aux_factory import OceanSigmaFactory
from iris.coords import AuxCoord, DimCoord


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def test_insufficient_coordinates(self):
        with pytest.raises(ValueError):
            OceanSigmaFactory()
        with pytest.raises(ValueError):
            OceanSigmaFactory(sigma=None, eta=self.eta, depth=self.depth)
        with pytest.raises(ValueError):
            OceanSigmaFactory(sigma=self.sigma, eta=None, depth=self.depth)
        with pytest.raises(ValueError):
            OceanSigmaFactory(sigma=self.sigma, eta=self.eta, depth=None)

    def test_sigma_too_many_bounds(self):
        self.sigma.nbounds = 4
        with pytest.raises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_sigma_incompatible_units(self):
        self.sigma.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_eta_incompatible_units(self):
        self.eta.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_depth_incompatible_units(self):
        self.depth.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaFactory(**self.kwargs)

    def test_promote_sigma_units_unknown_to_dimensionless(self):
        sigma = Mock(units=Unit("unknown"), nbounds=0)
        self.kwargs["sigma"] = sigma
        factory = OceanSigmaFactory(**self.kwargs)
        assert factory.dependencies["sigma"].units == "1"


class Test_dependencies:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def test_values(self):
        factory = OceanSigmaFactory(**self.kwargs)
        assert factory.dependencies == self.kwargs


class Test_make_coord:
    @staticmethod
    def coord_dims(coord):
        mapping = dict(sigma=(0,), eta=(1, 2), depth=(1, 2))
        return mapping[coord.name()]

    @staticmethod
    def derive(sigma, eta, depth, coord=True):
        result = eta + sigma * (depth + eta)
        if coord:
            name = "sea_surface_height_above_reference_ellipsoid"
            result = AuxCoord(
                result,
                standard_name=name,
                units="m",
                attributes=dict(positive="up"),
            )
        return result

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = DimCoord(np.linspace(-0.05, -1, 5), long_name="sigma", units="1")
        self.eta = AuxCoord(
            np.arange(-1, 3, dtype=np.float64).reshape(2, 2),
            long_name="eta",
            units="m",
        )
        self.depth = AuxCoord(
            np.arange(4, dtype=np.float64).reshape(2, 2) * 1e3,
            long_name="depth",
            units="m",
        )
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def test_derived_points(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth)
        # Calculate the actual result.
        factory = OceanSigmaFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.kwargs = dict(sigma=self.sigma, eta=self.eta, depth=self.depth)
        self.factory = OceanSigmaFactory(**self.kwargs)

    def test_sigma(self):
        new_sigma = Mock(units=Unit("1"), nbounds=0)
        self.factory.update(self.sigma, new_sigma)
        assert self.factory.sigma is new_sigma

    def test_sigma_too_many_bounds(self):
        new_sigma = Mock(units=Unit("1"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.sigma, new_sigma)

    def test_sigma_incompatible_units(self):
        new_sigma = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.sigma, new_sigma)

    def test_eta(self):
        new_eta = Mock(units=Unit("m"), nbounds=0)
        self.factory.update(self.eta, new_eta)
        assert self.factory.eta is new_eta

    def test_eta_incompatible_units(self):
        new_eta = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.eta, new_eta)

    def test_depth(self):
        new_depth = Mock(units=Unit("m"), nbounds=0)
        self.factory.update(self.depth, new_depth)
        assert self.factory.depth is new_depth

    def test_depth_incompatible_units(self):
        new_depth = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.depth, new_depth)

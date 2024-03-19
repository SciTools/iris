# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.OceanSg1Factory` class.

"""

from unittest.mock import Mock

from cf_units import Unit
import numpy as np
import pytest

from iris.aux_factory import OceanSg1Factory
from iris.coords import AuxCoord, DimCoord


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.s = Mock(units=Unit("1"), nbounds=0)
        self.c = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.kwargs = dict(
            s=self.s,
            c=self.c,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
        )

    def test_insufficient_coordinates(self):
        with pytest.raises(ValueError):
            OceanSg1Factory()
        with pytest.raises(ValueError):
            OceanSg1Factory(
                s=None,
                c=self.c,
                eta=self.eta,
                depth=self.depth,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSg1Factory(
                s=self.s,
                c=None,
                eta=self.eta,
                depth=self.depth,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSg1Factory(
                s=self.s,
                c=self.c,
                eta=None,
                depth=self.depth,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSg1Factory(
                s=self.s,
                c=self.c,
                eta=self.eta,
                depth=None,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSg1Factory(
                s=self.s,
                c=self.c,
                eta=self.eta,
                depth=self.depth,
                depth_c=None,
            )

    def test_s_too_many_bounds(self):
        self.s.nbounds = 4
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_c_too_many_bounds(self):
        self.c.nbounds = 4
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_depth_c_non_scalar(self):
        self.depth_c.shape = (2,)
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_s_incompatible_units(self):
        self.s.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_c_incompatible_units(self):
        self.c.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_eta_incompatible_units(self):
        self.eta.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_depth_c_incompatible_units(self):
        self.depth_c.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_depth_incompatible_units(self):
        self.depth.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSg1Factory(**self.kwargs)

    def test_promote_c_and_s_units_unknown_to_dimensionless(self):
        c = Mock(units=Unit("unknown"), nbounds=0)
        s = Mock(units=Unit("unknown"), nbounds=0)
        self.kwargs["c"] = c
        self.kwargs["s"] = s
        factory = OceanSg1Factory(**self.kwargs)
        assert factory.dependencies["c"].units == "1"
        assert factory.dependencies["s"].units == "1"


class Test_dependencies:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.s = Mock(units=Unit("1"), nbounds=0)
        self.c = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.kwargs = dict(
            s=self.s,
            c=self.c,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
        )

    def test_values(self):
        factory = OceanSg1Factory(**self.kwargs)
        assert factory.dependencies == self.kwargs


class Test_make_coord:
    @staticmethod
    def coord_dims(coord):
        mapping = dict(s=(0,), c=(0,), eta=(1, 2), depth=(1, 2), depth_c=())
        return mapping[coord.name()]

    @staticmethod
    def derive(s, c, eta, depth, depth_c, coord=True):
        S = depth_c * s + (depth - depth_c) * c
        result = S + eta * (1 + S / depth)
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
        self.s = DimCoord(np.linspace(-0.985, -0.014, 36), units="1", long_name="s")
        self.c = DimCoord(np.linspace(-0.959, -0.001, 36), units="1", long_name="c")
        self.eta = AuxCoord(
            np.arange(-1, 3, dtype=np.float64).reshape(2, 2),
            long_name="eta",
            units="m",
        )
        self.depth = AuxCoord(
            np.array([[5, 200], [1000, 4000]], dtype=np.float64),
            long_name="depth",
            units="m",
        )
        self.depth_c = AuxCoord([5], long_name="depth_c", units="m")
        self.kwargs = dict(
            s=self.s,
            c=self.c,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
        )

    def test_derived_points(self):
        # Broadcast expected points given the known dimensional mapping.
        s = self.s.points[..., np.newaxis, np.newaxis]
        c = self.c.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        depth_c = self.depth_c.points
        # Calculate the expected result.
        expected_coord = self.derive(s, c, eta, depth, depth_c)
        # Calculate the actual result.
        factory = OceanSg1Factory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.s = Mock(units=Unit("1"), nbounds=0)
        self.c = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.kwargs = dict(
            s=self.s,
            c=self.c,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
        )
        self.factory = OceanSg1Factory(**self.kwargs)

    def test_s(self):
        new_s = Mock(units=Unit("1"), nbounds=0)
        self.factory.update(self.s, new_s)
        assert self.factory.s is new_s

    def test_c(self):
        new_c = Mock(units=Unit("1"), nbounds=0)
        self.factory.update(self.c, new_c)
        assert self.factory.c is new_c

    def test_s_too_many_bounds(self):
        new_s = Mock(units=Unit("1"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.s, new_s)

    def test_c_too_many_bounds(self):
        new_c = Mock(units=Unit("1"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.c, new_c)

    def test_s_incompatible_units(self):
        new_s = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.s, new_s)

    def test_c_incompatible_units(self):
        new_c = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.c, new_c)

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

    def test_depth_c(self):
        new_depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.factory.update(self.depth_c, new_depth_c)
        assert self.factory.depth_c is new_depth_c

    def test_depth_c_non_scalar(self):
        new_depth_c = Mock(units=Unit("m"), nbounds=0, shape=(10,))
        with pytest.raises(ValueError):
            self.factory.update(self.depth_c, new_depth_c)

    def test_depth_c_incompatible_units(self):
        new_depth_c = Mock(units=Unit("Pa"), nbounds=0, shape=(1,))
        with pytest.raises(ValueError):
            self.factory.update(self.depth_c, new_depth_c)

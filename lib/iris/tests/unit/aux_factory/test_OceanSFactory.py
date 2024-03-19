# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.OceanSFactory` class.

"""

from unittest.mock import Mock

from cf_units import Unit
import numpy as np
import pytest

from iris.aux_factory import OceanSFactory
from iris.coords import AuxCoord, DimCoord


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.s = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.a = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.b = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.kwargs = dict(
            s=self.s,
            eta=self.eta,
            depth=self.depth,
            a=self.a,
            b=self.b,
            depth_c=self.depth_c,
        )

    def test_insufficient_coordinates(self):
        with pytest.raises(ValueError):
            OceanSFactory()
        with pytest.raises(ValueError):
            OceanSFactory(
                s=None,
                eta=self.eta,
                depth=self.depth,
                a=self.a,
                b=self.b,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSFactory(
                s=self.s,
                eta=None,
                depth=self.depth,
                a=self.a,
                b=self.b,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSFactory(
                s=self.s,
                eta=self.eta,
                depth=None,
                a=self.a,
                b=self.b,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSFactory(
                s=self.s,
                eta=self.eta,
                depth=self.depth,
                a=None,
                b=self.b,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSFactory(
                s=self.s,
                eta=self.eta,
                depth=self.depth,
                a=self.a,
                b=None,
                depth_c=self.depth_c,
            )
        with pytest.raises(ValueError):
            OceanSFactory(
                s=self.s,
                eta=self.eta,
                depth=self.depth,
                a=self.a,
                b=self.b,
                depth_c=None,
            )

    def test_s_too_many_bounds(self):
        self.s.nbounds = 4
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_a_non_scalar(self):
        self.a.shape = (2,)
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_b_non_scalar(self):
        self.b.shape = (2,)
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_depth_c_non_scalar(self):
        self.depth_c.shape = (2,)
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_s_incompatible_units(self):
        self.s.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_eta_incompatible_units(self):
        self.eta.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_depth_c_incompatible_units(self):
        self.depth_c.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_depth_incompatible_units(self):
        self.depth.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSFactory(**self.kwargs)

    def test_promote_s_units_unknown_to_dimensionless(self):
        s = Mock(units=Unit("unknown"), nbounds=0)
        self.kwargs["s"] = s
        factory = OceanSFactory(**self.kwargs)
        assert factory.dependencies["s"].units == "1"


class Test_dependencies:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.s = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.a = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.b = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.kwargs = dict(
            s=self.s,
            eta=self.eta,
            depth=self.depth,
            a=self.a,
            b=self.b,
            depth_c=self.depth_c,
        )

    def test_values(self):
        factory = OceanSFactory(**self.kwargs)
        assert factory.dependencies == self.kwargs


class Test_make_coord:
    @staticmethod
    def coord_dims(coord):
        mapping = dict(s=(0,), eta=(1, 2), depth=(1, 2), a=(), b=(), depth_c=())
        return mapping[coord.name()]

    @staticmethod
    def derive(s, eta, depth, a, b, depth_c, coord=True):
        c = (1 - b) * np.sinh(a * s) / np.sinh(a) + b * (
            np.tanh(a * (s + 0.5)) / (2 * np.tanh(0.5 * a)) - 0.5
        )
        result = eta * (1 + s) + depth_c * s + (depth - depth_c) * c
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
        self.s = DimCoord(
            np.arange(-0.975, 0, 0.05, dtype=float), units="1", long_name="s"
        )
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
        self.a = AuxCoord([4], units="1", long_name="a")
        self.b = AuxCoord([0.9], units="1", long_name="b")
        self.depth_c = AuxCoord([4], long_name="depth_c", units="m")
        self.kwargs = dict(
            s=self.s,
            eta=self.eta,
            depth=self.depth,
            a=self.a,
            b=self.b,
            depth_c=self.depth_c,
        )

    def test_derived_points(self):
        # Broadcast expected points given the known dimensional mapping.
        s = self.s.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        a = self.a.points
        b = self.b.points
        depth_c = self.depth_c.points
        # Calculate the expected result.
        expected_coord = self.derive(s, eta, depth, a, b, depth_c)
        # Calculate the actual result.
        factory = OceanSFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.s = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.a = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.b = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.kwargs = dict(
            s=self.s,
            eta=self.eta,
            depth=self.depth,
            a=self.a,
            b=self.b,
            depth_c=self.depth_c,
        )
        self.factory = OceanSFactory(**self.kwargs)

    def test_s(self):
        new_s = Mock(units=Unit("1"), nbounds=0)
        self.factory.update(self.s, new_s)
        assert self.factory.s is new_s

    def test_s_too_many_bounds(self):
        new_s = Mock(units=Unit("1"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.s, new_s)

    def test_s_incompatible_units(self):
        new_s = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.s, new_s)

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

    def test_a(self):
        new_a = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.factory.update(self.a, new_a)
        assert self.factory.a is new_a

    def test_a_non_scalar(self):
        new_a = Mock(units=Unit("1"), nbounds=0, shape=(10,))
        with pytest.raises(ValueError):
            self.factory.update(self.a, new_a)

    def test_b(self):
        new_b = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.factory.update(self.b, new_b)
        assert self.factory.b is new_b

    def test_b_non_scalar(self):
        new_b = Mock(units=Unit("1"), nbounds=0, shape=(10,))
        with pytest.raises(ValueError):
            self.factory.update(self.b, new_b)

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

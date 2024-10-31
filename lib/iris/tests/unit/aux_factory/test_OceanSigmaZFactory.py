# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.OceanSigmaZFactory` class.

"""

from unittest.mock import Mock

from cf_units import Unit
import numpy as np
import pytest

from iris.aux_factory import OceanSigmaZFactory
from iris.coords import AuxCoord, DimCoord


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.nsigma = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.zlev = Mock(units=Unit("m"), nbounds=0)
        self.kwargs = dict(
            sigma=self.sigma,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
            nsigma=self.nsigma,
            zlev=self.zlev,
        )

    def test_insufficient_coordinates(self):
        with pytest.raises(ValueError):
            OceanSigmaZFactory()
        with pytest.raises(ValueError):
            OceanSigmaZFactory(
                sigma=self.sigma,
                eta=self.eta,
                depth=self.depth,
                depth_c=self.depth_c,
                nsigma=self.nsigma,
                zlev=None,
            )
        with pytest.raises(ValueError):
            OceanSigmaZFactory(
                sigma=None,
                eta=None,
                depth=self.depth,
                depth_c=self.depth_c,
                nsigma=self.nsigma,
                zlev=self.zlev,
            )
        with pytest.raises(ValueError):
            OceanSigmaZFactory(
                sigma=self.sigma,
                eta=None,
                depth=None,
                depth_c=self.depth_c,
                nsigma=self.nsigma,
                zlev=self.zlev,
            )
        with pytest.raises(ValueError):
            OceanSigmaZFactory(
                sigma=self.sigma,
                eta=None,
                depth=self.depth,
                depth_c=None,
                nsigma=self.nsigma,
                zlev=self.zlev,
            )
        with pytest.raises(ValueError):
            OceanSigmaZFactory(
                sigma=self.sigma,
                eta=self.eta,
                depth=self.depth,
                depth_c=self.depth_c,
                nsigma=None,
                zlev=self.zlev,
            )

    def test_sigma_too_many_bounds(self):
        self.sigma.nbounds = 4
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_zlev_too_many_bounds(self):
        self.zlev.nbounds = 4
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_sigma_zlev_same_boundedness(self):
        self.zlev.nbounds = 2
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_depth_c_non_scalar(self):
        self.depth_c.shape = (2,)
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_nsigma_non_scalar(self):
        self.nsigma.shape = (4,)
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_zlev_incompatible_units(self):
        self.zlev.units = Unit("Pa")
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_sigma_incompatible_units(self):
        self.sigma.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_eta_incompatible_units(self):
        self.eta.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_depth_c_incompatible_units(self):
        self.depth_c.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_depth_incompatible_units(self):
        self.depth.units = Unit("km")
        with pytest.raises(ValueError):
            OceanSigmaZFactory(**self.kwargs)

    def test_promote_sigma_units_unknown_to_dimensionless(self):
        sigma = Mock(units=Unit("unknown"), nbounds=0)
        self.kwargs["sigma"] = sigma
        factory = OceanSigmaZFactory(**self.kwargs)
        assert factory.dependencies["sigma"].units == "1"


class Test_dependencies:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.nsigma = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.zlev = Mock(units=Unit("m"), nbounds=0)
        self.kwargs = dict(
            sigma=self.sigma,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
            nsigma=self.nsigma,
            zlev=self.zlev,
        )

    def test_values(self):
        factory = OceanSigmaZFactory(**self.kwargs)
        assert factory.dependencies == self.kwargs


class Test_make_coord:
    @staticmethod
    def coord_dims(coord):
        mapping = dict(
            sigma=(0,),
            eta=(1, 2),
            depth=(1, 2),
            depth_c=(),
            nsigma=(),
            zlev=(0,),
        )
        return mapping[coord.name()]

    @staticmethod
    def derive(sigma, eta, depth, depth_c, nsigma, zlev, coord=True):
        nsigma_slice = slice(0, int(nsigma))
        temp = eta + sigma * (np.minimum(depth_c, depth) + eta)
        shape = temp.shape
        result = np.ones(shape, dtype=temp.dtype) * zlev
        result[nsigma_slice] = temp[nsigma_slice]
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
        self.sigma = DimCoord(
            np.arange(5, dtype=np.float64) * 10, long_name="sigma", units="1"
        )
        self.eta = AuxCoord(
            np.arange(4, dtype=np.float64).reshape(2, 2),
            long_name="eta",
            units="m",
        )
        self.depth = AuxCoord(
            np.arange(4, dtype=np.float64).reshape(2, 2) * 10,
            long_name="depth",
            units="m",
        )
        self.depth_c = AuxCoord([15], long_name="depth_c", units="m")
        self.nsigma = AuxCoord([3], long_name="nsigma")
        self.zlev = DimCoord(
            np.arange(5, dtype=np.float64) * 10, long_name="zlev", units="m"
        )
        self.kwargs = dict(
            sigma=self.sigma,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
            nsigma=self.nsigma,
            zlev=self.zlev,
        )

    def test_derived_points(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        depth_c = self.depth_c.points
        nsigma = self.nsigma.points
        zlev = self.zlev.points[..., np.newaxis, np.newaxis]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth, depth_c, nsigma, zlev)
        # Calculate the actual result.
        factory = OceanSigmaZFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord

    def test_derived_points_with_bounds(self):
        self.sigma.guess_bounds()
        self.zlev.guess_bounds()
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        depth_c = self.depth_c.points
        nsigma = self.nsigma.points
        zlev = self.zlev.points[..., np.newaxis, np.newaxis]
        # Calculate the expected coordinate with points.
        expected_coord = self.derive(sigma, eta, depth, depth_c, nsigma, zlev)
        # Broadcast expected bounds given the known dimensional mapping.
        sigma = self.sigma.bounds.reshape(sigma.shape + (2,))
        eta = self.eta.points.reshape(eta.shape + (1,))
        depth = self.depth.points.reshape(depth.shape + (1,))
        depth_c = self.depth_c.points.reshape(depth_c.shape + (1,))
        nsigma = self.nsigma.points.reshape(nsigma.shape + (1,))
        zlev = self.zlev.bounds.reshape(zlev.shape + (2,))
        # Calculate the expected bounds.
        bounds = self.derive(sigma, eta, depth, depth_c, nsigma, zlev, coord=False)
        expected_coord.bounds = bounds
        # Calculate the actual result.
        factory = OceanSigmaZFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord

    def test_no_eta(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = 0
        depth = self.depth.points[np.newaxis, ...]
        depth_c = self.depth_c.points
        nsigma = self.nsigma.points
        zlev = self.zlev.points[..., np.newaxis, np.newaxis]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth, depth_c, nsigma, zlev)
        # Calculate the actual result.
        self.kwargs["eta"] = None
        factory = OceanSigmaZFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord

    def test_no_sigma(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = 0
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        depth_c = self.depth_c.points
        nsigma = self.nsigma.points
        zlev = self.zlev.points[..., np.newaxis, np.newaxis]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth, depth_c, nsigma, zlev)
        # Calculate the actual result.
        self.kwargs["sigma"] = None
        factory = OceanSigmaZFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord

    def test_no_depth_c(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = self.depth.points[np.newaxis, ...]
        depth_c = 0
        nsigma = self.nsigma.points
        zlev = self.zlev.points[..., np.newaxis, np.newaxis]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth, depth_c, nsigma, zlev)
        # Calculate the actual result.
        self.kwargs["depth_c"] = None
        factory = OceanSigmaZFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord

    def test_no_depth(self):
        # Broadcast expected points given the known dimensional mapping.
        sigma = self.sigma.points[..., np.newaxis, np.newaxis]
        eta = self.eta.points[np.newaxis, ...]
        depth = 0
        depth_c = self.depth_c.points
        nsigma = self.nsigma.points
        zlev = self.zlev.points[..., np.newaxis, np.newaxis]
        # Calculate the expected result.
        expected_coord = self.derive(sigma, eta, depth, depth_c, nsigma, zlev)
        # Calculate the actual result.
        self.kwargs["depth"] = None
        factory = OceanSigmaZFactory(**self.kwargs)
        coord = factory.make_coord(self.coord_dims)
        assert coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sigma = Mock(units=Unit("1"), nbounds=0)
        self.eta = Mock(units=Unit("m"), nbounds=0)
        self.depth = Mock(units=Unit("m"), nbounds=0)
        self.depth_c = Mock(units=Unit("m"), nbounds=0, shape=(1,))
        self.nsigma = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.zlev = Mock(units=Unit("m"), nbounds=0)
        self.kwargs = dict(
            sigma=self.sigma,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
            nsigma=self.nsigma,
            zlev=self.zlev,
        )
        self.factory = OceanSigmaZFactory(**self.kwargs)

    def test_sigma(self):
        new_sigma = Mock(units=Unit("1"), nbounds=0)
        self.factory.update(self.sigma, new_sigma)
        assert self.factory.sigma is new_sigma

    def test_sigma_too_many_bounds(self):
        new_sigma = Mock(units=Unit("1"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.sigma, new_sigma)

    def test_sigma_zlev_same_boundedness(self):
        new_sigma = Mock(units=Unit("1"), nbounds=2)
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

    def test_nsigma(self):
        new_nsigma = Mock(units=Unit("1"), nbounds=0, shape=(1,))
        self.factory.update(self.nsigma, new_nsigma)
        assert self.factory.nsigma is new_nsigma

    def test_nsigma_missing(self):
        with pytest.raises(ValueError):
            self.factory.update(self.nsigma, None)

    def test_nsigma_non_scalar(self):
        new_nsigma = Mock(units=Unit("1"), nbounds=0, shape=(10,))
        with pytest.raises(ValueError):
            self.factory.update(self.nsigma, new_nsigma)

    def test_zlev(self):
        new_zlev = Mock(units=Unit("m"), nbounds=0)
        self.factory.update(self.zlev, new_zlev)
        assert self.factory.zlev is new_zlev

    def test_zlev_missing(self):
        with pytest.raises(ValueError):
            self.factory.update(self.zlev, None)

    def test_zlev_too_many_bounds(self):
        new_zlev = Mock(units=Unit("m"), nbounds=4)
        with pytest.raises(ValueError):
            self.factory.update(self.zlev, new_zlev)

    def test_zlev_same_boundedness(self):
        new_zlev = Mock(units=Unit("m"), nbounds=2)
        with pytest.raises(ValueError):
            self.factory.update(self.zlev, new_zlev)

    def test_zlev_incompatible_units(self):
        new_zlev = new_zlev = Mock(units=Unit("Pa"), nbounds=0)
        with pytest.raises(ValueError):
            self.factory.update(self.zlev, new_zlev)

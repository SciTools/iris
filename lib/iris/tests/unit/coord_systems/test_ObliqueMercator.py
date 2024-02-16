# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.ObliqueMercator` class."""

from typing import List, NamedTuple
from unittest.mock import Mock

from cartopy import crs as ccrs
import pytest

from iris.coord_systems import GeogCS, ObliqueMercator

####
# ALL TESTS MUST BE CONTAINED IN CLASSES, TO ENABLE INHERITANCE BY
#  test_RotatedMercator.py .
####


class GlobeWithEq(ccrs.Globe):
    def __eq__(self, other):
        """Need eq to enable comparison with expected arguments."""
        result = NotImplemented
        if isinstance(other, ccrs.Globe):
            result = other.__dict__ == self.__dict__
        return result


class ParamTuple(NamedTuple):
    """Used for easy coupling of test parameters."""

    id: str
    class_kwargs: dict
    cartopy_kwargs: dict


kwarg_permutations: List[ParamTuple] = [
    ParamTuple(
        "default",
        dict(),
        dict(),
    ),
    ParamTuple(
        "azimuth",
        dict(azimuth_of_central_line=90),
        dict(azimuth=90),
    ),
    ParamTuple(
        "central_longitude",
        dict(longitude_of_projection_origin=90),
        dict(central_longitude=90),
    ),
    ParamTuple(
        "central_latitude",
        dict(latitude_of_projection_origin=45),
        dict(central_latitude=45),
    ),
    ParamTuple(
        "false_easting_northing",
        dict(false_easting=1000000, false_northing=-2000000),
        dict(false_easting=1000000, false_northing=-2000000),
    ),
    ParamTuple(
        "scale_factor",
        # Number inherited from Cartopy's test_mercator.py .
        dict(scale_factor_at_projection_origin=0.939692620786),
        dict(scale_factor=0.939692620786),
    ),
    ParamTuple(
        "globe",
        dict(ellipsoid=GeogCS(1)),
        dict(globe=GlobeWithEq(semimajor_axis=1, semiminor_axis=1, ellipse=None)),
    ),
    ParamTuple(
        "combo",
        dict(
            azimuth_of_central_line=90,
            longitude_of_projection_origin=90,
            latitude_of_projection_origin=45,
            false_easting=1000000,
            false_northing=-2000000,
            scale_factor_at_projection_origin=0.939692620786,
            ellipsoid=GeogCS(1),
        ),
        dict(
            azimuth=90.0,
            central_longitude=90.0,
            central_latitude=45.0,
            false_easting=1000000,
            false_northing=-2000000,
            scale_factor=0.939692620786,
            globe=GlobeWithEq(semimajor_axis=1, semiminor_axis=1, ellipse=None),
        ),
    ),
]
permutation_ids: List[str] = [p.id for p in kwarg_permutations]


class TestArgs:
    GeogCS = GeogCS
    class_kwargs_default = dict(
        azimuth_of_central_line=0.0,
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
    )
    cartopy_kwargs_default = dict(
        central_longitude=0.0,
        central_latitude=0.0,
        false_easting=0.0,
        false_northing=0.0,
        scale_factor=1.0,
        azimuth=0.0,
        globe=None,
    )

    @pytest.fixture(autouse=True, params=kwarg_permutations, ids=permutation_ids)
    def make_variant_inputs(self, request) -> None:
        """Parse a ParamTuple into usable test information."""
        inputs: ParamTuple = request.param
        self.class_kwargs = dict(self.class_kwargs_default, **inputs.class_kwargs)
        self.cartopy_kwargs_expected = dict(
            self.cartopy_kwargs_default, **inputs.cartopy_kwargs
        )

    def make_instance(self) -> ObliqueMercator:
        return ObliqueMercator(**self.class_kwargs)

    @pytest.fixture()
    def instance(self):
        return self.make_instance()

    def test_instantiate(self):
        _ = self.make_instance()

    def test_cartopy_crs(self, instance):
        ccrs.ObliqueMercator = Mock()
        instance.as_cartopy_crs()
        ccrs.ObliqueMercator.assert_called_with(**self.cartopy_kwargs_expected)

    def test_cartopy_projection(self, instance):
        ccrs.ObliqueMercator = Mock()
        instance.as_cartopy_projection()
        ccrs.ObliqueMercator.assert_called_with(**self.cartopy_kwargs_expected)

    @pytest.fixture()
    def label_class(self, instance):
        """Make the tested coordinate system available, even for subclasses."""
        from iris import coord_systems

        instance_class = "{!s}".format(instance.__class__.__name__)
        globals()[instance_class] = getattr(coord_systems, instance_class)

    def test_repr(self, instance, label_class):
        """Test that the repr can be used to regenerate an identical object."""
        assert eval(repr(instance)) == instance

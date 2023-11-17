# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coord_systems.RotatedMercator` class."""

import pytest

from iris._deprecation import IrisDeprecation
from iris.coord_systems import RotatedMercator

from . import test_ObliqueMercator


class TestArgs(test_ObliqueMercator.TestArgs):
    class_kwargs_default = dict(
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
    )
    cartopy_kwargs_default = dict(
        central_longitude=0.0,
        central_latitude=0.0,
        false_easting=0.0,
        false_northing=0.0,
        scale_factor=1.0,
        azimuth=90.0,
        globe=None,
    )

    def make_instance(self) -> RotatedMercator:
        kwargs = self.class_kwargs
        kwargs.pop("azimuth_of_central_line", None)
        return RotatedMercator(**kwargs)


def test_deprecated():
    with pytest.warns(IrisDeprecation, match="azimuth_of_central_line=90"):
        _ = RotatedMercator(0, 0)

# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coord_systems.RotatedMercator` class."""

from iris.coord_systems import RotatedMercator

from . import test_ObliqueMercator


class TestArgs(test_ObliqueMercator.TestArgs):
    def make_instance(self) -> RotatedMercator:
        kwargs = self.class_kwargs
        kwargs.pop("azimuth_of_central_line", None)
        return RotatedMercator(**kwargs)


TestArgs.cartopy_kwargs_default["azimuth"] = 90.0

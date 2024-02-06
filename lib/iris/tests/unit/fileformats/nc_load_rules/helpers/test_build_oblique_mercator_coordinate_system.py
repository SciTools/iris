# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.build_oblique_mercator_coordinate_system`."""
from typing import List, NamedTuple, Type
from unittest import mock

import pytest

from iris import coord_systems
from iris._deprecation import IrisDeprecation
from iris.coord_systems import CoordSystem, GeogCS, ObliqueMercator, RotatedMercator
from iris.fileformats._nc_load_rules.helpers import (
    build_oblique_mercator_coordinate_system,
)


class ParamTuple(NamedTuple):
    """Used for easy coupling of test parameters."""

    id: str
    nc_attributes: dict
    expected_class: Type[CoordSystem]
    coord_system_kwargs: dict


kwarg_permutations: List[ParamTuple] = [
    ParamTuple(
        "default",
        dict(),
        ObliqueMercator,
        dict(),
    ),
    ParamTuple(
        "azimuth",
        dict(azimuth_of_central_line=90),
        ObliqueMercator,
        dict(azimuth_of_central_line=90),
    ),
    ParamTuple(
        "central_longitude",
        dict(longitude_of_projection_origin=90),
        ObliqueMercator,
        dict(longitude_of_projection_origin=90),
    ),
    ParamTuple(
        "central_latitude",
        dict(latitude_of_projection_origin=45),
        ObliqueMercator,
        dict(latitude_of_projection_origin=45),
    ),
    ParamTuple(
        "false_easting_northing",
        dict(false_easting=1000000, false_northing=-2000000),
        ObliqueMercator,
        dict(false_easting=1000000, false_northing=-2000000),
    ),
    ParamTuple(
        "scale_factor",
        # Number inherited from Cartopy's test_mercator.py .
        dict(scale_factor_at_projection_origin=0.939692620786),
        ObliqueMercator,
        dict(scale_factor_at_projection_origin=0.939692620786),
    ),
    ParamTuple(
        "globe",
        dict(semi_major_axis=1),
        ObliqueMercator,
        dict(ellipsoid=GeogCS(semi_major_axis=1, semi_minor_axis=1)),
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
            semi_major_axis=1,
        ),
        ObliqueMercator,
        dict(
            azimuth_of_central_line=90.0,
            longitude_of_projection_origin=90.0,
            latitude_of_projection_origin=45.0,
            false_easting=1000000,
            false_northing=-2000000,
            scale_factor_at_projection_origin=0.939692620786,
            ellipsoid=GeogCS(semi_major_axis=1, semi_minor_axis=1),
        ),
    ),
    ParamTuple(
        "rotated",
        dict(grid_mapping_name="rotated_mercator"),
        RotatedMercator,
        dict(),
    ),
    ParamTuple(
        "rotated_azimuth_ignored",
        dict(
            grid_mapping_name="rotated_mercator",
            azimuth_of_central_line=45,
        ),
        RotatedMercator,
        dict(),
    ),
]
permutation_ids: List[str] = [p.id for p in kwarg_permutations]


class TestAttributes:
    """Test that NetCDF attributes are correctly converted to class arguments."""

    nc_attributes_default = dict(
        grid_mapping_name="oblique_mercator",
        azimuth_of_central_line=0.0,
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        scale_factor_at_projection_origin=1.0,
        # Optional attributes not included.
    )
    coord_system_kwargs_default = dict(
        azimuth_of_central_line=0.0,
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        false_easting=None,
        false_northing=None,
        scale_factor_at_projection_origin=1.0,
        ellipsoid=None,
    )

    @pytest.fixture(autouse=True, params=kwarg_permutations, ids=permutation_ids)
    def make_variant_inputs(self, request) -> None:
        """Parse a ParamTuple into usable test information."""
        inputs: ParamTuple = request.param

        self.nc_attributes = dict(self.nc_attributes_default, **inputs.nc_attributes)
        self.expected_class = inputs.expected_class
        coord_system_kwargs_expected = dict(
            self.coord_system_kwargs_default, **inputs.coord_system_kwargs
        )

        if self.expected_class is RotatedMercator:
            del coord_system_kwargs_expected["azimuth_of_central_line"]

        self.coord_system_args_expected = list(coord_system_kwargs_expected.values())

    def test_attributes(self):
        cf_var_mock = mock.Mock(spec=[], **self.nc_attributes)
        coord_system_mock = mock.Mock(spec=self.expected_class)
        setattr(coord_systems, self.expected_class.__name__, coord_system_mock)

        _ = build_oblique_mercator_coordinate_system(None, cf_var_mock)
        coord_system_mock.assert_called_with(*self.coord_system_args_expected)


def test_deprecation():
    nc_attributes = dict(
        grid_mapping_name="rotated_mercator",
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        scale_factor_at_projection_origin=1.0,
    )
    cf_var_mock = mock.Mock(spec=[], **nc_attributes)
    with pytest.warns(IrisDeprecation, match="azimuth_of_central_line = 90"):
        _ = build_oblique_mercator_coordinate_system(None, cf_var_mock)

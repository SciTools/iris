# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
build_polar_stereographic_coordinate_system`.

"""

import iris
from iris.coord_systems import PolarStereographic
from iris.fileformats._nc_load_rules.helpers import (
    build_polar_stereographic_coordinate_system,
)


class TestBuildPolarStereographicCoordinateSystem:
    def test_valid_north(self, mocker):
        cf_grid_var = mocker.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )

        cs = build_polar_stereographic_coordinate_system(None, cf_grid_var)

        expected = PolarStereographic(
            central_lon=(cf_grid_var.straight_vertical_longitude_from_pole),
            central_lat=(cf_grid_var.latitude_of_projection_origin),
            scale_factor_at_projection_origin=(
                cf_grid_var.scale_factor_at_projection_origin
            ),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis, cf_grid_var.semi_minor_axis
            ),
        )
        assert cs == expected

    def test_valid_south(self, mocker):
        cf_grid_var = mocker.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=-90,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )

        cs = build_polar_stereographic_coordinate_system(None, cf_grid_var)

        expected = PolarStereographic(
            central_lon=(cf_grid_var.straight_vertical_longitude_from_pole),
            central_lat=(cf_grid_var.latitude_of_projection_origin),
            scale_factor_at_projection_origin=(
                cf_grid_var.scale_factor_at_projection_origin
            ),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis, cf_grid_var.semi_minor_axis
            ),
        )
        assert cs == expected

    def test_valid_with_standard_parallel(self, mocker):
        cf_grid_var = mocker.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            standard_parallel=30,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )

        cs = build_polar_stereographic_coordinate_system(None, cf_grid_var)

        expected = PolarStereographic(
            central_lon=(cf_grid_var.straight_vertical_longitude_from_pole),
            central_lat=(cf_grid_var.latitude_of_projection_origin),
            true_scale_lat=(cf_grid_var.standard_parallel),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis, cf_grid_var.semi_minor_axis
            ),
        )
        assert cs == expected

    def test_valid_with_false_easting_northing(self, mocker):
        cf_grid_var = mocker.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            scale_factor_at_projection_origin=1,
            false_easting=30,
            false_northing=40,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )

        cs = build_polar_stereographic_coordinate_system(None, cf_grid_var)

        expected = PolarStereographic(
            central_lon=(cf_grid_var.straight_vertical_longitude_from_pole),
            central_lat=(cf_grid_var.latitude_of_projection_origin),
            scale_factor_at_projection_origin=(
                cf_grid_var.scale_factor_at_projection_origin
            ),
            false_easting=(cf_grid_var.false_easting),
            false_northing=(cf_grid_var.false_northing),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis, cf_grid_var.semi_minor_axis
            ),
        )
        assert cs == expected

    def test_valid_nonzero_veritcal_lon(self, mocker):
        cf_grid_var = mocker.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=30,
            latitude_of_projection_origin=90,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )

        cs = build_polar_stereographic_coordinate_system(None, cf_grid_var)

        expected = PolarStereographic(
            central_lon=(cf_grid_var.straight_vertical_longitude_from_pole),
            central_lat=(cf_grid_var.latitude_of_projection_origin),
            scale_factor_at_projection_origin=(
                cf_grid_var.scale_factor_at_projection_origin
            ),
            ellipsoid=iris.coord_systems.GeogCS(
                cf_grid_var.semi_major_axis, cf_grid_var.semi_minor_axis
            ),
        )
        assert cs == expected

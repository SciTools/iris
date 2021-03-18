# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for NetCDF-UGRID file loading.

todo: fold these tests into netcdf tests when experimental.ugrid is folded into
 standard behaviour.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections.abc import Iterable

from iris import Constraint, load
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD


def ugrid_load(uris, constraints=None, callback=None):
    # TODO: remove constraint once files no longer have orphan connectivities.
    orphan_connectivities = (
        "Mesh2d_half_levels_edge_face_links",
        "Mesh2d_half_levels_face_links",
        "Mesh2d_half_levels_face_edges",
        "Mesh2d_full_levels_edge_face_links",
        "Mesh2d_full_levels_face_links",
        "Mesh2d_full_levels_face_edges",
    )
    filter_orphan_connectivities = Constraint(
        cube_func=lambda cube: cube.var_name not in orphan_connectivities
    )
    if constraints is None:
        constraints = filter_orphan_connectivities
    else:
        if not isinstance(constraints, Iterable):
            constraints = [constraints]
        constraints.append(filter_orphan_connectivities)

    with PARSE_UGRID_ON_LOAD.context():
        return load(uris, constraints, callback)


@tests.skip_data
class TestBasic(tests.IrisTest):
    def common_test(self, load_filename, assert_filename):
        cube_list = ugrid_load(
            tests.get_data_path(
                ["NetCDF", "unstructured_grid", load_filename]
            ),
        )
        self.assertEqual(1, len(cube_list))
        cube = cube_list[0]
        self.assertCML(cube, ["experimental", "ugrid", assert_filename])

    def test_2D_1t_face_half_levels(self):
        self.common_test(
            "lfric_ngvat_2D_1t_face_half_levels_main_conv_rain.nc",
            "2D_1t_face_half_levels.cml",
        )

    def test_3D_1t_face_half_levels(self):
        self.common_test(
            "lfric_ngvat_3D_1t_half_level_face_grid_derived_theta_in_w3.nc",
            "3D_1t_face_half_levels.cml",
        )

    def test_3D_1t_face_full_levels(self):
        self.common_test(
            "lfric_ngvat_3D_1t_full_level_face_grid_main_area_fraction_unit1.nc",
            "3D_1t_face_full_levels.cml",
        )

    def test_2D_72t_face_half_levels(self):
        self.common_test(
            "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
            "2D_72t_face_half_levels.cml",
        )

    def test_3D_snow_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_snow_pseudo_levels_1t_face_half_levels_main_snow_layer_temp.nc",
            "3D_snow_pseudo_levels.cml",
        )

    def test_3D_soil_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_soil_pseudo_levels_1t_face_half_levels_main_soil_temperature.nc",
            "3D_soil_pseudo_levels.cml",
        )

    def test_3D_tile_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_tile_pseudo_levels_1t_face_half_levels_main_sw_up_tile.nc",
            "3D_tile_pseudo_levels.cml",
        )

    def test_3D_veg_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_veg_pseudo_levels_1t_face_half_levels_main_snowpack_density.nc",
            "3D_veg_pseudo_levels.cml",
        )


class TestMultiplePhenomena(tests.IrisTest):
    def test_multiple_phenomena(self):
        cube_list = ugrid_load(
            tests.get_data_path(
                ["NetCDF", "unstructured_grid", "lfric_surface_mean.nc"]
            ),
        )
        self.assertCML(
            cube_list, ("experimental", "ugrid", "surface_mean.cml")
        )

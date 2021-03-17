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

from iris import load, load_cube, NameConstraint
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD


def ugrid_load(*args, **kwargs):
    with PARSE_UGRID_ON_LOAD.context():
        return load(*args, **kwargs)


def ugrid_load_cube(*args, **kwargs):
    with PARSE_UGRID_ON_LOAD.context():
        return load_cube(*args, **kwargs)


@tests.skip_data
class TestBasic(tests.IrisTest):
    def test_2D_1t_face_half_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        conv_rain = NameConstraint(var_name="conv_rain")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_2D_1t_face_half_levels_main_conv_rain.nc",
                ]
            ),
            constraint=conv_rain,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "2D_1t_face_half_levels.cml")
        )

    def test_3D_1t_face_half_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        theta_in_w3 = NameConstraint(var_name="theta_in_w3")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_3D_1t_half_level_face_grid_derived_theta_in_w3.nc",
                ]
            ),
            constraint=theta_in_w3,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "3D_1t_face_half_levels.cml")
        )

    def test_3D_1t_face_full_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        area_fraction = NameConstraint(var_name="area_fraction")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_3D_1t_full_level_face_grid_main_area_fraction_unit1.nc",
                ]
            ),
            constraint=area_fraction,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "3D_1t_face_full_levels.cml")
        )

    def test_2D_72t_face_half_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        conv_rain = NameConstraint(var_name="conv_rain")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
                ]
            ),
            constraint=conv_rain,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "2D_72t_face_half_levels.cml")
        )


class TestPseudoLevels(tests.IrisTest):
    def test_3D_snow_pseudo_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        snow_layer_temp = NameConstraint(var_name="snow_layer_temp")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_3D_snow_pseudo_levels_1t_face_half_levels_main_snow_layer_temp.nc",
                ]
            ),
            constraint=snow_layer_temp,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "3D_snow_pseudo_levels.cml")
        )

    def test_3D_soil_pseudo_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        soil_temperature = NameConstraint(var_name="soil_temperature")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_3D_soil_pseudo_levels_1t_face_half_levels_main_soil_temperature.nc",
                ]
            ),
            constraint=soil_temperature,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "3D_soil_pseudo_levels.cml")
        )

    def test_3D_tile_pseudo_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        sw_up_tile = NameConstraint(var_name="sw_up_tile")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_3D_tile_pseudo_levels_1t_face_half_levels_main_sw_up_tile.nc",
                ]
            ),
            constraint=sw_up_tile,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "3D_tile_pseudo_levels.cml")
        )

    def test_3D_veg_pseudo_levels(self):
        # TODO: remove constraint once file no longer has orphan connectivities.
        snowpack_density = NameConstraint(var_name="snowpack_density")
        cube = ugrid_load_cube(
            tests.get_data_path(
                [
                    "NetCDF",
                    "unstructured_grid",
                    "lfric_ngvat_3D_veg_pseudo_levels_1t_face_half_levels_main_snowpack_density.nc",
                ]
            ),
            constraint=snowpack_density,
        )
        self.assertCML(
            cube, ("experimental", "ugrid", "3D_veg_pseudo_levels.cml")
        )

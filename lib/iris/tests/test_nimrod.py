# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

import iris
from iris.exceptions import TranslationError
import iris.fileformats.nimrod_load_rules as nimrod_load_rules


def mock_nimrod_field():
    field = iris.fileformats.nimrod.NimrodField()
    field.int_mdi = -32767
    field.float32_mdi = -32767.0
    return field


class TestLoad(tests.IrisTest):
    @tests.skip_data
    def test_multi_field_load(self):
        # load a cube with two fields
        cube = iris.load(
            tests.get_data_path(
                (
                    "NIMROD",
                    "uk2km",
                    "WO0000000003452",
                    "201007020900_u1096_ng_ey00_visibility0180_screen_2km",
                )
            )
        )
        self.assertCML(cube, ("nimrod", "load_2flds.cml"))

    @tests.skip_data
    def test_huge_field_load(self):
        # load a wide range of cubes with all meta-data variations
        for datafile in {
            "u1096_ng_ek07_precip0540_accum180_18km",
            "u1096_ng_ek00_cloud3d0060_2km",
            "u1096_ng_ek00_cloud_2km",
            "u1096_ng_ek00_convection_2km",
            "u1096_ng_ek00_convwind_2km",
            "u1096_ng_ek00_frzlev_2km",
            "u1096_ng_ek00_height_2km",
            "u1096_ng_ek00_precip_2km",
            "u1096_ng_ek00_precipaccum_2km",
            "u1096_ng_ek00_preciptype_2km",
            "u1096_ng_ek00_pressure_2km",
            "u1096_ng_ek00_radiation_2km",
            "u1096_ng_ek00_radiationuv_2km",
            "u1096_ng_ek00_refl_2km",
            "u1096_ng_ek00_relhumidity3d0060_2km",
            "u1096_ng_ek00_relhumidity_2km",
            "u1096_ng_ek00_snow_2km",
            "u1096_ng_ek00_soil3d0060_2km",
            "u1096_ng_ek00_soil_2km",
            "u1096_ng_ek00_temperature_2km",
            "u1096_ng_ek00_visibility_2km",
            "u1096_ng_ek00_wind_2km",
            "u1096_ng_ek00_winduv3d0015_2km",
            "u1096_ng_ek00_winduv_2km",
            "u1096_ng_ek01_cape_2km",
            "u1096_ng_umqv_fog_2km",
            "u1096_ng_bmr04_precip_2km",
            "u1096_ng_bsr05_precip_accum60_2km",
            "probability_fields",
        }:
            cube = iris.load(
                tests.get_data_path(("NIMROD", "uk2km", "cutouts", datafile))
            )
            self.assertCML(cube, ("nimrod", f"{datafile}.cml"))

    @tests.skip_data
    def test_load_kwarg(self):
        """Tests that the handle_metadata_errors kwarg is effective by setting it to
        False with a file with known incomplete meta-data (missing ellipsoid)."""
        datafile = "u1096_ng_ek00_pressure_2km"
        with self.assertRaisesRegex(
            TranslationError,
            "Ellipsoid not supported, proj_biaxial_ellipsoid:-32767, horizontal_grid_type:0",
        ):
            with open(
                tests.get_data_path(("NIMROD", "uk2km", "cutouts", datafile)),
                "rb",
            ) as infile:
                iris.fileformats.nimrod_load_rules.run(
                    iris.fileformats.nimrod.NimrodField(infile),
                    handle_metadata_errors=False,
                )

    def test_orography(self):
        # Mock an orography field we've seen.
        field = mock_nimrod_field()
        cube = iris.cube.Cube(np.arange(100).reshape(10, 10))

        field.dt_year = field.dt_month = field.dt_day = field.int_mdi
        field.dt_hour = field.dt_minute = field.int_mdi
        field.proj_biaxial_ellipsoid = 0
        field.tm_meridian_scaling = 0.999601
        field.field_code = 73
        field.reference_vertical_coord_type = field.int_mdi  # Not bounded
        field.reference_vertical_coord = field.int_mdi
        field.vertical_coord_type = 1
        field.vertical_coord = 8888
        field.ensemble_member = field.int_mdi
        field.threshold_value = field.int_mdi
        field.title = "(MOCK) 2km mean orography"
        field.units = "metres"
        field.source = "GLOBE DTM"

        nimrod_load_rules.name(cube, field, handle_metadata_errors=True)
        nimrod_load_rules.units(cube, field)
        nimrod_load_rules.reference_time(cube, field)
        nimrod_load_rules.vertical_coord(cube, field)
        nimrod_load_rules.attributes(cube, field)

        self.assertCML(cube, ("nimrod", "mockography.cml"))

    def test_levels_below_ground(self):
        # Mock a soil temperature field we've seen.
        field = mock_nimrod_field()
        cube = iris.cube.Cube(np.arange(100).reshape(10, 10))

        field.field_code = -1  # Not orography
        field.reference_vertical_coord_type = field.int_mdi  # Not bounded
        field.reference_vertical_coord = field.int_mdi
        field.vertical_coord_type = 12
        field.vertical_coord = 42
        nimrod_load_rules.vertical_coord(cube, field)

        self.assertCML(cube, ("nimrod", "levels_below_ground.cml"))

    def test_period_of_interest(self):
        # mock a pressure field
        field = mock_nimrod_field()
        cube = iris.cube.Cube(np.arange(100).reshape(10, 10))

        field.field_code = 0
        field.vt_year = 2013
        field.vt_month = 5
        field.vt_day = 7
        field.vt_hour = 6
        field.vt_minute = 0
        field.vt_second = 0
        field.dt_year = 2013
        field.dt_month = 5
        field.dt_day = 7
        field.dt_hour = 6
        field.dt_minute = 0
        field.dt_second = 0
        field.period_minutes = 60

        nimrod_load_rules.time(cube, field)

        self.assertCML(cube, ("nimrod", "period_of_interest.cml"))


if __name__ == "__main__":
    tests.main()

# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
has_supported_polar_stereographic_parameters`.

"""

from unittest import mock
import warnings

from iris.fileformats._nc_load_rules.helpers import (
    has_supported_polar_stereographic_parameters,
)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip


def _engine(cf_grid_var, cf_name):
    cf_group = {cf_name: cf_grid_var}
    cf_var = mock.Mock(cf_group=cf_group)
    return mock.Mock(cf_var=cf_var)


class TestHasSupportedPolarStereographicParameters(tests.IrisTest):
    def test_valid_base_north(self):
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_polar_stereographic_parameters(
            engine, cf_name
        )

        self.assertTrue(is_valid)

    def test_valid_base_south(self):
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=-90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_polar_stereographic_parameters(
            engine, cf_name
        )

        self.assertTrue(is_valid)

    def test_valid_straight_vertical_longitude(self):
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=30,
            latitude_of_projection_origin=90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_polar_stereographic_parameters(
            engine, cf_name
        )

        self.assertTrue(is_valid)

    def test_valid_false_easting_northing(self):
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            false_easting=15,
            false_northing=10,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_polar_stereographic_parameters(
            engine, cf_name
        )

        self.assertTrue(is_valid)

    def test_valid_standard_parallel(self):
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            false_easting=0,
            false_northing=0,
            standard_parallel=15,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_polar_stereographic_parameters(
            engine, cf_name
        )

        self.assertTrue(is_valid)

    def test_valid_scale_factor(self):
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=0.9,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_polar_stereographic_parameters(
            engine, cf_name
        )

        self.assertTrue(is_valid)

    def test_invalid_scale_factor_and_standard_parallel(self):
        # Scale factor and standard parallel cannot both be specified for
        # Polar Stereographic projections
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=0.9,
            standard_parallel=20,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_polar_stereographic_parameters(
                engine, cf_name
            )

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(
            str(warns[0]),
            "both "
            '"scale_factor_at_projection_origin" and "standard_parallel"',
        )

    def test_absent_scale_factor_and_standard_parallel(self):
        # Scale factor and standard parallel cannot both be specified for
        # Polar Stereographic projections
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=90,
            false_easting=0,
            false_northing=0,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_polar_stereographic_parameters(
                engine, cf_name
            )

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(
            str(warns[0]),
            'One of "scale_factor_at_projection_origin" and '
            '"standard_parallel" is required.',
        )

    def test_invalid_latitude_of_projection_origin(self):
        # Scale factor and standard parallel cannot both be specified for
        # Polar Stereographic projections
        cf_name = "polar_stereographic"
        cf_grid_var = mock.Mock(
            spec=[],
            straight_vertical_longitude_from_pole=0,
            latitude_of_projection_origin=45,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_polar_stereographic_parameters(
                engine, cf_name
            )

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(
            str(warns[0]),
            r'"latitude_of_projection_origin" must be \+90 or -90\.',
        )


if __name__ == "__main__":
    tests.main()

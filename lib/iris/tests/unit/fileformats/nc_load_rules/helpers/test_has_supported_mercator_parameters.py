# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test function :func:`iris.fileformats._nc_load_rules.helpers.\
has_supported_mercator_parameters`.

"""

from unittest import mock
import warnings

from iris.fileformats._nc_load_rules.helpers import (
    has_supported_mercator_parameters,
)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip


def _engine(cf_grid_var, cf_name):
    cf_group = {cf_name: cf_grid_var}
    cf_var = mock.Mock(cf_group=cf_group)
    return mock.Mock(cf_var=cf_var)


class TestHasSupportedMercatorParameters(tests.IrisTest):
    def test_valid(self):
        cf_name = "mercator"
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=-90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        is_valid = has_supported_mercator_parameters(engine, cf_name)

        self.assertTrue(is_valid)

    def test_invalid_scale_factor(self):
        # Iris does not yet support scale factors other than one for
        # Mercator projections
        cf_name = "mercator"
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=0,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=0.9,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_mercator_parameters(engine, cf_name)

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(str(warns[0]), "Scale factor")

    def test_invalid_standard_parallel(self):
        # Iris does not yet support standard parallels other than zero for
        # Mercator projections
        cf_name = "mercator"
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=0,
            false_easting=0,
            false_northing=0,
            standard_parallel=30,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_mercator_parameters(engine, cf_name)

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(str(warns[0]), "Standard parallel")

    def test_invalid_false_easting(self):
        # Iris does not yet support false eastings other than zero for
        # Mercator projections
        cf_name = "mercator"
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=0,
            false_easting=100,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_mercator_parameters(engine, cf_name)

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(str(warns[0]), "False easting")

    def test_invalid_false_northing(self):
        # Iris does not yet support false northings other than zero for
        # Mercator projections
        cf_name = "mercator"
        cf_grid_var = mock.Mock(
            spec=[],
            longitude_of_projection_origin=0,
            false_easting=0,
            false_northing=100,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = _engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_mercator_parameters(engine, cf_name)

        self.assertFalse(is_valid)
        self.assertEqual(len(warns), 1)
        self.assertRegex(str(warns[0]), "False northing")


if __name__ == "__main__":
    tests.main()

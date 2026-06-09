# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers.\
has_supported_mercator_parameters`.

"""

import re
import warnings

from iris.fileformats._nc_load_rules.helpers import has_supported_mercator_parameters
from iris.tests.unit.fileformats.nc_load_rules.helpers import MockerMixin


class _EngineMixin(MockerMixin):
    def engine(self, cf_grid_var, cf_name):
        cf_group = {cf_name: cf_grid_var}
        cf_var = self.mocker.Mock(cf_group=cf_group)
        return self.mocker.Mock(cf_var=cf_var)


class TestHasSupportedMercatorParameters(_EngineMixin):
    def test_valid_base(self, mocker):
        cf_name = "mercator"
        cf_grid_var = mocker.Mock(
            spec=[],
            longitude_of_projection_origin=-90,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = self.engine(cf_grid_var, cf_name)

        is_valid = has_supported_mercator_parameters(engine, cf_name)

        assert is_valid

    def test_valid_false_easting_northing(self, mocker):
        cf_name = "mercator"
        cf_grid_var = mocker.Mock(
            spec=[],
            longitude_of_projection_origin=-90,
            false_easting=15,
            false_northing=10,
            scale_factor_at_projection_origin=1,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = self.engine(cf_grid_var, cf_name)

        is_valid = has_supported_mercator_parameters(engine, cf_name)

        assert is_valid

    def test_valid_standard_parallel(self, mocker):
        cf_name = "mercator"
        cf_grid_var = mocker.Mock(
            spec=[],
            longitude_of_projection_origin=-90,
            false_easting=0,
            false_northing=0,
            standard_parallel=15,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = self.engine(cf_grid_var, cf_name)

        is_valid = has_supported_mercator_parameters(engine, cf_name)

        assert is_valid

    def test_valid_scale_factor(self, mocker):
        cf_name = "mercator"
        cf_grid_var = mocker.Mock(
            spec=[],
            longitude_of_projection_origin=0,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=0.9,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = self.engine(cf_grid_var, cf_name)

        is_valid = has_supported_mercator_parameters(engine, cf_name)

        assert is_valid

    def test_invalid_scale_factor_and_standard_parallel(self, mocker):
        # Scale factor and standard parallel cannot both be specified for
        # Mercator projections
        cf_name = "mercator"
        cf_grid_var = mocker.Mock(
            spec=[],
            longitude_of_projection_origin=0,
            false_easting=0,
            false_northing=0,
            scale_factor_at_projection_origin=0.9,
            standard_parallel=20,
            semi_major_axis=6377563.396,
            semi_minor_axis=6356256.909,
        )
        engine = self.engine(cf_grid_var, cf_name)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            is_valid = has_supported_mercator_parameters(engine, cf_name)

        assert not is_valid
        assert len(warns) == 1

        msg = re.escape(
            'both "scale_factor_at_projection_origin" and "standard_parallel"'
        )
        assert re.search(msg, str(warns[0]))

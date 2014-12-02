# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Unit tests for
:meth:`iris.fileformats.grib._save_rules.grid_definition_template_12`.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coord_systems import GeogCS, TransverseMercator
from iris.exceptions import TranslationError
from iris.tests.unit.fileformats.grib.save_rules import GdtTestMixin

from iris.fileformats.grib._save_rules import grid_definition_template_12


class Test(tests.IrisTest, GdtTestMixin):
    def setUp(self):
        self.default_ellipsoid = GeogCS(semi_major_axis=6377563.396,
                                        semi_minor_axis=6356256.909)
        self.default_cs = self._default_coord_system()
        self.test_cube = self._make_test_cube(cs=self.default_cs)
        GdtTestMixin.setUp(self)

    def _default_coord_system(self):
        cs = TransverseMercator(latitude_of_projection_origin=49.0,
                                longitude_of_central_meridian=-2.0,
                                false_easting=400000.0,
                                false_northing=-100000.0,
                                scale_factor_at_central_meridian=0.9996012717,
                                ellipsoid=self.default_ellipsoid)
        return cs

    def test__template_number(self):
        # A GRIBAPI bug in setting the key scaleFactorAtReferencePoint means
        # setting GDT12 can currently only raise an exception.
        with self.assertRaisesRegexp(TranslationError, "GRIBAPI error"):
            grid_definition_template_12(self.test_cube, self.mock_grib)


if __name__ == "__main__":
    tests.main()

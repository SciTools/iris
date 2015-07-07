# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for :class:`iris.experimental.regrid.CurvilinearRegridder`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.experimental.regrid import CurvilinearRegridder as Regridder
from iris.tests.stock import global_pp, lat_lon_cube


RESULT_DIR = ('analysis', 'regrid')


class Test__initialisation(tests.IrisTest):
    def setUp(self):
        self.ok = lat_lon_cube()
        self.bad = np.ones((3, 4))
        self.weights = np.ones(self.ok.shape, self.ok.dtype)

    def test_bad_src_type(self):
        with self.assertRaisesRegexp(TypeError, "'src_grid_cube'"):
            Regridder(self.bad, self.ok, self.weights)

    def test_bad_grid_type(self):
        with self.assertRaisesRegexp(TypeError, "'target_grid_cube'"):
            Regridder(self.ok, self.bad, self.weights)


class Test__operation(tests.IrisTest):
    def setUp(self):
        self.func = ('iris.experimental.regrid.'
                     'regrid_weighted_curvilinear_to_rectilinear')
        self.ok = global_pp()
        y = self.ok.coord('latitude')
        x = self.ok.coord('longitude')
        self.ok.remove_coord('latitude')
        self.ok.remove_coord('longitude')
        self.ok.add_aux_coord(y, 0)
        self.ok.add_aux_coord(x, 1)
        self.weights = np.ones(self.ok.shape, self.ok.dtype)

    def test__use_once(self):
        with mock.patch(self.func,
                        return_value=mock.sentinel.regridded) as clr:
            Regridder(self.ok, self.ok, self.weights)

        clr.assertCalledOnceWith(self.ok, self.weights, self.ok)

    def test__cache_regridder(self):
        regridder = Regridder(self.ok, self.ok, self.weights)
        src = self.ok
        src.data = np.ones(src.shape, src.dtype)
        with mock.patch(self.func,
                        return_value=mock.sentinel.regridded) as clr:
            res = regridder(src)

        clr.assertCalledOnceWith(src, self.weights, self.ok)
        self.assertIs(res, mock.sentinel.regridded)


class Test___call____bad_src(tests.IrisTest):
    def setUp(self):
        self.ok = global_pp()
        y = self.ok.coord('latitude')
        x = self.ok.coord('longitude')
        self.ok.remove_coord('latitude')
        self.ok.remove_coord('longitude')
        self.ok.add_aux_coord(y, 0)
        self.ok.add_aux_coord(x, 1)
        weights = np.ones(self.ok.shape, self.ok.dtype)
        self.regridder = Regridder(self.ok, self.ok, weights)

    def test_bad_src_type(self):
        with self.assertRaisesRegexp(TypeError, 'must be a Cube'):
            self.regridder(np.ones((3, 4)))

    def test_bad_src_shape(self):
        with self.assertRaisesRegexp(ValueError,
                                     'not defined on the same source grid'):
            self.regridder(self.ok[::2, ::2])

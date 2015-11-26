# (C) British Crown Copyright 2014 - 2015, Met Office
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
:func:`iris.fileformats.pp_rules._convert_scalar_vertical_coords`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coords import DimCoord, AuxCoord
from iris.aux_factory import HybridPressureFactory, HybridHeightFactory
from iris.fileformats.pp import SplittableInt, STASH
from iris.fileformats.pp_rules import Reference
from iris.tests.unit.fileformats import TestField


from iris.fileformats.pp_rules import _convert_scalar_vertical_coords


def _lbcode(value=None, ix=None, iy=None):
    if value is not None:
        result = SplittableInt(value, {'iy': slice(0, 2), 'ix': slice(2, 4)})
    else:
        # N.B. if 'value' is None, both ix and iy must be set.
        result = SplittableInt(10000 + 100 * ix + iy,
                               {'iy': slice(0, 2), 'ix': slice(2, 4)})
    return result


class TestLBVC001_Height(TestField):
    def _check_height(self, blev, stash,
                      expect_normal=True, expect_fixed_height=None):
        lbvc = 1
        lbcode = _lbcode(0)  # effectively unused in this case
        lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = \
            None, None, None, None, None, None
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_normal:
            expect_result = [
                (DimCoord(blev, standard_name='height', units='m',
                          attributes={'positive': 'up'}),
                 None)]
        elif expect_fixed_height:
            expect_result = [
                (DimCoord([expect_fixed_height], standard_name='height',
                          units='m', attributes={'positive': 'up'}),
                 None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal_height__present(self):
        self._check_height(blev=12.3, stash=STASH(1, 1, 1))

    def test_normal_height__absent(self):
        self._check_height(blev=-1, stash=STASH(1, 1, 1),
                           expect_normal=False)

    def test_implied_height_1m5(self):
        self._check_height(blev=75.2, stash=STASH(1, 3, 236),
                           expect_normal=False, expect_fixed_height=1.5)

    def test_implied_height_10m(self):
        self._check_height(blev=75.2, stash=STASH(1, 3, 225),
                           expect_normal=False, expect_fixed_height=10.0)


class TestLBVC002_Depth(TestField):
    def _check_depth(self, lbcode, lblev, brlev=0.0, brsvd1=0.0,
                     expect_bounds=True, expect_match=True):
        lbvc = 2
        lblev = 23.0
        blev = 123.4
        stash = STASH(1, 1, 1)
        bhlev, bhrlev, brsvd2 = None, None, None
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_result = [
                (DimCoord([lblev],
                          standard_name='model_level_number',
                          attributes={'positive': 'down'}), None)]
            if expect_bounds:
                expect_result.append(
                    (DimCoord(blev, standard_name='depth',
                              units='m',
                              bounds=[brsvd1, brlev],
                              attributes={'positive': 'down'}), None))
            else:
                expect_result.append(
                    (DimCoord(blev, standard_name='depth', units='m',
                              attributes={'positive': 'down'}), None))
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_unbounded(self):
        self._check_depth(_lbcode(1), lblev=23.0,
                          expect_bounds=False)

    def test_bounded(self):
        self._check_depth(_lbcode(1), lblev=23.0, brlev=22.5, brsvd1=23.5,
                          expect_bounds=True)

    def test_cross_section(self):
        self._check_depth(_lbcode(ix=1, iy=2), lblev=23.0,
                          expect_match=False)


class TestLBVC006_SoilLevel(TestField):
    def _check_soil_level(self, lbcode, expect_match=True):
        lbvc = 6
        lblev = 12.3
        brsvd1, brlev = 0, 0
        stash = STASH(1, 1, 1)
        blev, bhlev, bhrlev, brsvd2 = None, None, None, None
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_result = [
                (DimCoord([lblev], long_name='soil_model_level_number',
                          attributes={'positive': 'down'}), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_soil_level(_lbcode(0))

    def test_cross_section(self):
        self._check_soil_level(_lbcode(ix=1, iy=2), expect_match=False)


class TestLBVC006_SoilDepth(TestField):
    def _check_soil_depth(self, lbcode, expect_match=True):
        lbvc = 6
        blev = 0.05
        brsvd1, brlev = 0, 0.1
        stash = STASH(1, 1, 1)
        lblev, bhlev, bhrlev, brsvd2 = None, None, None, None
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_result = [
                (DimCoord([blev], standard_name='depth', units='m',
                          bounds=[[brsvd1, brlev]],
                          attributes={'positive': 'down'}), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_soil_depth(_lbcode(0))

    def test_cross_section(self):
        self._check_soil_depth(_lbcode(ix=1, iy=2), expect_match=False)


class TestLBVC008_Pressure(TestField):
    def _check_pressure(self, lbcode, expect_match=True):
        lbvc = 8
        blev = 250.3
        stash = STASH(1, 1, 1)
        lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = \
            None, None, None, None, None, None
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_result = [
                (DimCoord([blev], long_name='pressure', units='hPa'), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_pressure(_lbcode(0))

    def test_non_pressure_cross_section(self):
        self._check_pressure(_lbcode(ix=10, iy=11))

    def test_pressure_cross_section(self):
        self._check_pressure(_lbcode(ix=10, iy=1), expect_match=False)


class TestLBVC019_PotentialTemperature(TestField):
    def _check_potm(self, lbcode, expect_match=True):
        lbvc = 19
        blev = 130.6
        stash = STASH(1, 1, 1)
        lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = \
            None, None, None, None, None, None
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_result = [
                (DimCoord([blev], standard_name='air_potential_temperature',
                          units='K', attributes={'positive': 'up'}), None)]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_potm(_lbcode(0))

    def test_cross_section(self):
        self._check_potm(_lbcode(ix=10, iy=11), expect_match=False)


class TestLBVC009_HybridPressure(TestField):
    def test_valid(self, expect_match=True):
        lbvc = 9
        lblev = 37.0
        bhlev = 850.1  # pressure
        bhrlev, brsvd2 = 810.0, 875.0  # pressure bounds
        blev = 0.15  # sigma
        brlev, brsvd1 = 0.11, 0.19  # sigma bounds
        lbcode = _lbcode(0)  # unused
        stash = STASH(1, 1, 1)  # unused
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_coords_and_dims = [
                (DimCoord([37.0],
                          standard_name='model_level_number',
                          attributes={'positive': 'up'}), None),
                (DimCoord([850.1],
                          long_name='level_pressure',
                          units='Pa',
                          bounds=[810.0, 875.0]), None),
                (AuxCoord([0.15],
                          long_name='sigma',
                          bounds=[brlev, brsvd1]), None)]
            expect_factories = [(HybridPressureFactory,
                                 [{'long_name': 'level_pressure'},
                                  {'long_name': 'sigma'},
                                  Reference('surface_air_pressure')])]
        else:
            expect_coords_and_dims = []
            expect_factories = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims,
                                           expect_coords_and_dims)
        self.assertEqual(factories, expect_factories)


class TestLBVC065_HybridHeight(TestField):
    def test_valid(self, expect_match=True):
        lbvc = 65
        lblev = 37.0
        bhlev = 0.35  # sigma
        bhrlev, brsvd2 = 0.31, 0.39  # sigma bounds
        blev = 9596.3  # level_height
        brlev, brsvd1 = 9500.0, 9800.0  # level_height bounds
        lbcode = _lbcode(0)  # unused
        stash = STASH(1, 1, 1)  # unused
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        if expect_match:
            expect_coords_and_dims = [
                (DimCoord([37.0],
                          standard_name='model_level_number',
                          attributes={'positive': 'up'}), None),
                (DimCoord([9596.3],
                          long_name='level_height', units='m',
                          bounds=[brlev, brsvd1],
                          attributes={'positive': 'up'}), None),
                (AuxCoord([0.35],
                          long_name='sigma', bounds=[bhrlev, brsvd2]), None)]
            expect_factories = [(HybridHeightFactory,
                                 [{'long_name': 'level_height'},
                                  {'long_name': 'sigma'},
                                  Reference('orography')])]
        else:
            expect_coords_and_dims = []
            expect_factories = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims,
                                           expect_coords_and_dims)
        self.assertEqual(factories, expect_factories)


class TestLBVCxxx_Unhandled(TestField):
    def test_unknown_lbvc(self):
        lbvc = 999
        blev, lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = \
            None, None, None, None, None, None, None
        lbcode = _lbcode(0)  # unused
        stash = STASH(1, 1, 1)  # unused
        coords_and_dims, factories = _convert_scalar_vertical_coords(
            lbcode=lbcode, lbvc=lbvc, blev=blev, lblev=lblev, stash=stash,
            bhlev=bhlev, bhrlev=bhrlev, brsvd1=brsvd1, brsvd2=brsvd2,
            brlev=brlev)
        self.assertEqual(coords_and_dims, [])
        self.assertEqual(factories, [])


if __name__ == "__main__":
    tests.main()

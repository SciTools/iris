# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for
:func:`iris.fileformats.pp_load_rules._convert_vertical_coords`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import DimCoord, AuxCoord
from iris.aux_factory import HybridPressureFactory, HybridHeightFactory
from iris.fileformats.pp import SplittableInt, STASH
from iris.fileformats.pp_load_rules import Reference, _convert_vertical_coords
from iris.tests.unit.fileformats import TestField


def _lbcode(value=None, ix=None, iy=None):
    if value is not None:
        result = SplittableInt(value, {"iy": slice(0, 2), "ix": slice(2, 4)})
    else:
        # N.B. if 'value' is None, both ix and iy must be set.
        result = SplittableInt(
            10000 + 100 * ix + iy, {"iy": slice(0, 2), "ix": slice(2, 4)}
        )
    return result


class TestLBVC001_Height(TestField):
    def _check_height(
        self,
        blev,
        stash,
        expect_normal=True,
        expect_fixed_height=None,
        dim=None,
    ):
        lbvc = 1
        lbcode = _lbcode(0)  # effectively unused in this case
        lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        if expect_normal:
            expect_result = [
                (
                    DimCoord(
                        blev,
                        standard_name="height",
                        units="m",
                        attributes={"positive": "up"},
                    ),
                    dim,
                )
            ]
        elif expect_fixed_height:
            expect_result = [
                (
                    DimCoord(
                        expect_fixed_height,
                        standard_name="height",
                        units="m",
                        attributes={"positive": "up"},
                    ),
                    None,
                )
            ]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal_height__present(self):
        self._check_height(blev=12.3, stash=STASH(1, 1, 1))

    def test_normal_height__present_vector(self):
        data = [12.3, 123.4, 1234.5]
        dim = 0
        for blev in [data, np.asarray(data)]:
            for dim_i in [dim, (dim,)]:
                self._check_height(blev=blev, stash=STASH(1, 1, 1), dim=dim_i)

    def test_normal_height__absent(self):
        self._check_height(blev=-1, stash=STASH(1, 1, 1), expect_normal=False)

    def test_normal_height__absent_vector(self):
        data = [-1, -1, -1]
        dim = 1
        for blev in [data, np.asarray(data)]:
            for dim_i in [dim, (dim,)]:
                self._check_height(
                    blev=blev,
                    stash=STASH(1, 1, 1),
                    expect_normal=False,
                    dim=dim_i,
                )

    def test_normal_height__absent_mixed_vector(self):
        data = [-1, 12.3, -1, 123.4]
        dim = 2
        for blev in [data, np.asarray(data)]:
            for dim_i in [dim, (dim,)]:
                self._check_height(
                    blev=blev,
                    stash=STASH(1, 1, 1),
                    expect_normal=False,
                    dim=dim_i,
                )

    def test_implied_height_1m5(self):
        self._check_height(
            blev=75.2,
            stash=STASH(1, 3, 236),
            expect_normal=False,
            expect_fixed_height=1.5,
        )

    def test_implied_height_1m5__vector(self):
        data = [1, 2, 3, 4]
        dim = 3
        for blev in [data, np.asarray(data)]:
            for dim_i in [dim, (dim,)]:
                self._check_height(
                    blev=blev,
                    stash=STASH(1, 3, 236),
                    expect_normal=False,
                    expect_fixed_height=1.5,
                    dim=dim_i,
                )

    def test_implied_height_10m(self):
        self._check_height(
            blev=75.2,
            stash=STASH(1, 3, 225),
            expect_normal=False,
            expect_fixed_height=10.0,
        )

    def test_implied_height_10m__vector(self):
        data = list(range(10))
        dim = 4
        for blev in [data, np.asarray(data)]:
            for dim_i in [dim, (dim,)]:
                self._check_height(
                    blev=blev,
                    stash=STASH(1, 3, 225),
                    expect_normal=False,
                    expect_fixed_height=10.0,
                    dim=dim_i,
                )


class TestLBVC002_Depth(TestField):
    def _check_depth(
        self,
        lbcode,
        lblev=23.0,
        blev=123.4,
        brlev=0.0,
        brsvd1=0.0,
        expect_bounds=True,
        expect_match=True,
        expect_mixed=False,
        dim=None,
    ):
        lbvc = 2
        stash = STASH(1, 1, 1)
        bhlev, bhrlev, brsvd2 = None, None, None
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        if expect_match:
            expect_result = [
                (
                    DimCoord(
                        lblev,
                        standard_name="model_level_number",
                        attributes={"positive": "down"},
                        units="1",
                    ),
                    dim,
                )
            ]
            if expect_bounds:
                brsvd1 = np.atleast_1d(brsvd1)
                brlev = np.atleast_1d(brlev)
                if expect_mixed:
                    lower = np.where(brsvd1 == brlev, blev, brsvd1)
                    upper = np.where(brsvd1 == brlev, blev, brlev)
                else:
                    lower, upper = brsvd1, brlev
                bounds = np.vstack((lower, upper)).T
                expect_result.append(
                    (
                        DimCoord(
                            blev,
                            standard_name="depth",
                            units="m",
                            bounds=bounds,
                            attributes={"positive": "down"},
                        ),
                        dim,
                    )
                )
            else:
                expect_result.append(
                    (
                        DimCoord(
                            blev,
                            standard_name="depth",
                            units="m",
                            attributes={"positive": "down"},
                        ),
                        dim,
                    )
                )
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_unbounded(self):
        self._check_depth(_lbcode(1), lblev=23.0, expect_bounds=False)

    def test_unbounded__vector(self):
        lblev = [1, 2, 3]
        blev = [10, 20, 30]
        brsvd1 = [5, 15, 25]
        brlev = [5, 15, 25]
        self._check_depth(
            _lbcode(1),
            lblev=lblev,
            blev=blev,
            brsvd1=brsvd1,
            brlev=brlev,
            expect_bounds=False,
            dim=1,
        )

    def test_unbounded__vector_no_depth(self):
        lblev = [1, 2, 3]
        blev = [10, 20, 30]
        brsvd1 = [5, 15, 25]
        brlev = [5, 15, 666]  # not all equal or all unequal!
        self._check_depth(
            _lbcode(1),
            lblev=lblev,
            blev=blev,
            brsvd1=brsvd1,
            brlev=brlev,
            expect_mixed=True,
            dim=0,
        )

    def test_bounded(self):
        self._check_depth(
            _lbcode(1), lblev=23.0, brlev=22.5, brsvd1=23.5, expect_bounds=True
        )

    def test_bounded__vector(self):
        lblev = [1, 2, 3]
        blev = [10, 20, 30]
        brsvd1 = [5, 15, 25]
        brlev = [15, 25, 35]
        self._check_depth(
            _lbcode(1),
            lblev=lblev,
            blev=blev,
            brsvd1=brsvd1,
            brlev=brlev,
            expect_bounds=True,
            dim=1,
        )

    def test_cross_section(self):
        self._check_depth(_lbcode(ix=1, iy=2), lblev=23.0, expect_match=False)

    def test_cross_section__vector(self):
        lblev = [1, 2, 3]
        blev = [10, 20, 30]
        brsvd1 = [5, 15, 25]
        brlev = [15, 25, 35]
        self._check_depth(
            _lbcode(ix=1, iy=2),
            lblev=lblev,
            blev=blev,
            brsvd1=brsvd1,
            brlev=brlev,
            expect_match=False,
            dim=1,
        )


class TestLBVC006_SoilLevel(TestField):
    def _check_soil_level(
        self, lbcode, lblev=12.3, expect_match=True, dim=None
    ):
        lbvc = 6
        stash = STASH(1, 1, 1)
        brsvd1, brlev = 0, 0
        if hasattr(lblev, "__iter__"):
            brsvd1 = [0] * len(lblev)
            brlev = [0] * len(lblev)
        blev, bhlev, bhrlev, brsvd2 = None, None, None, None
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        expect_result = []
        if expect_match:
            coord = DimCoord(
                lblev,
                long_name="soil_model_level_number",
                attributes={"positive": "down"},
                units="1",
            )
            expect_result = [(coord, dim)]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_soil_level(_lbcode(0))

    def test_normal__vector(self):
        lblev = np.arange(10)
        self._check_soil_level(_lbcode(0), lblev=lblev, dim=0)

    def test_cross_section(self):
        self._check_soil_level(_lbcode(ix=1, iy=2), expect_match=False)

    def test_cross_section__vector(self):
        lblev = np.arange(10)
        self._check_soil_level(
            _lbcode(ix=1, iy=2), lblev=lblev, expect_match=False, dim=0
        )


class TestLBVC006_SoilDepth(TestField):
    def _check_soil_depth(
        self,
        lbcode,
        blev=0.05,
        brsvd1=0,
        brlev=0.1,
        expect_match=True,
        dim=None,
    ):
        lbvc = 6
        stash = STASH(1, 1, 1)
        lblev, bhlev, bhrlev, brsvd2 = None, None, None, None
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        expect_result = []
        if expect_match:
            coord = DimCoord(
                blev,
                standard_name="depth",
                bounds=np.vstack((brsvd1, brlev)).T,
                units="m",
                attributes={"positive": "down"},
            )
            expect_result = [(coord, dim)]
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_soil_depth(_lbcode(0))

    def test_normal__vector(self):
        points = np.arange(10)
        self._check_soil_depth(
            _lbcode(0), blev=points, brsvd1=points - 1, brlev=points + 1, dim=0
        )

    def test_bad_bounds(self):
        points = [-0.5, 0.5]
        lower = [-1, 1]
        upper = [-1, 1]
        self._check_soil_depth(
            _lbcode(0),
            blev=points,
            brsvd1=lower,
            brlev=upper,
            dim=0,
            expect_match=False,
        )

    def test_cross_section(self):
        self._check_soil_depth(_lbcode(ix=1, iy=2), expect_match=False)

    def test_cross_section__vector(self):
        points = np.arange(10)
        self._check_soil_depth(
            _lbcode(ix=1, iy=2),
            blev=points,
            brsvd1=points - 1,
            brlev=points + 1,
            expect_match=False,
            dim=0,
        )


class TestLBVC008_Pressure(TestField):
    def _check_pressure(self, lbcode, blev=250.3, expect_match=True, dim=None):
        lbvc = 8
        stash = STASH(1, 1, 1)
        lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        if expect_match:
            expect_result = [
                (DimCoord(blev, long_name="pressure", units="hPa"), dim)
            ]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_pressure(_lbcode(0))

    def test_normal__vector(self):
        blev = [10, 100, 1000, 10000]
        self._check_pressure(_lbcode(0), blev=blev, dim=2)

    def test_non_pressure_cross_section(self):
        self._check_pressure(_lbcode(ix=10, iy=11))

    def test_non_pressure_cross_section__vector(self):
        blev = np.arange(10)
        self._check_pressure(_lbcode(ix=10, iy=11), blev=blev, dim=0)

    def test_pressure_cross_section(self):
        self._check_pressure(_lbcode(ix=10, iy=1), expect_match=False)

    def test_pressure_cross_section__vector(self):
        blev = np.arange(10)
        self._check_pressure(
            _lbcode(ix=10, iy=1), blev=blev, dim=1, expect_match=False
        )


class TestLBVC019_PotentialTemperature(TestField):
    def _check_potm(self, lbcode, blev=130.6, expect_match=True, dim=None):
        lbvc = 19
        stash = STASH(1, 1, 1)
        lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        if expect_match:
            expect_result = [
                (
                    DimCoord(
                        blev,
                        standard_name="air_potential_temperature",
                        units="K",
                        attributes={"positive": "up"},
                    ),
                    dim,
                )
            ]
        else:
            expect_result = []
        self.assertCoordsAndDimsListsMatch(coords_and_dims, expect_result)
        self.assertEqual(factories, [])

    def test_normal(self):
        self._check_potm(_lbcode(0))

    def test_normal__vector(self):
        blev = list(range(10))
        self._check_potm(_lbcode(0), blev=blev, dim=0)

    def test_cross_section(self):
        self._check_potm(_lbcode(ix=10, iy=11), expect_match=False)

    def test_cross_section__vector(self):
        blev = np.arange(5) + 100
        self._check_potm(
            _lbcode(ix=10, iy=11), blev=blev, dim=1, expect_match=False
        )


class TestLBVC009_HybridPressure(TestField):
    def _check(
        self,
        lblev=37.0,
        bhlev=850.1,
        bhrlev=810.0,
        brsvd2=875.0,
        blev=0.15,
        brlev=0.11,
        brsvd1=0.19,
        expect_match=True,
        dim=None,
    ):
        lbvc = 9
        lbcode = _lbcode(0)  # unused
        stash = STASH(1, 1, 1)  # unused
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        expect_coords_and_dims = [
            (
                DimCoord(
                    lblev,
                    standard_name="model_level_number",
                    attributes={"positive": "up"},
                    units="1",
                ),
                dim,
            )
        ]

        bhrlev = np.atleast_1d(bhrlev)
        brsvd2 = np.atleast_1d(brsvd2)
        expect_coords_and_dims.append(
            (
                DimCoord(
                    bhlev,
                    long_name="level_pressure",
                    units="Pa",
                    bounds=np.vstack((bhrlev, brsvd2)).T,
                ),
                dim,
            )
        )
        brlev = np.atleast_1d(brlev)
        brsvd1 = np.atleast_1d(brsvd1)
        expect_coords_and_dims.append(
            (
                AuxCoord(
                    blev,
                    long_name="sigma",
                    bounds=np.vstack((brlev, brsvd1)).T,
                    units="1",
                ),
                dim,
            )
        )
        expect_factories = [
            (
                HybridPressureFactory,
                [
                    {"long_name": "level_pressure"},
                    {"long_name": "sigma"},
                    Reference("surface_air_pressure"),
                ],
            )
        ]
        self.assertCoordsAndDimsListsMatch(
            coords_and_dims, expect_coords_and_dims
        )
        self.assertEqual(factories, expect_factories)

    def test_normal(self):
        self._check()

    def test_normal__vector(self):
        lblev = list(range(3))
        bhlev = [10, 20, 30]
        bhrlev = [5, 15, 25]
        brsvd2 = [15, 25, 35]
        blev = [100, 200, 300]
        brlev = [50, 150, 250]
        brsvd1 = [150, 250, 350]
        self._check(
            lblev=lblev,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd2=brsvd2,
            blev=blev,
            brlev=brlev,
            brsvd1=brsvd1,
            dim=0,
        )


class TestLBVC065_HybridHeight(TestField):
    def _check(
        self,
        lblev=37.0,
        blev=9596.3,
        brlev=9500.0,
        brsvd1=9800.0,
        bhlev=0.35,
        bhrlev=0.31,
        brsvd2=0.39,
        dim=None,
    ):
        lbvc = 65
        lbcode = _lbcode(0)  # unused
        stash = STASH(1, 1, 1)  # unused
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
            dim=dim,
        )
        expect_coords_and_dims = [
            (
                DimCoord(
                    lblev,
                    standard_name="model_level_number",
                    attributes={"positive": "up"},
                    units="1",
                ),
                dim,
            )
        ]
        brlev = np.atleast_1d(brlev)
        brsvd1 = np.atleast_1d(brsvd1)
        expect_coords_and_dims.append(
            (
                DimCoord(
                    blev,
                    long_name="level_height",
                    units="m",
                    bounds=np.vstack((brlev, brsvd1)).T,
                    attributes={"positive": "up"},
                ),
                dim,
            )
        )
        bhrlev = np.atleast_1d(bhrlev)
        brsvd2 = np.atleast_1d(brsvd2)
        expect_coords_and_dims.append(
            (
                AuxCoord(
                    bhlev,
                    long_name="sigma",
                    bounds=np.vstack((bhrlev, brsvd2)).T,
                    units="1",
                ),
                dim,
            )
        )
        expect_factories = [
            (
                HybridHeightFactory,
                [
                    {"long_name": "level_height"},
                    {"long_name": "sigma"},
                    Reference("orography"),
                ],
            )
        ]
        self.assertCoordsAndDimsListsMatch(
            coords_and_dims, expect_coords_and_dims
        )
        self.assertEqual(factories, expect_factories)

    def test_normal(self):
        self._check()

    def test_normal__vector(self):
        npts = 5
        lblev = np.arange(npts)
        blev = np.arange(npts) + 10
        brlev = np.arange(npts) + 5
        brsvd1 = np.arange(npts) + 15
        bhlev = np.arange(npts) + 12
        bhrlev = np.arange(npts) + 6
        brsvd2 = np.arange(npts) + 18
        self._check(
            lblev=lblev,
            blev=blev,
            brlev=brlev,
            brsvd1=brsvd1,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd2=brsvd2,
            dim=1,
        )


class TestLBVCxxx_Unhandled(TestField):
    def test_unknown_lbvc(self):
        lbvc = 999
        blev, lblev, bhlev, bhrlev, brsvd1, brsvd2, brlev = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        lbcode = _lbcode(0)  # unused
        stash = STASH(1, 1, 1)  # unused
        coords_and_dims, factories = _convert_vertical_coords(
            lbcode=lbcode,
            lbvc=lbvc,
            blev=blev,
            lblev=lblev,
            stash=stash,
            bhlev=bhlev,
            bhrlev=bhrlev,
            brsvd1=brsvd1,
            brsvd2=brsvd2,
            brlev=brlev,
        )
        self.assertEqual(coords_and_dims, [])
        self.assertEqual(factories, [])


if __name__ == "__main__":
    tests.main()

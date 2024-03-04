# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.save` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import cf_units
import cftime
import numpy as np
import pytest

from iris.coords import CellMethod, DimCoord
from iris.fileformats._ff_cross_references import STASH_TRANS
import iris.fileformats.pp as pp
from iris.fileformats.pp_save_rules import _lbproc_rules, verify
import iris.tests.stock as stock


@pytest.mark.parametrize(
    "unit,modulus",
    [
        (cf_units.Unit("radians"), 2 * np.pi),
        (cf_units.Unit("degrees"), 360.0),
        (None, 360.0),
    ],
)
def test_grid_and_pole__scalar_dim_longitude(unit, modulus):
    cube = stock.lat_lon_cube()[:, -1:]
    assert cube.ndim == 2
    lon = cube.coord("longitude")
    lon.units = unit

    field = _pp_save_ppfield_values(cube)
    bdx = modulus
    assert field.bdx == bdx
    assert field.bzx == (lon.points[0] - bdx)
    assert field.lbnpt == lon.points.size


def _pp_save_ppfield_values(cube):
    """
    Emulate saving a cube as PP, and capture the resulting PP field values.

    """
    # Create a test object to stand in for a real PPField.
    pp_field = mock.MagicMock(spec=pp.PPField3)
    # Add minimal content required by the pp.save operation.
    pp_field.HEADER_DEFN = pp.PPField3.HEADER_DEFN
    # Save cube to a dummy file, mocking the internally created PPField
    with mock.patch("iris.fileformats.pp.PPField3", return_value=pp_field):
        target_filelike = mock.Mock(name="target")
        target_filelike.mode = "b"
        pp.save(cube, target_filelike)
    # Return pp-field mock with all the written properties
    return pp_field


class TestVertical(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()

    def test_pseudo_level(self):
        pseudo_level = 123
        coord = DimCoord(pseudo_level, long_name="pseudo_level", units="1")
        self.cube.add_aux_coord(coord)
        lbuser5_produced = _pp_save_ppfield_values(self.cube).lbuser[4]
        self.assertEqual(pseudo_level, lbuser5_produced)

    def test_soil_level(self):
        soil_level = 314
        coord = DimCoord(soil_level, long_name="soil_model_level_number")
        self.cube.add_aux_coord(coord)
        self.cube.standard_name = "moisture_content_of_soil_layer"
        field = _pp_save_ppfield_values(self.cube)
        self.assertEqual(field.lbvc, 6)
        self.assertEqual(field.lblev, soil_level)
        self.assertEqual(field.blev, soil_level)
        self.assertEqual(field.brsvd[0], 0)
        self.assertEqual(field.brlev, 0)

    def test_soil_depth(self):
        lower, point, upper = 1, 2, 3
        coord = DimCoord(point, standard_name="depth", bounds=[[lower, upper]])
        self.cube.add_aux_coord(coord)
        self.cube.standard_name = "moisture_content_of_soil_layer"
        field = _pp_save_ppfield_values(self.cube)
        self.assertEqual(field.lbvc, 6)
        self.assertEqual(field.lblev, 0)
        self.assertEqual(field.blev, point)
        self.assertEqual(field.brsvd[0], lower)
        self.assertEqual(field.brlev, upper)


class TestLbfcProduction(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()

    def check_cube_stash_yields_lbfc(self, stash, lbfc_expected):
        if stash:
            self.cube.attributes["STASH"] = stash
        lbfc_produced = _pp_save_ppfield_values(self.cube).lbfc
        self.assertEqual(lbfc_produced, lbfc_expected)

    def test_known_stash(self):
        stashcode_str = "m04s07i002"
        self.assertIn(stashcode_str, STASH_TRANS)
        self.check_cube_stash_yields_lbfc(stashcode_str, 359)

    def test_unknown_stash(self):
        stashcode_str = "m99s99i999"
        self.assertNotIn(stashcode_str, STASH_TRANS)
        self.check_cube_stash_yields_lbfc(stashcode_str, 0)

    def test_no_stash(self):
        self.assertNotIn("STASH", self.cube.attributes)
        self.check_cube_stash_yields_lbfc(None, 0)

    def check_cube_name_units_yields_lbfc(self, name, units, lbfc_expected):
        self.cube.rename(name)
        self.cube.units = units
        lbfc_produced = _pp_save_ppfield_values(self.cube).lbfc
        self.assertEqual(
            lbfc_produced,
            lbfc_expected,
            "Lbfc for ({!r} / {!r}) should be {:d}, "
            "got {:d}".format(name, units, lbfc_expected, lbfc_produced),
        )

    def test_name_units_to_lbfc(self):
        # Check LBFC value produced from name and units.
        self.check_cube_name_units_yields_lbfc("sea_ice_temperature", "K", 209)

    def test_bad_name_units_to_lbfc_0(self):
        # Check that badly-formed / unrecognised cases yield LBFC == 0.
        self.check_cube_name_units_yields_lbfc(
            "sea_ice_temperature", "degC", 0
        )
        self.check_cube_name_units_yields_lbfc("Junk_Name", "K", 0)


class TestLbsrceProduction(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()

    def check_cube_um_source_yields_lbsrce(
        self, source_str=None, um_version_str=None, lbsrce_expected=None
    ):
        if source_str is not None:
            self.cube.attributes["source"] = source_str
        if um_version_str is not None:
            self.cube.attributes["um_version"] = um_version_str
        lbsrce_produced = _pp_save_ppfield_values(self.cube).lbsrce
        self.assertEqual(lbsrce_produced, lbsrce_expected)

    def test_none(self):
        self.check_cube_um_source_yields_lbsrce(None, None, 0)

    def test_source_only_no_version(self):
        self.check_cube_um_source_yields_lbsrce(
            "Data from Met Office Unified Model", None, 1111
        )

    def test_source_only_with_version(self):
        self.check_cube_um_source_yields_lbsrce(
            "Data from Met Office Unified Model 12.17", None, 12171111
        )

    def test_um_version(self):
        self.check_cube_um_source_yields_lbsrce(
            "Data from Met Office Unified Model 12.17", "25.36", 25361111
        )


class Test_Save__LbprocProduction(tests.IrisTest):
    # This test class is a little different to the others.
    # If it called `pp.save` via `_pp_save_ppfield_values` it would run
    # `pp_save_rules.verify` and run all the save rules. As this class uses
    # a 3D cube with a time coord it would run the time rules, which would fail
    # because the mock object does not set up the `pp.lbtim` attribute
    # correctly (i.e. as a `SplittableInt` object).
    # To work around this we call the lbproc rules directly here.

    def setUp(self):
        self.cube = stock.realistic_3d()
        self.pp_field = mock.MagicMock(spec=pp.PPField3)
        self.pp_field.HEADER_DEFN = pp.PPField3.HEADER_DEFN
        self.patch("iris.fileformats.pp.PPField3", return_value=self.pp_field)

    def test_no_cell_methods(self):
        lbproc = _lbproc_rules(self.cube, self.pp_field).lbproc
        self.assertEqual(lbproc, 0)

    def test_mean(self):
        self.cube.cell_methods = (CellMethod("mean", "time", "1 hour"),)
        lbproc = _lbproc_rules(self.cube, self.pp_field).lbproc
        self.assertEqual(lbproc, 128)

    def test_minimum(self):
        self.cube.cell_methods = (CellMethod("minimum", "time", "1 hour"),)
        lbproc = _lbproc_rules(self.cube, self.pp_field).lbproc
        self.assertEqual(lbproc, 4096)

    def test_maximum(self):
        self.cube.cell_methods = (CellMethod("maximum", "time", "1 hour"),)
        lbproc = _lbproc_rules(self.cube, self.pp_field).lbproc
        self.assertEqual(lbproc, 8192)


class TestTimeMean(tests.IrisTest):
    """
    Tests that time mean cell method is converted to pp appropriately.

    Pattern is pairs of tests - one with time mean method, and one without, to
    show divergent behaviour.

    """

    def test_t1_time_mean(self):
        cube = _get_single_time_cube(set_time_mean=True)
        tc = cube.coord(axis="t")
        expected = tc.units.num2date(0)

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual = pp_field.t1

        self.assertEqual(expected, actual)

    def test_t1_no_time_mean(self):
        cube = _get_single_time_cube()
        tc = cube.coord(axis="t")
        expected = tc.units.num2date(15)

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual = pp_field.t1

        self.assertEqual(expected, actual)

    def test_t2_time_mean(self):
        cube = _get_single_time_cube(set_time_mean=True)
        tc = cube.coord(axis="t")
        expected = tc.units.num2date(30)

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual = pp_field.t2

        self.assertEqual(expected, actual)

    def test_t2_no_time_mean(self):
        cube = _get_single_time_cube(set_time_mean=False)
        expected = cftime.datetime(0, 0, 0, calendar=None, has_year_zero=True)

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual = pp_field.t2
        self.assertEqual(expected, actual)

    def test_lbft_no_forecast_time(self):
        # Different pattern here: checking that lbft hasn't been changed from
        # the default value.
        cube = _get_single_time_cube()
        mock_lbft = mock.sentinel.lbft

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            pp_field.lbft = mock_lbft
            verify(cube, pp_field)
        actual = pp_field.lbft

        assert mock_lbft is actual

    def test_lbtim_no_time_mean(self):
        cube = _get_single_time_cube()
        expected_ib = 0
        expected_ic = 2  # 360 day calendar

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual_ib = pp_field.lbtim.ib
        actual_ic = pp_field.lbtim.ic

        self.assertEqual(expected_ib, actual_ib)
        self.assertEqual(expected_ic, actual_ic)

    def test_lbtim_time_mean(self):
        cube = _get_single_time_cube(set_time_mean=True)
        expected_ib = 2  # Time mean
        expected_ic = 2  # 360 day calendar

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual_ib = pp_field.lbtim.ib
        actual_ic = pp_field.lbtim.ic

        self.assertEqual(expected_ib, actual_ib)
        self.assertEqual(expected_ic, actual_ic)

    def test_lbproc_no_time_mean(self):
        cube = _get_single_time_cube()
        expected = 0

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual = pp_field.lbproc

        self.assertEqual(expected, actual)

    def test_lbproc_time_mean(self):
        cube = _get_single_time_cube(set_time_mean=True)
        expected = 128

        with mock.patch(
            "iris.fileformats.pp.PPField3", autospec=True
        ) as pp_field:
            verify(cube, pp_field)
        actual = pp_field.lbproc

        self.assertEqual(expected, actual)


def _get_single_time_cube(set_time_mean=False):
    cube = stock.realistic_3d()[0:1, :, :]
    cube.remove_coord("time")
    cube.remove_coord("forecast_period")
    tc = DimCoord(
        points=[15],
        standard_name="time",
        units=cf_units.Unit("days since epoch", calendar="360_day"),
        bounds=[[0, 30]],
    )
    cube.add_dim_coord(tc, 0)
    if set_time_mean:
        cube.cell_methods = (CellMethod("mean", coords="time"),)
    return cube


if __name__ == "__main__":
    tests.main()

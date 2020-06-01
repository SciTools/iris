# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading and saving PP files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np
import os
from unittest import mock

from cf_units import Unit
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube
import iris.fileformats.pp
import iris.fileformats.pp_load_rules
from iris.fileformats.pp_save_rules import verify
from iris.exceptions import IgnoreCubeException
from iris.fileformats.pp import load_pairs_from_fields
import iris.util


class TestVertical(tests.IrisTest):
    def _test_coord(self, cube, point, bounds=None, **kwargs):
        coords = cube.coords(**kwargs)
        self.assertEqual(
            len(coords),
            1,
            "failed to find exactly one coord" " using: {}".format(kwargs),
        )
        self.assertEqual(coords[0].points, point)
        if bounds is not None:
            self.assertArrayEqual(coords[0].bounds, [bounds])

    @staticmethod
    def _mock_field(**kwargs):
        mock_data = np.zeros(1)
        mock_core_data = mock.MagicMock(return_value=mock_data)
        field = mock.MagicMock(
            lbuser=[0] * 7,
            lbrsvd=[0] * 4,
            brsvd=[0] * 4,
            brlev=0,
            t1=mock.MagicMock(year=1990, month=1, day=3),
            t2=mock.MagicMock(year=1990, month=1, day=3),
            core_data=mock_core_data,
            realised_dtype=mock_data.dtype,
        )
        field.configure_mock(**kwargs)
        return field

    def test_soil_level_round_trip(self):
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        soil_level = 1234
        field = self._mock_field(
            lbvc=6, lblev=soil_level, stash=iris.fileformats.pp.STASH(1, 0, 9)
        )
        load = mock.Mock(return_value=iter([field]))
        with mock.patch("iris.fileformats.pp.load", new=load) as load:
            cube = next(iris.fileformats.pp.load_cubes("DUMMY"))

        self.assertIn("soil", cube.standard_name)
        self._test_coord(cube, soil_level, long_name="soil_model_level_number")

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None] * 4
        field.brlev = None
        field = verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 6)
        self.assertEqual(field.lblev, soil_level)
        self.assertEqual(field.blev, soil_level)
        self.assertEqual(field.brsvd[0], 0)
        self.assertEqual(field.brlev, 0)

    def test_soil_depth_round_trip(self):
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        lower, point, upper = 1.2, 3.4, 5.6
        brsvd = [lower, 0, 0, 0]
        field = self._mock_field(
            lbvc=6,
            blev=point,
            brsvd=brsvd,
            brlev=upper,
            stash=iris.fileformats.pp.STASH(1, 0, 9),
        )
        load = mock.Mock(return_value=iter([field]))
        with mock.patch("iris.fileformats.pp.load", new=load) as load:
            cube = next(iris.fileformats.pp.load_cubes("DUMMY"))

        self.assertIn("soil", cube.standard_name)
        self._test_coord(
            cube, point, bounds=[lower, upper], standard_name="depth"
        )

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brlev = None
        field.brsvd = [None] * 4
        field = verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 6)
        self.assertEqual(field.blev, point)
        self.assertEqual(field.brsvd[0], lower)
        self.assertEqual(field.brlev, upper)

    def test_potential_temperature_level_round_trip(self):
        # Check save+load for data on 'potential temperature' levels.

        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        potm_value = 22.5
        field = self._mock_field(lbvc=19, blev=potm_value)
        load = mock.Mock(return_value=iter([field]))
        with mock.patch("iris.fileformats.pp.load", new=load):
            cube = next(iris.fileformats.pp.load_cubes("DUMMY"))

        self._test_coord(
            cube, potm_value, standard_name="air_potential_temperature"
        )

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field = verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        self.assertEqual(field.lbvc, 19)
        self.assertEqual(field.blev, potm_value)

    @staticmethod
    def _field_with_data(scale=1, **kwargs):
        x, y = 40, 30
        mock_data = np.arange(1200).reshape(y, x) * scale
        mock_core_data = mock.MagicMock(return_value=mock_data)
        field = mock.MagicMock(
            core_data=mock_core_data,
            realised_dtype=mock_data.dtype,
            lbcode=[1],
            lbnpt=x,
            lbrow=y,
            bzx=350,
            bdx=1.5,
            bzy=40,
            bdy=1.5,
            lbuser=[0] * 7,
            lbrsvd=[0] * 4,
            t1=mock.MagicMock(year=1990, month=1, day=3),
            t2=mock.MagicMock(year=1990, month=1, day=3),
        )

        field._x_coord_name = lambda: "longitude"
        field._y_coord_name = lambda: "latitude"
        field.coord_system = lambda: None
        field.configure_mock(**kwargs)
        return field

    def test_hybrid_pressure_round_trip(self):
        # Use pp.load_cubes() to convert fake PPFields into Cubes.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().

        # Make a fake reference surface field.
        pressure_field = self._field_with_data(
            10,
            stash=iris.fileformats.pp.STASH(1, 0, 409),
            lbuser=[0, 0, 0, 409, 0, 0, 0],
        )

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            lbvc=9,
            lblev=model_level,
            bhlev=delta,
            bhrlev=delta_lower,
            blev=sigma,
            brlev=sigma_lower,
            brsvd=[sigma_upper, delta_upper],
        )

        # Convert both fields to cubes.
        load = mock.Mock(return_value=iter([pressure_field, data_field]))
        with mock.patch("iris.fileformats.pp.load", new=load) as load:
            pressure_cube, data_cube = iris.fileformats.pp.load_cubes("DUMMY")

        # Check the reference surface cube looks OK.
        self.assertEqual(pressure_cube.standard_name, "surface_air_pressure")
        self.assertEqual(pressure_cube.units, "Pa")

        # Check the data cube is set up to use hybrid-pressure.
        self._test_coord(
            data_cube, model_level, standard_name="model_level_number"
        )
        self._test_coord(
            data_cube,
            delta,
            [delta_lower, delta_upper],
            long_name="level_pressure",
        )
        self._test_coord(
            data_cube, sigma, [sigma_lower, sigma_upper], long_name="sigma"
        )
        aux_factories = data_cube.aux_factories
        self.assertEqual(len(aux_factories), 1)
        surface_coord = aux_factories[0].dependencies["surface_air_pressure"]
        self.assertArrayEqual(
            surface_coord.points, np.arange(12000, step=10).reshape(30, 40)
        )

        # Now use the save rules to convert the Cubes back into PPFields.
        pressure_field = iris.fileformats.pp.PPField3()
        pressure_field.lbfc = 0
        pressure_field.lbvc = 0
        pressure_field.brsvd = [None, None]
        pressure_field.lbuser = [None] * 7
        pressure_field = verify(pressure_cube, pressure_field)

        data_field = iris.fileformats.pp.PPField3()
        data_field.lbfc = 0
        data_field.lbvc = 0
        data_field.brsvd = [None, None]
        data_field.lbuser = [None] * 7
        data_field = verify(data_cube, data_field)

        # The reference surface field should have STASH=409
        self.assertArrayEqual(
            pressure_field.lbuser, [None, None, None, 409, None, None, 1]
        )

        # Check the data field has the vertical coordinate as originally
        # specified.
        self.assertEqual(data_field.lbvc, 9)
        self.assertEqual(data_field.lblev, model_level)
        self.assertEqual(data_field.bhlev, delta)
        self.assertEqual(data_field.bhrlev, delta_lower)
        self.assertEqual(data_field.blev, sigma)
        self.assertEqual(data_field.brlev, sigma_lower)
        self.assertEqual(data_field.brsvd, [sigma_upper, delta_upper])

    def test_hybrid_pressure_with_duplicate_references(self):
        # Make a fake reference surface field.
        pressure_field = self._field_with_data(
            10,
            stash=iris.fileformats.pp.STASH(1, 0, 409),
            lbuser=[0, 0, 0, 409, 0, 0, 0],
        )

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            lbvc=9,
            lblev=model_level,
            bhlev=delta,
            bhrlev=delta_lower,
            blev=sigma,
            brlev=sigma_lower,
            brsvd=[sigma_upper, delta_upper],
        )

        # Convert both fields to cubes.
        load = mock.Mock(
            return_value=iter([data_field, pressure_field, pressure_field])
        )
        msg = "Multiple reference cubes for surface_air_pressure"
        with mock.patch(
            "iris.fileformats.pp.load", new=load
        ) as load, mock.patch("warnings.warn") as warn:
            _, _, _ = iris.fileformats.pp.load_cubes("DUMMY")
            warn.assert_called_with(msg)

    def test_hybrid_height_with_non_standard_coords(self):
        # Check the save rules are using the AuxFactory to find the
        # hybrid height coordinates and not relying on their names.
        ny, nx = 30, 40
        sigma_lower, sigma, sigma_upper = 0.75, 0.8, 0.75
        delta_lower, delta, delta_upper = 150, 200, 250

        cube = Cube(np.zeros((ny, nx)), "air_temperature")
        level_coord = AuxCoord(0, "model_level_number", units="1")
        cube.add_aux_coord(level_coord)
        delta_coord = AuxCoord(
            delta,
            bounds=[[delta_lower, delta_upper]],
            long_name="moog",
            units="m",
        )
        sigma_coord = AuxCoord(
            sigma,
            bounds=[[sigma_lower, sigma_upper]],
            long_name="mavis",
            units="1",
        )
        surface_altitude_coord = AuxCoord(
            np.zeros((ny, nx)), "surface_altitude", units="m"
        )
        cube.add_aux_coord(delta_coord)
        cube.add_aux_coord(sigma_coord)
        cube.add_aux_coord(surface_altitude_coord, (0, 1))
        cube.add_aux_factory(
            HybridHeightFactory(
                delta_coord, sigma_coord, surface_altitude_coord
            )
        )

        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None, None]
        field.lbuser = [None] * 7
        field = verify(cube, field)

        self.assertEqual(field.blev, delta)
        self.assertEqual(field.brlev, delta_lower)
        self.assertEqual(field.bhlev, sigma)
        self.assertEqual(field.bhrlev, sigma_lower)
        self.assertEqual(field.brsvd, [delta_upper, sigma_upper])

    def test_hybrid_pressure_with_non_standard_coords(self):
        # Check the save rules are using the AuxFactory to find the
        # hybrid pressure coordinates and not relying on their names.
        ny, nx = 30, 40
        sigma_lower, sigma, sigma_upper = 0.75, 0.8, 0.75
        delta_lower, delta, delta_upper = 0.15, 0.2, 0.25

        cube = Cube(np.zeros((ny, nx)), "air_temperature")
        level_coord = AuxCoord(0, "model_level_number", units="1")
        cube.add_aux_coord(level_coord)
        delta_coord = AuxCoord(
            delta,
            bounds=[[delta_lower, delta_upper]],
            long_name="moog",
            units="Pa",
        )
        sigma_coord = AuxCoord(
            sigma,
            bounds=[[sigma_lower, sigma_upper]],
            long_name="mavis",
            units="1",
        )
        surface_air_pressure_coord = AuxCoord(
            np.zeros((ny, nx)), "surface_air_pressure", units="Pa"
        )
        cube.add_aux_coord(delta_coord)
        cube.add_aux_coord(sigma_coord)
        cube.add_aux_coord(surface_air_pressure_coord, (0, 1))
        cube.add_aux_factory(
            HybridPressureFactory(
                delta_coord, sigma_coord, surface_air_pressure_coord
            )
        )

        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None, None]
        field.lbuser = [None] * 7
        field = verify(cube, field)

        self.assertEqual(field.bhlev, delta)
        self.assertEqual(field.bhrlev, delta_lower)
        self.assertEqual(field.blev, sigma)
        self.assertEqual(field.brlev, sigma_lower)
        self.assertEqual(field.brsvd, [sigma_upper, delta_upper])

    def test_hybrid_height_round_trip_no_reference(self):
        # Use pp.load_cubes() to convert fake PPFields into Cubes.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            lbvc=65,
            lblev=model_level,
            bhlev=sigma,
            bhrlev=sigma_lower,
            blev=delta,
            brlev=delta_lower,
            brsvd=[delta_upper, sigma_upper],
        )

        # Convert field to a cube.
        load = mock.Mock(return_value=iter([data_field]))
        with mock.patch(
            "iris.fileformats.pp.load", new=load
        ) as load, mock.patch("warnings.warn") as warn:
            (data_cube,) = iris.fileformats.pp.load_cubes("DUMMY")

        msg = (
            "Unable to create instance of HybridHeightFactory. "
            "The source data contains no field(s) for 'orography'."
        )
        warn.assert_called_once_with(msg)

        # Check the data cube is set up to use hybrid height.
        self._test_coord(
            data_cube, model_level, standard_name="model_level_number"
        )
        self._test_coord(
            data_cube,
            delta,
            [delta_lower, delta_upper],
            long_name="level_height",
        )
        self._test_coord(
            data_cube, sigma, [sigma_lower, sigma_upper], long_name="sigma"
        )
        # Check that no aux factory is created (due to missing
        # reference surface).
        aux_factories = data_cube.aux_factories
        self.assertEqual(len(aux_factories), 0)

        # Now use the save rules to convert the Cube back into a PPField.
        data_field = iris.fileformats.pp.PPField3()
        data_field.lbfc = 0
        data_field.lbvc = 0
        data_field.brsvd = [None, None]
        data_field.lbuser = [None] * 7
        data_field = verify(data_cube, data_field)

        # Check the data field has the vertical coordinate as originally
        # specified.
        self.assertEqual(data_field.lbvc, 65)
        self.assertEqual(data_field.lblev, model_level)
        self.assertEqual(data_field.bhlev, sigma)
        self.assertEqual(data_field.bhrlev, sigma_lower)
        self.assertEqual(data_field.blev, delta)
        self.assertEqual(data_field.brlev, delta_lower)
        self.assertEqual(data_field.brsvd, [delta_upper, sigma_upper])


class TestSaveLBFT(tests.IrisTest):
    def setUp(self):
        delta_start = 24
        delta_mid = 36
        self.delta_end = 369 * 24
        ref_offset = 10 * 24
        self.args = (delta_start, delta_mid, self.delta_end, ref_offset)

    def create_cube(self, fp_min, fp_mid, fp_max, ref_offset, season=None):
        cube = Cube(np.zeros((3, 4)))
        cube.add_aux_coord(
            AuxCoord(
                standard_name="forecast_period",
                units="hours",
                points=fp_mid,
                bounds=[fp_min, fp_max],
            )
        )
        cube.add_aux_coord(
            AuxCoord(
                standard_name="time",
                units="hours since epoch",
                points=ref_offset + fp_mid,
                bounds=[ref_offset + fp_min, ref_offset + fp_max],
            )
        )
        if season:
            cube.add_aux_coord(
                AuxCoord(long_name="clim_season", points=season)
            )
            cube.add_cell_method(CellMethod("DUMMY", "clim_season"))
        return cube

    def convert_cube_to_field(self, cube):
        # Use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.lbtim = 0
        field = verify(cube, field)
        return field

    def test_time_mean_from_forecast_period(self):
        cube = self.create_cube(24, 36, 48, 72)
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 48)

    def test_time_mean_from_forecast_reference_time(self):
        cube = Cube(np.zeros((3, 4)))
        cube.add_aux_coord(
            AuxCoord(
                standard_name="forecast_reference_time",
                units="hours since epoch",
                points=72,
            )
        )
        cube.add_aux_coord(
            AuxCoord(
                standard_name="time",
                units="hours since epoch",
                points=72 + 36,
                bounds=[72 + 24, 72 + 48],
            )
        )
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 48)

    def test_climatological_mean_single_year(self):
        cube = Cube(np.zeros((3, 4)))
        cube.add_aux_coord(
            AuxCoord(
                standard_name="forecast_period",
                units="hours",
                points=36,
                bounds=[24, 4 * 24],
            )
        )
        cube.add_aux_coord(
            AuxCoord(
                standard_name="time",
                units="hours since epoch",
                points=240 + 36,
                bounds=[240 + 24, 240 + 4 * 24],
            )
        )
        cube.add_aux_coord(AuxCoord(long_name="clim_season", points="DUMMY"))
        cube.add_cell_method(CellMethod("DUMMY", "clim_season"))
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, 4 * 24)

    def test_climatological_mean_multi_year_djf(self):
        cube = self.create_cube(*self.args, season="djf")
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, self.delta_end)

    def test_climatological_mean_multi_year_mam(self):
        cube = self.create_cube(*self.args, season="mam")
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, self.delta_end)

    def test_climatological_mean_multi_year_jja(self):
        cube = self.create_cube(*self.args, season="jja")
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, self.delta_end)

    def test_climatological_mean_multi_year_son(self):
        cube = self.create_cube(*self.args, season="son")
        field = self.convert_cube_to_field(cube)
        self.assertEqual(field.lbft, self.delta_end)


class TestCoordinateForms(tests.IrisTest):
    def _common(self, x_coord):
        nx = len(x_coord.points)
        ny = 2
        data = np.zeros((ny, nx), dtype=np.float32)
        test_cube = iris.cube.Cube(data)
        y0 = np.float32(20.5)
        dy = np.float32(3.72)
        y_coord = iris.coords.DimCoord.from_regular(
            zeroth=y0,
            step=dy,
            count=ny,
            standard_name="latitude",
            units="degrees_north",
        )
        test_cube.add_dim_coord(x_coord, 1)
        test_cube.add_dim_coord(y_coord, 0)
        # Write to a temporary PP file and read it back as a PPField
        with self.temp_filename(".pp") as pp_filepath:
            iris.save(test_cube, pp_filepath)
            pp_loader = iris.fileformats.pp.load(pp_filepath)
            pp_field = next(pp_loader)
        return pp_field

    def test_save_awkward_case_is_regular(self):
        # Check that specific "awkward" values still save in a regular form.
        nx = 3
        x0 = np.float32(355.626)
        dx = np.float32(0.0135)
        x_coord = iris.coords.DimCoord.from_regular(
            zeroth=x0,
            step=dx,
            count=nx,
            standard_name="longitude",
            units="degrees_east",
        )
        pp_field = self._common(x_coord)
        # Check that the result has the regular coordinates as expected.
        self.assertEqual(pp_field.bzx, x0)
        self.assertEqual(pp_field.bdx, dx)
        self.assertEqual(pp_field.lbnpt, nx)

    def test_save_irregular(self):
        # Check that a non-regular coordinate saves as expected.
        nx = 3
        x_values = [0.0, 1.1, 2.0]
        x_coord = iris.coords.DimCoord(
            x_values, standard_name="longitude", units="degrees_east"
        )
        pp_field = self._common(x_coord)
        # Check that the result has the regular/irregular Y and X as expected.
        self.assertEqual(pp_field.bdx, 0.0)
        self.assertArrayAllClose(pp_field.x, x_values)
        self.assertEqual(pp_field.lbnpt, nx)


@tests.skip_data
class TestLoadLittleendian(tests.IrisTest):
    def test_load_sample(self):
        file_path = tests.get_data_path(
            ("PP", "little_endian", "qrparm.orog.pp")
        )
        # Ensure it just loads.
        cube = iris.load_cube(file_path, "surface_altitude")
        self.assertEqual(cube.shape, (110, 160))

        # Check for sensible floating point numbers.
        def check_minmax(array, expect_min, expect_max):
            found = np.array([np.min(array), np.max(array)])
            expected = np.array([expect_min, expect_max])
            self.assertArrayAlmostEqual(found, expected, decimal=2)

        lons = cube.coord("grid_longitude").points
        lats = cube.coord("grid_latitude").points
        data = cube.data
        check_minmax(lons, 342.0, 376.98)
        check_minmax(lats, -10.48, 13.5)
        check_minmax(data, -30.48, 6029.1)


@tests.skip_data
class TestAsCubes(tests.IrisTest):
    def setUp(self):
        dpath = tests.get_data_path(
            ["PP", "meanMaxMin", "200806081200__qwpb.T24.pp"]
        )
        self.ppfs = iris.fileformats.pp.load(dpath)

    def test_pseudo_level_filter(self):
        chosen_ppfs = []
        for ppf in self.ppfs:
            if ppf.lbuser[4] == 3:
                chosen_ppfs.append(ppf)
        cubes_fields = list(load_pairs_from_fields(chosen_ppfs))
        self.assertEqual(len(cubes_fields), 8)

    def test_pseudo_level_filter_none(self):
        chosen_ppfs = []
        for ppf in self.ppfs:
            if ppf.lbuser[4] == 30:
                chosen_ppfs.append(ppf)
        cubes = list(load_pairs_from_fields(chosen_ppfs))
        self.assertEqual(len(cubes), 0)

    def test_as_pairs(self):
        cube_ppf_pairs = load_pairs_from_fields(self.ppfs)
        cubes = []
        for cube, ppf in cube_ppf_pairs:
            if ppf.lbuser[4] == 3:
                cube.attributes["pseudo level"] = ppf.lbuser[4]
                cubes.append(cube)
        for cube in cubes:
            self.assertEqual(cube.attributes["pseudo level"], 3)


class TestSaveLBPROC(tests.IrisTest):
    def create_cube(self, longitude_coord="longitude"):
        cube = Cube(np.zeros((2, 3, 4)))
        tunit = Unit("days since epoch", calendar="gregorian")
        tcoord = DimCoord(np.arange(2), standard_name="time", units=tunit)
        xcoord = DimCoord(
            np.arange(3), standard_name=longitude_coord, units="degrees"
        )
        ycoord = DimCoord(points=np.arange(4))
        cube.add_dim_coord(tcoord, 0)
        cube.add_dim_coord(xcoord, 1)
        cube.add_dim_coord(ycoord, 2)
        return cube

    def convert_cube_to_field(self, cube):
        field = iris.fileformats.pp.PPField3()
        field.lbvc = 0
        return verify(cube, field)

    def test_time_mean_only(self):
        cube = self.create_cube()
        cube.add_cell_method(CellMethod(method="mean", coords="time"))
        field = self.convert_cube_to_field(cube)
        self.assertEqual(int(field.lbproc), 128)

    def test_longitudinal_mean_only(self):
        cube = self.create_cube()
        cube.add_cell_method(CellMethod(method="mean", coords="longitude"))
        field = self.convert_cube_to_field(cube)
        self.assertEqual(int(field.lbproc), 64)

    def test_grid_longitudinal_mean_only(self):
        cube = self.create_cube(longitude_coord="grid_longitude")
        cube.add_cell_method(
            CellMethod(method="mean", coords="grid_longitude")
        )
        field = self.convert_cube_to_field(cube)
        self.assertEqual(int(field.lbproc), 64)

    def test_time_mean_and_zonal_mean(self):
        cube = self.create_cube()
        cube.add_cell_method(CellMethod(method="mean", coords="time"))
        cube.add_cell_method(CellMethod(method="mean", coords="longitude"))
        field = self.convert_cube_to_field(cube)
        self.assertEqual(int(field.lbproc), 192)


@tests.skip_data
class TestCallbackLoad(tests.IrisTest):
    def setUp(self):
        self.pass_name = "air_potential_temperature"

    def callback_wrapper(self):
        # Wrap the `iris.exceptions.IgnoreCubeException`-calling callback.
        def callback_ignore_cube_exception(cube, field, filename):
            if cube.name() != self.pass_name:
                raise IgnoreCubeException

        return callback_ignore_cube_exception

    def test_ignore_cube_callback(self):
        test_dataset = tests.get_data_path(
            ["PP", "globClim1", "dec_subset.pp"]
        )
        exception_callback = self.callback_wrapper()
        result_cubes = iris.load(test_dataset, callback=exception_callback)
        n_result_cubes = len(result_cubes)
        # We ignore all but one cube (the `air_potential_temperature` cube).
        self.assertEqual(n_result_cubes, 1)
        self.assertEqual(result_cubes[0].name(), self.pass_name)


@tests.skip_data
class TestZonalMeanBounds(tests.IrisTest):
    def test_mulitple_longitude(self):
        # test that bounds are set for a zonal mean file with many longitude
        # values
        orig_file = tests.get_data_path(("PP", "aPPglob1", "global.pp"))

        f = next(iris.fileformats.pp.load(orig_file))
        f.lbproc = 192  # time and zonal mean

        # Write out pp file
        temp_filename = iris.util.create_temp_filename(".pp")
        with open(temp_filename, "wb") as temp_fh:
            f.save(temp_fh)

        # Load pp file
        cube = iris.load_cube(temp_filename)

        self.assertTrue(cube.coord("longitude").has_bounds())

        os.remove(temp_filename)

    def test_singular_longitude(self):
        # test that bounds are set for a zonal mean file with a single
        # longitude value

        pp_file = tests.get_data_path(("PP", "zonal_mean", "zonal_mean.pp"))

        # Load pp file
        cube = iris.load_cube(pp_file)

        self.assertTrue(cube.coord("longitude").has_bounds())


@tests.skip_data
class TestLoadPartialMask(tests.IrisTest):
    def test_data(self):
        # Ensure that fields merge correctly where one has a mask and one
        # doesn't.
        filename = tests.get_data_path(["PP", "simple_pp", "partial_mask.pp"])

        expected_data = np.ma.masked_array(
            [[[0, 1], [11, 12]], [[99, 100], [-1, -1]]],
            [[[0, 0], [0, 0]], [[0, 0], [1, 1]]],
            dtype=np.int32,
        )
        cube = iris.load_cube(filename)

        self.assertEqual(expected_data.dtype, cube.data.dtype)
        self.assertMaskedArrayEqual(expected_data, cube.data, strict=False)


if __name__ == "__main__":
    tests.main()

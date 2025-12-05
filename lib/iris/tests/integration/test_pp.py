# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading and saving PP files."""

import re

from cf_units import Unit
import dask.array as da
import numpy as np
import pytest

from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube
from iris.exceptions import IgnoreCubeException
import iris.fileformats.pp
from iris.fileformats.pp import load_pairs_from_fields
import iris.fileformats.pp_load_rules
from iris.fileformats.pp_save_rules import verify
from iris.loading import _CONCRETE_DERIVED_LOADING
from iris.tests import _shared_utils
import iris.util
from iris.warnings import IrisUserWarning


class TestVertical:
    def _test_coord(self, cube, point, bounds=None, **kwargs):
        coords = cube.coords(**kwargs)
        assert len(coords) == 1, "failed to find exactly one coord using: {}".format(
            kwargs
        )
        assert coords[0].points == point
        if bounds is not None:
            _shared_utils.assert_array_equal(coords[0].bounds, [bounds])

    @staticmethod
    def _mock_field(mocker, **kwargs):
        mock_data = np.zeros(1)
        mock_core_data = mocker.MagicMock(return_value=mock_data)
        field = mocker.MagicMock(
            lbuser=[0] * 7,
            lbrsvd=[0] * 4,
            brsvd=[0] * 4,
            brlev=0,
            t1=mocker.MagicMock(year=1990, month=1, day=3),
            t2=mocker.MagicMock(year=1990, month=1, day=3),
            core_data=mock_core_data,
            realised_dtype=mock_data.dtype,
        )
        field.configure_mock(**kwargs)
        return field

    def test_soil_level_round_trip(self, mocker):
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        soil_level = 1234
        field = self._mock_field(
            mocker, lbvc=6, lblev=soil_level, stash=iris.fileformats.pp.STASH(1, 0, 9)
        )
        load = mocker.Mock(return_value=iter([field]))
        mocker.patch("iris.fileformats.pp.load", new=load)
        cube = next(iris.fileformats.pp.load_cubes("DUMMY"))

        assert "soil" in cube.standard_name
        self._test_coord(cube, soil_level, long_name="soil_model_level_number")

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None] * 4
        field.brlev = None
        field = verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        assert field.lbvc == 6
        assert field.lblev == soil_level
        assert field.blev == soil_level
        assert field.brsvd[0] == 0
        assert field.brlev == 0

    def test_soil_depth_round_trip(self, mocker):
        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        lower, point, upper = 1.2, 3.4, 5.6
        brsvd = [lower, 0, 0, 0]
        field = self._mock_field(
            mocker,
            lbvc=6,
            blev=point,
            brsvd=brsvd,
            brlev=upper,
            stash=iris.fileformats.pp.STASH(1, 0, 9),
        )
        load = mocker.Mock(return_value=iter([field]))
        mocker.patch("iris.fileformats.pp.load", new=load)
        cube = next(iris.fileformats.pp.load_cubes("DUMMY"))

        assert "soil" in cube.standard_name
        self._test_coord(cube, point, bounds=[lower, upper], standard_name="depth")

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brlev = None
        field.brsvd = [None] * 4
        field = verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        assert field.lbvc == 6
        assert field.blev == point
        assert field.brsvd[0] == lower
        assert field.brlev == upper

    def test_potential_temperature_level_round_trip(self, mocker):
        # Check save+load for data on 'potential temperature' levels.

        # Use pp.load_cubes() to convert a fake PPField into a Cube.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        potm_value = 22.5
        field = self._mock_field(mocker, lbvc=19, blev=potm_value)
        load = mocker.Mock(return_value=iter([field]))
        mocker.patch("iris.fileformats.pp.load", new=load)
        cube = next(iris.fileformats.pp.load_cubes("DUMMY"))

        self._test_coord(cube, potm_value, standard_name="air_potential_temperature")

        # Now use the save rules to convert the Cube back into a PPField.
        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field = verify(cube, field)

        # Check the vertical coordinate is as originally specified.
        assert field.lbvc == 19
        assert field.blev == potm_value

    @staticmethod
    def _field_with_data(mocker, scale=1, lazy=False, **kwargs):
        x, y = 40, 30
        if lazy:
            mock_data = da.arange(1200).reshape(y, x) * scale
        else:
            mock_data = np.arange(1200).reshape(y, x) * scale
        mock_core_data = mocker.MagicMock(return_value=mock_data)
        field = mocker.MagicMock(
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
            t1=mocker.MagicMock(year=1990, month=1, day=3),
            t2=mocker.MagicMock(year=1990, month=1, day=3),
        )

        field._x_coord_name = lambda: "longitude"
        field._y_coord_name = lambda: "latitude"
        field.coord_system = lambda: None
        field.configure_mock(**kwargs)
        return field

    def test_hybrid_pressure_round_trip(self, mocker):
        # Use pp.load_cubes() to convert fake PPFields into Cubes.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().

        # Make a fake reference surface field.
        pressure_field = self._field_with_data(
            mocker,
            10,
            stash=iris.fileformats.pp.STASH(1, 0, 409),
            lbuser=[0, 0, 0, 409, 0, 0, 0],
        )

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            mocker,
            lbvc=9,
            lblev=model_level,
            bhlev=delta,
            bhrlev=delta_lower,
            blev=sigma,
            brlev=sigma_lower,
            brsvd=[sigma_upper, delta_upper],
        )

        # Convert both fields to cubes.
        load = mocker.Mock(return_value=iter([pressure_field, data_field]))
        mocker.patch("iris.fileformats.pp.load", new=load)
        pressure_cube, data_cube = iris.fileformats.pp.load_cubes("DUMMY")

        # Check the reference surface cube looks OK.
        assert pressure_cube.standard_name == "surface_air_pressure"
        assert pressure_cube.units == "Pa"

        # Check the data cube is set up to use hybrid-pressure.
        self._test_coord(data_cube, model_level, standard_name="model_level_number")
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
        assert len(aux_factories) == 1
        surface_coord = aux_factories[0].dependencies["surface_air_pressure"]
        _shared_utils.assert_array_equal(
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
        _shared_utils.assert_array_equal(
            pressure_field.lbuser, [None, None, None, 409, None, None, 1]
        )

        # Check the data field has the vertical coordinate as originally
        # specified.
        assert data_field.lbvc == 9
        assert data_field.lblev == model_level
        assert data_field.bhlev == delta
        assert data_field.bhrlev == delta_lower
        assert data_field.blev == sigma
        assert data_field.brlev == sigma_lower
        assert data_field.brsvd == [sigma_upper, delta_upper]

    def test_hybrid_pressure_lazy_load(self, mocker):
        pressure_field = self._field_with_data(
            mocker,
            10,
            lazy=True,
            stash=iris.fileformats.pp.STASH(1, 0, 409),
            lbuser=[0, 0, 0, 409, 0, 0, 0],
        )

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            mocker,
            lbvc=9,
            lblev=model_level,
            bhlev=delta,
            bhrlev=delta_lower,
            blev=sigma,
            brlev=sigma_lower,
            brsvd=[sigma_upper, delta_upper],
        )

        # Convert both fields to cubes.
        load = mocker.Mock(return_value=iter([pressure_field, data_field]))
        mocker.patch("iris.fileformats.pp.load", new=load)
        pressure_cube, data_cube = iris.fileformats.pp.load_cubes("DUMMY")

        assert data_cube.coord("surface_air_pressure").has_lazy_points()

        # TODO: _CONCRETE_DERIVED_LOADING is a temporary fix, remove from test when a permanent fix exists
        load = mocker.Mock(return_value=iter([pressure_field, data_field]))
        mocker.patch("iris.fileformats.pp.load", new=load)
        with _CONCRETE_DERIVED_LOADING.context():
            _, realised_data_cube = iris.fileformats.pp.load_cubes("DUMMY")
        assert not realised_data_cube.coord("surface_air_pressure").has_lazy_points()

    def test_hybrid_pressure_with_duplicate_references(self, mocker):
        # Make a fake reference surface field.
        pressure_field = self._field_with_data(
            mocker,
            10,
            stash=iris.fileformats.pp.STASH(1, 0, 409),
            lbuser=[0, 0, 0, 409, 0, 0, 0],
        )

        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            mocker,
            lbvc=9,
            lblev=model_level,
            bhlev=delta,
            bhrlev=delta_lower,
            blev=sigma,
            brlev=sigma_lower,
            brsvd=[sigma_upper, delta_upper],
        )

        # Convert both fields to cubes.
        load = mocker.Mock(
            return_value=iter([data_field, pressure_field, pressure_field])
        )
        msg = "Multiple reference cubes for surface_air_pressure"
        mocker.patch("iris.fileformats.pp.load", new=load)
        with pytest.warns(IrisUserWarning, match=re.escape(msg)):
            _, _, _ = iris.fileformats.pp.load_cubes("DUMMY")

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
            HybridHeightFactory(delta_coord, sigma_coord, surface_altitude_coord)
        )

        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None, None]
        field.lbuser = [None] * 7
        field = verify(cube, field)

        assert field.blev == delta
        assert field.brlev == delta_lower
        assert field.bhlev == sigma
        assert field.bhrlev == sigma_lower
        assert field.brsvd == [delta_upper, sigma_upper]

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
            HybridPressureFactory(delta_coord, sigma_coord, surface_air_pressure_coord)
        )

        field = iris.fileformats.pp.PPField3()
        field.lbfc = 0
        field.lbvc = 0
        field.brsvd = [None, None]
        field.lbuser = [None] * 7
        field = verify(cube, field)

        assert field.bhlev == delta
        assert field.bhrlev == delta_lower
        assert field.blev == sigma
        assert field.brlev == sigma_lower
        assert field.brsvd == [sigma_upper, delta_upper]

    def test_hybrid_height_round_trip_no_reference(self, mocker):
        # Use pp.load_cubes() to convert fake PPFields into Cubes.
        # NB. Use MagicMock so that SplittableInt header items, such as
        # LBCODE, support len().
        # Make a fake data field which needs the reference surface.
        model_level = 5678
        sigma_lower, sigma, sigma_upper = 0.85, 0.9, 0.95
        delta_lower, delta, delta_upper = 0.05, 0.1, 0.15
        data_field = self._field_with_data(
            mocker,
            lbvc=65,
            lblev=model_level,
            bhlev=sigma,
            bhrlev=sigma_lower,
            blev=delta,
            brlev=delta_lower,
            brsvd=[delta_upper, sigma_upper],
        )

        # Convert field to a cube.
        load = mocker.Mock(return_value=iter([data_field]))
        msg = (
            "Unable to create instance of HybridHeightFactory. "
            "The source data contains no field(s) for 'orography'."
        )
        mocker.patch("iris.fileformats.pp.load", new=load)
        with pytest.warns(IrisUserWarning, match=re.escape(msg)):
            (data_cube,) = iris.fileformats.pp.load_cubes("DUMMY")

        # Check the data cube is set up to use hybrid height.
        self._test_coord(data_cube, model_level, standard_name="model_level_number")
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
        assert len(aux_factories) == 0

        # Now use the save rules to convert the Cube back into a PPField.
        data_field = iris.fileformats.pp.PPField3()
        data_field.lbfc = 0
        data_field.lbvc = 0
        data_field.brsvd = [None, None]
        data_field.lbuser = [None] * 7
        data_field = verify(data_cube, data_field)

        # Check the data field has the vertical coordinate as originally
        # specified.
        assert data_field.lbvc == 65
        assert data_field.lblev == model_level
        assert data_field.bhlev == sigma
        assert data_field.bhrlev == sigma_lower
        assert data_field.blev == delta
        assert data_field.brlev == delta_lower
        assert data_field.brsvd == [delta_upper, sigma_upper]


class TestSaveLBFT:
    @pytest.fixture(autouse=True)
    def _setup(self):
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
            cube.add_aux_coord(AuxCoord(long_name="clim_season", points=season))
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
        assert field.lbft == 48

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
        assert field.lbft == 48

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
        assert field.lbft == 4 * 24

    def test_climatological_mean_multi_year_djf(self):
        cube = self.create_cube(*self.args, season="djf")
        field = self.convert_cube_to_field(cube)
        assert field.lbft == self.delta_end

    def test_climatological_mean_multi_year_mam(self):
        cube = self.create_cube(*self.args, season="mam")
        field = self.convert_cube_to_field(cube)
        assert field.lbft == self.delta_end

    def test_climatological_mean_multi_year_jja(self):
        cube = self.create_cube(*self.args, season="jja")
        field = self.convert_cube_to_field(cube)
        assert field.lbft == self.delta_end

    def test_climatological_mean_multi_year_son(self):
        cube = self.create_cube(*self.args, season="son")
        field = self.convert_cube_to_field(cube)
        assert field.lbft == self.delta_end


class TestCoordinateForms:
    def _common(self, x_coord, tmp_path):
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
        pp_filepath = tmp_path / "test.pp"
        iris.save(test_cube, pp_filepath)
        pp_loader = iris.fileformats.pp.load(pp_filepath)
        pp_field = next(pp_loader)
        return pp_field

    def test_save_awkward_case_is_regular(self, tmp_path):
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
        pp_field = self._common(x_coord, tmp_path)
        # Check that the result has the regular coordinates as expected.
        assert pp_field.bzx == x0
        assert pp_field.bdx == dx
        assert pp_field.lbnpt == nx

    def test_save_irregular(self, tmp_path):
        # Check that a non-regular coordinate saves as expected.
        nx = 3
        x_values = [0.0, 1.1, 2.0]
        x_coord = iris.coords.DimCoord(
            x_values, standard_name="longitude", units="degrees_east"
        )
        pp_field = self._common(x_coord, tmp_path)
        # Check that the result has the regular/irregular Y and X as expected.
        assert pp_field.bdx == 0.0
        _shared_utils.assert_array_all_close(pp_field.x, x_values)
        assert pp_field.lbnpt == nx


@_shared_utils.skip_data
class TestLoadLittleendian:
    def test_load_sample(self):
        file_path = _shared_utils.get_data_path(
            ("PP", "little_endian", "qrparm.orog.pp")
        )
        # Ensure it just loads.
        cube = iris.load_cube(file_path, "surface_altitude")
        assert cube.shape == (110, 160)

        # Check for sensible floating point numbers.
        def check_minmax(array, expect_min, expect_max):
            found = np.array([np.min(array), np.max(array)])
            expected = np.array([expect_min, expect_max])
            _shared_utils.assert_array_almost_equal(found, expected, decimal=2)

        lons = cube.coord("grid_longitude").points
        lats = cube.coord("grid_latitude").points
        data = cube.data
        check_minmax(lons, 342.0, 376.98)
        check_minmax(lats, -10.48, 13.5)
        check_minmax(data, -30.48, 6029.1)


@_shared_utils.skip_data
class TestAsCubes:
    @pytest.fixture(autouse=True)
    def _setup(self):
        dpath = _shared_utils.get_data_path(
            ["PP", "meanMaxMin", "200806081200__qwpb.T24.pp"]
        )
        self.ppfs = iris.fileformats.pp.load(dpath)

    def test_pseudo_level_filter(self):
        chosen_ppfs = []
        for ppf in self.ppfs:
            if ppf.lbuser[4] == 3:
                chosen_ppfs.append(ppf)
        cubes_fields = list(load_pairs_from_fields(chosen_ppfs))
        assert len(cubes_fields) == 8

    def test_pseudo_level_filter_none(self):
        chosen_ppfs = []
        for ppf in self.ppfs:
            if ppf.lbuser[4] == 30:
                chosen_ppfs.append(ppf)
        cubes = list(load_pairs_from_fields(chosen_ppfs))
        assert len(cubes) == 0

    def test_as_pairs(self):
        cube_ppf_pairs = load_pairs_from_fields(self.ppfs)
        cubes = []
        for cube, ppf in cube_ppf_pairs:
            if ppf.lbuser[4] == 3:
                cube.attributes["pseudo level"] = ppf.lbuser[4]
                cubes.append(cube)
        for cube in cubes:
            assert cube.attributes["pseudo level"] == 3


class TestSaveLBPROC:
    def create_cube(self, longitude_coord="longitude"):
        cube = Cube(np.zeros((2, 3, 4)))
        tunit = Unit("days since epoch", calendar="standard")
        tcoord = DimCoord(np.arange(2), standard_name="time", units=tunit)
        xcoord = DimCoord(np.arange(3), standard_name=longitude_coord, units="degrees")
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
        assert int(field.lbproc) == 128

    def test_longitudinal_mean_only(self):
        cube = self.create_cube()
        cube.add_cell_method(CellMethod(method="mean", coords="longitude"))
        field = self.convert_cube_to_field(cube)
        assert int(field.lbproc) == 64

    def test_grid_longitudinal_mean_only(self):
        cube = self.create_cube(longitude_coord="grid_longitude")
        cube.add_cell_method(CellMethod(method="mean", coords="grid_longitude"))
        field = self.convert_cube_to_field(cube)
        assert int(field.lbproc) == 64

    def test_time_mean_and_zonal_mean(self):
        cube = self.create_cube()
        cube.add_cell_method(CellMethod(method="mean", coords="time"))
        cube.add_cell_method(CellMethod(method="mean", coords="longitude"))
        field = self.convert_cube_to_field(cube)
        assert int(field.lbproc) == 192


@_shared_utils.skip_data
class TestCallbackLoad:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pass_name = "air_potential_temperature"

    def callback_wrapper(self):
        # Wrap the `iris.exceptions.IgnoreCubeException`-calling callback.
        def callback_ignore_cube_exception(cube, field, filename):
            if cube.name() != self.pass_name:
                raise IgnoreCubeException

        return callback_ignore_cube_exception

    def test_ignore_cube_callback(self):
        test_dataset = _shared_utils.get_data_path(["PP", "globClim1", "dec_subset.pp"])
        exception_callback = self.callback_wrapper()
        result_cubes = iris.load(test_dataset, callback=exception_callback)
        n_result_cubes = len(result_cubes)
        # We ignore all but one cube (the `air_potential_temperature` cube).
        assert n_result_cubes == 1
        assert result_cubes[0].name() == self.pass_name


@_shared_utils.skip_data
class TestZonalMeanBounds:
    def test_mulitple_longitude(self, tmp_path):
        # test that bounds are set for a zonal mean file with many longitude
        # values
        orig_file = _shared_utils.get_data_path(("PP", "aPPglob1", "global.pp"))

        f = next(iris.fileformats.pp.load(orig_file))
        f.lbproc = 192  # time and zonal mean

        # Write out pp file
        temp_filename = tmp_path / "test.pp"
        with open(temp_filename, "wb") as temp_fh:
            f.save(temp_fh)

        # Load pp file
        cube = iris.load_cube(temp_filename)

        assert cube.coord("longitude").has_bounds()

    def test_singular_longitude(self):
        # test that bounds are set for a zonal mean file with a single
        # longitude value

        pp_file = _shared_utils.get_data_path(("PP", "zonal_mean", "zonal_mean.pp"))

        # Load pp file
        cube = iris.load_cube(pp_file)

        assert cube.coord("longitude").has_bounds()


@_shared_utils.skip_data
class TestLoadPartialMask:
    def test_data(self):
        # Ensure that fields merge correctly where one has a mask and one
        # doesn't.
        filename = _shared_utils.get_data_path(["PP", "simple_pp", "partial_mask.pp"])

        expected_data = np.ma.masked_array(
            [[[0, 1], [11, 12]], [[99, 100], [-1, -1]]],
            [[[0, 0], [0, 0]], [[0, 0], [1, 1]]],
            dtype=np.int32,
        )
        cube = iris.load_cube(filename)

        assert expected_data.dtype == cube.data.dtype
        _shared_utils.assert_masked_array_equal(expected_data, cube.data, strict=False)

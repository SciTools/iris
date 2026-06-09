# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the cf module."""

import contextlib
import io

import pytest

import iris
import iris.fileformats.cf as cf
from iris.tests import _shared_utils


class TestCaching:
    def test_cached(self, mocker):
        # Make sure attribute access to the underlying netCDF4.Variable
        # is cached.
        name = "foo"
        nc_var = mocker.MagicMock()
        cf_var = cf.CFAncillaryDataVariable(name, nc_var)
        assert nc_var.ncattrs.call_count == 1

        # Accessing a netCDF attribute should result in no further calls
        # to nc_var.ncattrs() and the creation of an attribute on the
        # cf_var.
        # NB. Can't use hasattr() because that triggers the attribute
        # to be created!
        assert "coordinates" not in cf_var.__dict__
        _ = cf_var.coordinates
        assert nc_var.ncattrs.call_count == 1
        assert "coordinates" in cf_var.__dict__

        # Trying again results in no change.
        _ = cf_var.coordinates
        assert nc_var.ncattrs.call_count == 1
        assert "coordinates" in cf_var.__dict__

        # Trying another attribute results in just a new attribute.
        assert "standard_name" not in cf_var.__dict__
        _ = cf_var.standard_name
        assert nc_var.ncattrs.call_count == 1
        assert "standard_name" in cf_var.__dict__


@_shared_utils.skip_data
class TestCFReader:
    @pytest.fixture(autouse=True)
    def _setup(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "rotated", "xyt", "small_rotPole_precipitation.nc")
        )
        self.cfr = cf.CFReader(filename)
        with self.cfr:
            yield

    def test_ancillary_variables_pass_0(self):
        assert self.cfr.cf_group.ancillary_variables == {}

    def test_auxiliary_coordinates_pass_0(self):
        assert sorted(self.cfr.cf_group.auxiliary_coordinates.keys()) == ["lat", "lon"]

        lat = self.cfr.cf_group["lat"]
        assert lat.shape == (190, 174)
        assert lat.dimensions == ("rlat", "rlon")
        assert lat.ndim == 2
        assert lat.cf_attrs() == (
            ("long_name", "latitude"),
            ("standard_name", "latitude"),
            ("units", "degrees_north"),
        )

        lon = self.cfr.cf_group["lon"]
        assert lon.shape == (190, 174)
        assert lon.dimensions == ("rlat", "rlon")
        assert lon.ndim == 2
        assert lon.cf_attrs() == (
            ("long_name", "longitude"),
            ("standard_name", "longitude"),
            ("units", "degrees_east"),
        )

    def test_bounds_pass_0(self):
        assert sorted(self.cfr.cf_group.bounds.keys()) == ["time_bnds"]

        time_bnds = self.cfr.cf_group["time_bnds"]
        assert time_bnds.shape == (4, 2)
        assert time_bnds.dimensions == ("time", "time_bnds")
        assert time_bnds.ndim == 2
        assert time_bnds.cf_attrs() == ()

    def test_coordinates_pass_0(self):
        assert sorted(self.cfr.cf_group.coordinates.keys()) == ["rlat", "rlon", "time"]

        rlat = self.cfr.cf_group["rlat"]
        assert rlat.shape == (190,)
        assert rlat.dimensions == ("rlat",)
        assert rlat.ndim == 1
        attr = []
        attr.append(("axis", "Y"))
        attr.append(("long_name", "rotated latitude"))
        attr.append(("standard_name", "grid_latitude"))
        attr.append(("units", "degrees"))
        assert rlat.cf_attrs() == tuple(attr)

        rlon = self.cfr.cf_group["rlon"]
        assert rlon.shape == (174,)
        assert rlon.dimensions == ("rlon",)
        assert rlon.ndim == 1
        attr = []
        attr.append(("axis", "X"))
        attr.append(("long_name", "rotated longitude"))
        attr.append(("standard_name", "grid_longitude"))
        attr.append(("units", "degrees"))
        assert rlon.cf_attrs() == tuple(attr)

        time = self.cfr.cf_group["time"]
        assert time.shape == (4,)
        assert time.dimensions == ("time",)
        assert time.ndim == 1
        attr = []
        attr.append(("axis", "T"))
        attr.append(("bounds", "time_bnds"))
        attr.append(("calendar", "gregorian"))
        attr.append(("long_name", "Julian Day"))
        attr.append(("units", "days since 1950-01-01 00:00:00.0"))
        assert time.cf_attrs() == tuple(attr)

    def test_data_pass_0(self):
        assert sorted(self.cfr.cf_group.data_variables.keys()) == ["pr"]

        data = self.cfr.cf_group["pr"]
        assert data.shape == (4, 190, 174)
        assert data.dimensions == ("time", "rlat", "rlon")
        assert data.ndim == 3
        attr = []
        attr.append(("_FillValue", 1e30))
        attr.append(("cell_methods", "time: mean"))
        attr.append(("coordinates", "lon lat"))
        attr.append(("grid_mapping", "rotated_pole"))
        attr.append(("long_name", "Precipitation"))
        attr.append(("missing_value", 1e30))
        attr.append(("standard_name", "precipitation_flux"))
        attr.append(("units", "kg m-2 s-1"))
        attr = tuple(attr)
        assert data.cf_attrs()[0][0] == attr[0][0]
        assert data.cf_attrs()[0][1] == pytest.approx(attr[0][1], abs=1.6e22)
        assert data.cf_attrs()[1:5] == attr[1:5]
        assert data.cf_attrs()[5][1] == pytest.approx(attr[5][1], abs=1.6e22)
        assert data.cf_attrs()[6:] == attr[6:]

    def test_formula_terms_pass_0(self):
        assert self.cfr.cf_group.formula_terms == {}

    def test_grid_mapping_pass_0(self):
        assert sorted(self.cfr.cf_group.grid_mappings.keys()) == ["rotated_pole"]

        rotated_pole = self.cfr.cf_group["rotated_pole"]
        assert rotated_pole.shape == ()
        assert rotated_pole.dimensions == ()
        assert rotated_pole.ndim == 0
        attr = []
        attr.append(("grid_mapping_name", "rotated_latitude_longitude"))
        attr.append(("grid_north_pole_latitude", 18.0))
        attr.append(("grid_north_pole_longitude", -140.75))
        assert rotated_pole.cf_attrs() == tuple(attr)

    def test_cell_measures_pass_0(self):
        assert self.cfr.cf_group.cell_measures == {}

    def test_global_attributes_pass_0(self):
        assert sorted(self.cfr.cf_group.global_attributes.keys()) == [
            "Conventions",
            "NCO",
            "experiment",
            "history",
            "institution",
            "source",
        ]

        assert self.cfr.cf_group.global_attributes["Conventions"] == "CF-1.0"
        assert self.cfr.cf_group.global_attributes["experiment"] == "ER3"
        assert self.cfr.cf_group.global_attributes["institution"] == "DMI"
        assert self.cfr.cf_group.global_attributes["source"] == "HIRHAM"

    def test_variable_cf_group_pass_0(self):
        assert sorted(self.cfr.cf_group["time"].cf_group.keys()) == ["time_bnds"]
        assert sorted(self.cfr.cf_group["pr"].cf_group.keys()) == [
            "lat",
            "lon",
            "rlat",
            "rlon",
            "rotated_pole",
            "time",
        ]

    def test_variable_attribute_touch_pass_0(self):
        lat = self.cfr.cf_group["lat"]

        assert lat.cf_attrs() == (
            ("long_name", "latitude"),
            ("standard_name", "latitude"),
            ("units", "degrees_north"),
        )
        assert lat.cf_attrs_used() == ()
        assert lat.cf_attrs_unused() == (
            ("long_name", "latitude"),
            ("standard_name", "latitude"),
            ("units", "degrees_north"),
        )

        # touch some variable attributes.
        lat.long_name
        lat.units
        assert lat.cf_attrs_used() == (
            ("long_name", "latitude"),
            ("units", "degrees_north"),
        )
        assert lat.cf_attrs_unused() == (("standard_name", "latitude"),)

        # clear the attribute touch history.
        lat.cf_attrs_reset()
        assert lat.cf_attrs_used() == ()
        assert lat.cf_attrs_unused() == (
            ("long_name", "latitude"),
            ("standard_name", "latitude"),
            ("units", "degrees_north"),
        )

    def test_destructor(self, tmp_path):
        """Test the destructor when reading the dataset fails.
        Related to issue #3312: previously, the `CFReader` would
        always call `close()` on its `_dataset` attribute, even if it
        didn't exist because opening the dataset had failed.
        """
        fn = tmp_path / "tmp.nc"
        with fn.open("wb+") as fh:
            fh.write(b"\x89HDF\r\n\x1a\nBroken file with correct signature")
            fh.flush()

            with io.StringIO() as buf:
                with contextlib.redirect_stderr(buf):
                    try:
                        _ = cf.CFReader(str(fn))
                    except OSError:
                        pass
                    try:
                        _ = iris.load_cubes(str(fn))
                    except OSError:
                        pass
                buf.seek(0)
                assert buf.read() == ""


@_shared_utils.skip_data
class TestLoad:
    def test_attributes_empty(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        cube = iris.load_cube(filename)
        assert cube.coord("time").attributes == {}

    def test_attributes_contain_positive(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        cube = iris.load_cube(filename)
        assert cube.coord("height").attributes["positive"] == "up"

    def test_attributes_populated(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "label_and_climate", "small_FC_167_mon_19601101.nc")
        )
        cube = iris.load_cube(filename, "air_temperature")
        assert sorted(cube.coord("longitude").attributes.items()) == [
            ("data_type", "float"),
            ("modulo", 360),
            ("topology", "circular"),
            ("valid_max", 359.0),
            ("valid_min", 0.0),
        ]

    def test_cell_methods(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        cube = iris.load_cube(filename)
        assert cube.cell_methods == (
            iris.coords.CellMethod(
                method="mean",
                coords=("time",),
                intervals=("6 minutes",),
                comments=(),
            ),
        )


@_shared_utils.skip_data
class TestClimatology:
    @pytest.fixture(autouse=True)
    def _setup(self):
        filename = _shared_utils.get_data_path(
            (
                "NetCDF",
                "label_and_climate",
                "A1B-99999a-river-sep-2070-2099.nc",
            )
        )
        self.cfr = cf.CFReader(filename)
        with self.cfr:
            yield

    def test_bounds(self):
        time = self.cfr.cf_group["temp_dmax_tmean_abs"].cf_group.coordinates["time"]
        climatology = time.cf_group.climatology
        assert len(climatology) == 1
        assert list(climatology.keys()) == ["climatology_bounds"]

        climatology_var = climatology["climatology_bounds"]
        assert climatology_var.ndim == 2
        assert climatology_var.shape == (1, 2)


@_shared_utils.skip_data
class TestLabels:
    @pytest.fixture(autouse=True)
    def _setup(self):
        filename = _shared_utils.get_data_path(
            (
                "NetCDF",
                "label_and_climate",
                "A1B-99999a-river-sep-2070-2099.nc",
            )
        )
        self.cfr_start = cf.CFReader(filename)

        filename = _shared_utils.get_data_path(
            ("NetCDF", "label_and_climate", "small_FC_167_mon_19601101.nc")
        )
        self.cfr_end = cf.CFReader(filename)

        with self.cfr_start:
            with self.cfr_end:
                yield

    def test_label_dim_start(self):
        cf_data_var = self.cfr_start.cf_group["temp_dmax_tmean_abs"]

        region_group = self.cfr_start.cf_group.labels["region_name"]
        assert sorted(self.cfr_start.cf_group.labels.keys()) == ["region_name"]
        assert sorted(cf_data_var.cf_group.labels.keys()) == ["region_name"]

        assert region_group.cf_label_dimensions(cf_data_var) == ("georegion",)
        assert region_group.cf_label_data(cf_data_var)[0] == "Anglian"

        cf_data_var = self.cfr_start.cf_group["cdf_temp_dmax_tmean_abs"]

        assert sorted(self.cfr_start.cf_group.labels.keys()) == ["region_name"]
        assert sorted(cf_data_var.cf_group.labels.keys()) == ["region_name"]

        assert region_group.cf_label_dimensions(cf_data_var) == ("georegion",)
        assert region_group.cf_label_data(cf_data_var)[0] == "Anglian"

    def test_label_dim_end(self):
        cf_data_var = self.cfr_end.cf_group["tas"]

        assert sorted(self.cfr_end.cf_group.labels.keys()) == [
            "experiment_id",
            "institution",
            "source",
        ]
        assert sorted(cf_data_var.cf_group.labels.keys()) == [
            "experiment_id",
            "institution",
            "source",
        ]

        assert self.cfr_end.cf_group.labels["experiment_id"].cf_label_dimensions(
            cf_data_var
        ) == ("ensemble",)
        assert (
            self.cfr_end.cf_group.labels["experiment_id"].cf_label_data(cf_data_var)[0]
            == "2005"
        )

        assert self.cfr_end.cf_group.labels["institution"].cf_label_dimensions(
            cf_data_var
        ) == ("ensemble",)
        assert (
            self.cfr_end.cf_group.labels["institution"].cf_label_data(cf_data_var)[0]
            == "ECMWF"
        )

        assert self.cfr_end.cf_group.labels["source"].cf_label_dimensions(
            cf_data_var
        ) == ("ensemble",)
        assert (
            self.cfr_end.cf_group.labels["source"].cf_label_data(cf_data_var)[0]
            == "IFS33R1/HOPE-E, Sys 1, Met 1, ENSEMBLES"
        )

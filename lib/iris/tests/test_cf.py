# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the cf module."""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import contextlib
import io
from unittest import mock

import pytest

import iris
import iris.fileformats.cf as cf


class TestCaching(tests.IrisTest):
    def test_cached(self):
        # Make sure attribute access to the underlying netCDF4.Variable
        # is cached.
        name = "foo"
        nc_var = mock.MagicMock()
        cf_var = cf.CFAncillaryDataVariable(name, nc_var)
        self.assertEqual(nc_var.ncattrs.call_count, 1)

        # Accessing a netCDF attribute should result in no further calls
        # to nc_var.ncattrs() and the creation of an attribute on the
        # cf_var.
        # NB. Can't use hasattr() because that triggers the attribute
        # to be created!
        self.assertTrue("coordinates" not in cf_var.__dict__)
        _ = cf_var.coordinates
        self.assertEqual(nc_var.ncattrs.call_count, 1)
        self.assertTrue("coordinates" in cf_var.__dict__)

        # Trying again results in no change.
        _ = cf_var.coordinates
        self.assertEqual(nc_var.ncattrs.call_count, 1)
        self.assertTrue("coordinates" in cf_var.__dict__)

        # Trying another attribute results in just a new attribute.
        self.assertTrue("standard_name" not in cf_var.__dict__)
        _ = cf_var.standard_name
        self.assertEqual(nc_var.ncattrs.call_count, 1)
        self.assertTrue("standard_name" in cf_var.__dict__)


@tests.skip_data
class TestCFReader(tests.IrisTest):
    @pytest.fixture(autouse=True)
    def set_up(self):
        filename = tests.get_data_path(
            ("NetCDF", "rotated", "xyt", "small_rotPole_precipitation.nc")
        )
        self.cfr = cf.CFReader(filename)
        with self.cfr:
            yield

    def test_ancillary_variables_pass_0(self):
        self.assertEqual(self.cfr.cf_group.ancillary_variables, {})

    def test_auxiliary_coordinates_pass_0(self):
        self.assertEqual(
            sorted(self.cfr.cf_group.auxiliary_coordinates.keys()),
            ["lat", "lon"],
        )

        lat = self.cfr.cf_group["lat"]
        self.assertEqual(lat.shape, (190, 174))
        self.assertEqual(lat.dimensions, ("rlat", "rlon"))
        self.assertEqual(lat.ndim, 2)
        self.assertEqual(
            lat.cf_attrs(),
            (
                ("long_name", "latitude"),
                ("standard_name", "latitude"),
                ("units", "degrees_north"),
            ),
        )

        lon = self.cfr.cf_group["lon"]
        self.assertEqual(lon.shape, (190, 174))
        self.assertEqual(lon.dimensions, ("rlat", "rlon"))
        self.assertEqual(lon.ndim, 2)
        self.assertEqual(
            lon.cf_attrs(),
            (
                ("long_name", "longitude"),
                ("standard_name", "longitude"),
                ("units", "degrees_east"),
            ),
        )

    def test_bounds_pass_0(self):
        self.assertEqual(sorted(self.cfr.cf_group.bounds.keys()), ["time_bnds"])

        time_bnds = self.cfr.cf_group["time_bnds"]
        self.assertEqual(time_bnds.shape, (4, 2))
        self.assertEqual(time_bnds.dimensions, ("time", "time_bnds"))
        self.assertEqual(time_bnds.ndim, 2)
        self.assertEqual(time_bnds.cf_attrs(), ())

    def test_coordinates_pass_0(self):
        self.assertEqual(
            sorted(self.cfr.cf_group.coordinates.keys()),
            ["rlat", "rlon", "time"],
        )

        rlat = self.cfr.cf_group["rlat"]
        self.assertEqual(rlat.shape, (190,))
        self.assertEqual(rlat.dimensions, ("rlat",))
        self.assertEqual(rlat.ndim, 1)
        attr = []
        attr.append(("axis", "Y"))
        attr.append(("long_name", "rotated latitude"))
        attr.append(("standard_name", "grid_latitude"))
        attr.append(("units", "degrees"))
        self.assertEqual(rlat.cf_attrs(), tuple(attr))

        rlon = self.cfr.cf_group["rlon"]
        self.assertEqual(rlon.shape, (174,))
        self.assertEqual(rlon.dimensions, ("rlon",))
        self.assertEqual(rlon.ndim, 1)
        attr = []
        attr.append(("axis", "X"))
        attr.append(("long_name", "rotated longitude"))
        attr.append(("standard_name", "grid_longitude"))
        attr.append(("units", "degrees"))
        self.assertEqual(rlon.cf_attrs(), tuple(attr))

        time = self.cfr.cf_group["time"]
        self.assertEqual(time.shape, (4,))
        self.assertEqual(time.dimensions, ("time",))
        self.assertEqual(time.ndim, 1)
        attr = []
        attr.append(("axis", "T"))
        attr.append(("bounds", "time_bnds"))
        attr.append(("calendar", "gregorian"))
        attr.append(("long_name", "Julian Day"))
        attr.append(("units", "days since 1950-01-01 00:00:00.0"))
        self.assertEqual(time.cf_attrs(), tuple(attr))

    def test_data_pass_0(self):
        self.assertEqual(sorted(self.cfr.cf_group.data_variables.keys()), ["pr"])

        data = self.cfr.cf_group["pr"]
        self.assertEqual(data.shape, (4, 190, 174))
        self.assertEqual(data.dimensions, ("time", "rlat", "rlon"))
        self.assertEqual(data.ndim, 3)
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
        self.assertEqual(data.cf_attrs()[0][0], attr[0][0])
        self.assertAlmostEqual(data.cf_attrs()[0][1], attr[0][1], delta=1.6e22)
        self.assertEqual(data.cf_attrs()[1:5], attr[1:5])
        self.assertAlmostEqual(data.cf_attrs()[5][1], attr[5][1], delta=1.6e22)
        self.assertEqual(data.cf_attrs()[6:], attr[6:])

    def test_formula_terms_pass_0(self):
        self.assertEqual(self.cfr.cf_group.formula_terms, {})

    def test_grid_mapping_pass_0(self):
        self.assertEqual(
            sorted(self.cfr.cf_group.grid_mappings.keys()), ["rotated_pole"]
        )

        rotated_pole = self.cfr.cf_group["rotated_pole"]
        self.assertEqual(rotated_pole.shape, ())
        self.assertEqual(rotated_pole.dimensions, ())
        self.assertEqual(rotated_pole.ndim, 0)
        attr = []
        attr.append(("grid_mapping_name", "rotated_latitude_longitude"))
        attr.append(("grid_north_pole_latitude", 18.0))
        attr.append(("grid_north_pole_longitude", -140.75))
        self.assertEqual(rotated_pole.cf_attrs(), tuple(attr))

    def test_cell_measures_pass_0(self):
        self.assertEqual(self.cfr.cf_group.cell_measures, {})

    def test_global_attributes_pass_0(self):
        self.assertEqual(
            sorted(self.cfr.cf_group.global_attributes.keys()),
            [
                "Conventions",
                "NCO",
                "experiment",
                "history",
                "institution",
                "source",
            ],
        )

        self.assertEqual(self.cfr.cf_group.global_attributes["Conventions"], "CF-1.0")
        self.assertEqual(self.cfr.cf_group.global_attributes["experiment"], "ER3")
        self.assertEqual(self.cfr.cf_group.global_attributes["institution"], "DMI")
        self.assertEqual(self.cfr.cf_group.global_attributes["source"], "HIRHAM")

    def test_variable_cf_group_pass_0(self):
        self.assertEqual(
            sorted(self.cfr.cf_group["time"].cf_group.keys()), ["time_bnds"]
        )
        self.assertEqual(
            sorted(self.cfr.cf_group["pr"].cf_group.keys()),
            ["lat", "lon", "rlat", "rlon", "rotated_pole", "time"],
        )

    def test_variable_attribute_touch_pass_0(self):
        lat = self.cfr.cf_group["lat"]

        self.assertEqual(
            lat.cf_attrs(),
            (
                ("long_name", "latitude"),
                ("standard_name", "latitude"),
                ("units", "degrees_north"),
            ),
        )
        self.assertEqual(lat.cf_attrs_used(), ())
        self.assertEqual(
            lat.cf_attrs_unused(),
            (
                ("long_name", "latitude"),
                ("standard_name", "latitude"),
                ("units", "degrees_north"),
            ),
        )

        # touch some variable attributes.
        lat.long_name
        lat.units
        self.assertEqual(
            lat.cf_attrs_used(),
            (("long_name", "latitude"), ("units", "degrees_north")),
        )
        self.assertEqual(lat.cf_attrs_unused(), (("standard_name", "latitude"),))

        # clear the attribute touch history.
        lat.cf_attrs_reset()
        self.assertEqual(lat.cf_attrs_used(), ())
        self.assertEqual(
            lat.cf_attrs_unused(),
            (
                ("long_name", "latitude"),
                ("standard_name", "latitude"),
                ("units", "degrees_north"),
            ),
        )

    def test_destructor(self):
        """Test the destructor when reading the dataset fails.
        Related to issue #3312: previously, the `CFReader` would
        always call `close()` on its `_dataset` attribute, even if it
        didn't exist because opening the dataset had failed.
        """
        with self.temp_filename(suffix=".nc") as fn:
            with open(fn, "wb+") as fh:
                fh.write(b"\x89HDF\r\n\x1a\nBroken file with correct signature")
                fh.flush()

                with io.StringIO() as buf:
                    with contextlib.redirect_stderr(buf):
                        try:
                            _ = cf.CFReader(fn)
                        except OSError:
                            pass
                        try:
                            _ = iris.load_cubes(fn)
                        except OSError:
                            pass
                    buf.seek(0)
                    self.assertMultiLineEqual("", buf.read())


@tests.skip_data
class TestLoad(tests.IrisTest):
    def test_attributes_empty(self):
        filename = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        cube = iris.load_cube(filename)
        self.assertEqual(cube.coord("time").attributes, {})

    def test_attributes_contain_positive(self):
        filename = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        cube = iris.load_cube(filename)
        self.assertEqual(cube.coord("height").attributes["positive"], "up")

    def test_attributes_populated(self):
        filename = tests.get_data_path(
            ("NetCDF", "label_and_climate", "small_FC_167_mon_19601101.nc")
        )
        cube = iris.load_cube(filename, "air_temperature")
        self.assertEqual(
            sorted(cube.coord("longitude").attributes.items()),
            [
                ("data_type", "float"),
                ("modulo", 360),
                ("topology", "circular"),
                ("valid_max", 359.0),
                ("valid_min", 0.0),
            ],
        )

    def test_cell_methods(self):
        filename = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
        cube = iris.load_cube(filename)
        self.assertEqual(
            cube.cell_methods,
            (
                iris.coords.CellMethod(
                    method="mean",
                    coords=("time",),
                    intervals=("6 minutes",),
                    comments=(),
                ),
            ),
        )


@tests.skip_data
class TestClimatology(tests.IrisTest):
    @pytest.fixture(autouse=True)
    def set_up(self):
        filename = tests.get_data_path(
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
        self.assertEqual(len(climatology), 1)
        self.assertEqual(list(climatology.keys()), ["climatology_bounds"])

        climatology_var = climatology["climatology_bounds"]
        self.assertEqual(climatology_var.ndim, 2)
        self.assertEqual(climatology_var.shape, (1, 2))


@tests.skip_data
class TestLabels(tests.IrisTest):
    @pytest.fixture(autouse=True)
    def set_up(self):
        filename = tests.get_data_path(
            (
                "NetCDF",
                "label_and_climate",
                "A1B-99999a-river-sep-2070-2099.nc",
            )
        )
        self.cfr_start = cf.CFReader(filename)

        filename = tests.get_data_path(
            ("NetCDF", "label_and_climate", "small_FC_167_mon_19601101.nc")
        )
        self.cfr_end = cf.CFReader(filename)

        with self.cfr_start:
            with self.cfr_end:
                yield

    def test_label_dim_start(self):
        cf_data_var = self.cfr_start.cf_group["temp_dmax_tmean_abs"]

        region_group = self.cfr_start.cf_group.labels["region_name"]
        self.assertEqual(sorted(self.cfr_start.cf_group.labels.keys()), ["region_name"])
        self.assertEqual(sorted(cf_data_var.cf_group.labels.keys()), ["region_name"])

        self.assertEqual(region_group.cf_label_dimensions(cf_data_var), ("georegion",))
        self.assertEqual(region_group.cf_label_data(cf_data_var)[0], "Anglian")

        cf_data_var = self.cfr_start.cf_group["cdf_temp_dmax_tmean_abs"]

        self.assertEqual(sorted(self.cfr_start.cf_group.labels.keys()), ["region_name"])
        self.assertEqual(sorted(cf_data_var.cf_group.labels.keys()), ["region_name"])

        self.assertEqual(region_group.cf_label_dimensions(cf_data_var), ("georegion",))
        self.assertEqual(region_group.cf_label_data(cf_data_var)[0], "Anglian")

    def test_label_dim_end(self):
        cf_data_var = self.cfr_end.cf_group["tas"]

        self.assertEqual(
            sorted(self.cfr_end.cf_group.labels.keys()),
            ["experiment_id", "institution", "source"],
        )
        self.assertEqual(
            sorted(cf_data_var.cf_group.labels.keys()),
            ["experiment_id", "institution", "source"],
        )

        self.assertEqual(
            self.cfr_end.cf_group.labels["experiment_id"].cf_label_dimensions(
                cf_data_var
            ),
            ("ensemble",),
        )
        self.assertEqual(
            self.cfr_end.cf_group.labels["experiment_id"].cf_label_data(cf_data_var)[0],
            "2005",
        )

        self.assertEqual(
            self.cfr_end.cf_group.labels["institution"].cf_label_dimensions(
                cf_data_var
            ),
            ("ensemble",),
        )
        self.assertEqual(
            self.cfr_end.cf_group.labels["institution"].cf_label_data(cf_data_var)[0],
            "ECMWF",
        )

        self.assertEqual(
            self.cfr_end.cf_group.labels["source"].cf_label_dimensions(cf_data_var),
            ("ensemble",),
        )
        self.assertEqual(
            self.cfr_end.cf_group.labels["source"].cf_label_data(cf_data_var)[0],
            "IFS33R1/HOPE-E, Sys 1, Met 1, ENSEMBLES",
        )


if __name__ == "__main__":
    tests.main()

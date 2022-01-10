# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the `iris.fileformats.cf.CFReader` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

from iris.fileformats.cf import CFReader


def netcdf_variable(
    name,
    dimensions,
    dtype,
    ancillary_variables=None,
    coordinates="",
    bounds=None,
    climatology=None,
    formula_terms=None,
    grid_mapping=None,
    cell_measures=None,
    standard_name=None,
):
    """Return a mock NetCDF4 variable."""
    ndim = 0
    if dimensions is not None:
        dimensions = dimensions.split()
        ndim = len(dimensions)
    else:
        dimensions = []
    ncvar = mock.Mock(
        name=name,
        dimensions=dimensions,
        ncattrs=mock.Mock(return_value=[]),
        ndim=ndim,
        dtype=dtype,
        ancillary_variables=ancillary_variables,
        coordinates=coordinates,
        bounds=bounds,
        climatology=climatology,
        formula_terms=formula_terms,
        grid_mapping=grid_mapping,
        cell_measures=cell_measures,
        standard_name=standard_name,
    )
    return ncvar


class Test_translate__global_attributes(tests.IrisTest):
    def setUp(self):
        ncvar = netcdf_variable("ncvar", "height", np.float64)
        ncattrs = mock.Mock(return_value=["dimensions"])
        getncattr = mock.Mock(return_value="something something_else")
        self.dataset = mock.Mock(
            file_format="NetCDF4",
            variables={"ncvar": ncvar},
            ncattrs=ncattrs,
            getncattr=getncattr,
        )

    def test_create_global_attributes(self):
        with mock.patch("netCDF4.Dataset", return_value=self.dataset):
            global_attrs = CFReader("dummy").cf_group.global_attributes
            self.assertEqual(
                global_attrs["dimensions"], "something something_else"
            )


class Test_translate__formula_terms(tests.IrisTest):
    def setUp(self):
        self.delta = netcdf_variable(
            "delta", "height", np.float64, bounds="delta_bnds"
        )
        self.delta_bnds = netcdf_variable(
            "delta_bnds", "height bnds", np.float
        )
        self.sigma = netcdf_variable(
            "sigma", "height", np.float64, bounds="sigma_bnds"
        )
        self.sigma_bnds = netcdf_variable(
            "sigma_bnds", "height bnds", np.float
        )
        self.orography = netcdf_variable("orography", "lat lon", np.float64)
        formula_terms = "a: delta b: sigma orog: orography"
        standard_name = "atmosphere_hybrid_height_coordinate"
        self.height = netcdf_variable(
            "height",
            "height",
            np.float64,
            formula_terms=formula_terms,
            bounds="height_bnds",
            standard_name=standard_name,
        )
        # Over-specify the formula terms on the bounds variable,
        # which will be ignored by the cf loader.
        formula_terms = "a: delta_bnds b: sigma_bnds orog: orography"
        self.height_bnds = netcdf_variable(
            "height_bnds",
            "height bnds",
            np.float64,
            formula_terms=formula_terms,
        )
        self.lat = netcdf_variable("lat", "lat", np.float64)
        self.lon = netcdf_variable("lon", "lon", np.float64)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable(
            "temp", "height lat lon", np.float64, coordinates="lat lon"
        )

        self.variables = dict(
            delta=self.delta,
            sigma=self.sigma,
            orography=self.orography,
            height=self.height,
            lat=self.lat,
            lon=self.lon,
            temp=self.temp,
            delta_bnds=self.delta_bnds,
            sigma_bnds=self.sigma_bnds,
            height_bnds=self.height_bnds,
        )
        ncattrs = mock.Mock(return_value=[])
        self.dataset = mock.Mock(
            file_format="NetCDF4", variables=self.variables, ncattrs=ncattrs
        )
        # Restrict the CFReader functionality to only performing translations.
        build_patch = mock.patch(
            "iris.fileformats.cf.CFReader._build_cf_groups"
        )
        reset_patch = mock.patch("iris.fileformats.cf.CFReader._reset")
        build_patch.start()
        reset_patch.start()
        self.addCleanup(build_patch.stop)
        self.addCleanup(reset_patch.stop)

    def test_create_formula_terms(self):
        with mock.patch("netCDF4.Dataset", return_value=self.dataset):
            cf_group = CFReader("dummy").cf_group
            self.assertEqual(len(cf_group), len(self.variables))
            # Check there is a singular data variable.
            group = cf_group.data_variables
            self.assertEqual(len(group), 1)
            self.assertEqual(list(group.keys()), ["temp"])
            self.assertIs(group["temp"].cf_data, self.temp)
            # Check there are three coordinates.
            group = cf_group.coordinates
            self.assertEqual(len(group), 3)
            coordinates = ["height", "lat", "lon"]
            self.assertEqual(set(group.keys()), set(coordinates))
            for name in coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check there are three auxiliary coordinates.
            group = cf_group.auxiliary_coordinates
            self.assertEqual(len(group), 3)
            aux_coordinates = ["delta", "sigma", "orography"]
            self.assertEqual(set(group.keys()), set(aux_coordinates))
            for name in aux_coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check all the auxiliary coordinates are formula terms.
            formula_terms = cf_group.formula_terms
            self.assertEqual(set(group.items()), set(formula_terms.items()))
            # Check there are three bounds.
            group = cf_group.bounds
            self.assertEqual(len(group), 3)
            bounds = ["height_bnds", "delta_bnds", "sigma_bnds"]
            self.assertEqual(set(group.keys()), set(bounds))
            for name in bounds:
                self.assertEqual(group[name].cf_data, getattr(self, name))


class Test_build_cf_groups__formula_terms(tests.IrisTest):
    def setUp(self):
        self.delta = netcdf_variable(
            "delta", "height", np.float64, bounds="delta_bnds"
        )
        self.delta_bnds = netcdf_variable(
            "delta_bnds", "height bnds", np.float
        )
        self.sigma = netcdf_variable(
            "sigma", "height", np.float64, bounds="sigma_bnds"
        )
        self.sigma_bnds = netcdf_variable(
            "sigma_bnds", "height bnds", np.float
        )
        self.orography = netcdf_variable("orography", "lat lon", np.float64)
        formula_terms = "a: delta b: sigma orog: orography"
        standard_name = "atmosphere_hybrid_height_coordinate"
        self.height = netcdf_variable(
            "height",
            "height",
            np.float64,
            formula_terms=formula_terms,
            bounds="height_bnds",
            standard_name=standard_name,
        )
        # Over-specify the formula terms on the bounds variable,
        # which will be ignored by the cf loader.
        formula_terms = "a: delta_bnds b: sigma_bnds orog: orography"
        self.height_bnds = netcdf_variable(
            "height_bnds",
            "height bnds",
            np.float64,
            formula_terms=formula_terms,
        )
        self.lat = netcdf_variable("lat", "lat", np.float64)
        self.lon = netcdf_variable("lon", "lon", np.float64)
        self.x = netcdf_variable("x", "lat lon", np.float64)
        self.y = netcdf_variable("y", "lat lon", np.float64)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable(
            "temp", "height lat lon", np.float64, coordinates="x y"
        )

        self.variables = dict(
            delta=self.delta,
            sigma=self.sigma,
            orography=self.orography,
            height=self.height,
            lat=self.lat,
            lon=self.lon,
            temp=self.temp,
            delta_bnds=self.delta_bnds,
            sigma_bnds=self.sigma_bnds,
            height_bnds=self.height_bnds,
            x=self.x,
            y=self.y,
        )
        ncattrs = mock.Mock(return_value=[])
        self.dataset = mock.Mock(
            file_format="NetCDF4", variables=self.variables, ncattrs=ncattrs
        )
        # Restrict the CFReader functionality to only performing translations
        # and building first level cf-groups for variables.
        patcher = mock.patch("iris.fileformats.cf.CFReader._reset")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_associate_formula_terms_with_data_variable(self):
        with mock.patch("netCDF4.Dataset", return_value=self.dataset):
            cf_group = CFReader("dummy").cf_group
            self.assertEqual(len(cf_group), len(self.variables))
            # Check the cf-group associated with the data variable.
            temp_cf_group = cf_group["temp"].cf_group
            # Check the data variable is associated with eight variables.
            self.assertEqual(len(temp_cf_group), 8)
            # Check there are three coordinates.
            group = temp_cf_group.coordinates
            self.assertEqual(len(group), 3)
            coordinates = ["height", "lat", "lon"]
            self.assertEqual(set(group.keys()), set(coordinates))
            for name in coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check the height coordinate is bounded.
            group = group["height"].cf_group
            self.assertEqual(len(group.bounds), 1)
            self.assertIn("height_bnds", group.bounds)
            self.assertIs(group["height_bnds"].cf_data, self.height_bnds)
            # Check there are five auxiliary coordinates.
            group = temp_cf_group.auxiliary_coordinates
            self.assertEqual(len(group), 5)
            aux_coordinates = ["delta", "sigma", "orography", "x", "y"]
            self.assertEqual(set(group.keys()), set(aux_coordinates))
            for name in aux_coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check all the auxiliary coordinates are formula terms.
            formula_terms = cf_group.formula_terms
            self.assertTrue(
                set(formula_terms.items()).issubset(list(group.items()))
            )
            # Check the terms by root.
            for name, term in zip(aux_coordinates, ["a", "b", "orog"]):
                self.assertEqual(
                    formula_terms[name].cf_terms_by_root, dict(height=term)
                )
            # Check the bounded auxiliary coordinates.
            for name, name_bnds in zip(
                ["delta", "sigma"], ["delta_bnds", "sigma_bnds"]
            ):
                aux_coord_group = group[name].cf_group
                self.assertEqual(len(aux_coord_group.bounds), 1)
                self.assertIn(name_bnds, aux_coord_group.bounds)
                self.assertIs(
                    aux_coord_group[name_bnds].cf_data,
                    getattr(self, name_bnds),
                )

    def test_promote_reference(self):
        with mock.patch("netCDF4.Dataset", return_value=self.dataset):
            cf_group = CFReader("dummy").cf_group
            self.assertEqual(len(cf_group), len(self.variables))
            # Check the number of data variables.
            self.assertEqual(len(cf_group.data_variables), 1)
            self.assertEqual(list(cf_group.data_variables.keys()), ["temp"])
            # Check the number of promoted variables.
            self.assertEqual(len(cf_group.promoted), 1)
            self.assertEqual(list(cf_group.promoted.keys()), ["orography"])
            # Check the promoted variable dependencies.
            group = cf_group.promoted["orography"].cf_group.coordinates
            self.assertEqual(len(group), 2)
            coordinates = ("lat", "lon")
            self.assertEqual(set(group.keys()), set(coordinates))
            for name in coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))

    def test_formula_terms_ignore(self):
        self.orography.dimensions = ["lat", "wibble"]
        with mock.patch(
            "netCDF4.Dataset", return_value=self.dataset
        ), mock.patch("warnings.warn") as warn:
            cf_group = CFReader("dummy").cf_group
            group = cf_group.promoted
            self.assertEqual(list(group.keys()), ["orography"])
            self.assertIs(group["orography"].cf_data, self.orography)
            self.assertEqual(warn.call_count, 1)

    def test_auxiliary_ignore(self):
        self.x.dimensions = ["lat", "wibble"]
        with mock.patch(
            "netCDF4.Dataset", return_value=self.dataset
        ), mock.patch("warnings.warn") as warn:
            cf_group = CFReader("dummy").cf_group
            promoted = ["x", "orography"]
            group = cf_group.promoted
            self.assertEqual(set(group.keys()), set(promoted))
            for name in promoted:
                self.assertIs(group[name].cf_data, getattr(self, name))
            self.assertEqual(warn.call_count, 1)

    def test_promoted_auxiliary_ignore(self):
        self.wibble = netcdf_variable("wibble", "lat wibble", np.float64)
        self.variables["wibble"] = self.wibble
        self.orography.coordinates = "wibble"
        with mock.patch(
            "netCDF4.Dataset", return_value=self.dataset
        ), mock.patch("warnings.warn") as warn:
            cf_group = CFReader("dummy").cf_group.promoted
            promoted = ["wibble", "orography"]
            self.assertEqual(set(cf_group.keys()), set(promoted))
            for name in promoted:
                self.assertIs(cf_group[name].cf_data, getattr(self, name))
            self.assertEqual(warn.call_count, 2)


if __name__ == "__main__":
    tests.main()

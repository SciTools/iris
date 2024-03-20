# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.cf.CFReader` class."""
from unittest import mock

import numpy as np
import pytest

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


class Test_translate__global_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        ncvar = netcdf_variable("ncvar", "height", np.float64)
        ncattrs = mock.Mock(return_value=["dimensions"])
        getncattr = mock.Mock(return_value="something something_else")
        dataset = mock.Mock(
            file_format="NetCDF4",
            variables={"ncvar": ncvar},
            ncattrs=ncattrs,
            getncattr=getncattr,
        )
        mocker.patch(
            "iris.fileformats.netcdf._thread_safe_nc.DatasetWrapper",
            return_value=dataset,
        )

    def test_create_global_attributes(self, mocker):
        global_attrs = CFReader("dummy").cf_group.global_attributes
        assert global_attrs["dimensions"] == "something something_else"


class Test_translate__formula_terms:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.delta = netcdf_variable("delta", "height", np.float64, bounds="delta_bnds")
        self.delta_bnds = netcdf_variable("delta_bnds", "height bnds", np.float64)
        self.sigma = netcdf_variable("sigma", "height", np.float64, bounds="sigma_bnds")
        self.sigma_bnds = netcdf_variable("sigma_bnds", "height bnds", np.float64)
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
        mocker.patch("iris.fileformats.cf.CFReader._build_cf_groups")
        mocker.patch("iris.fileformats.cf.CFReader._reset")
        mocker.patch(
            "iris.fileformats.netcdf._thread_safe_nc.DatasetWrapper",
            return_value=self.dataset,
        )

    def test_create_formula_terms(self, mocker):
        cf_group = CFReader("dummy").cf_group
        assert len(cf_group) == len(self.variables)
        # Check there is a singular data variable.
        group = cf_group.data_variables
        assert len(group) == 1
        assert list(group.keys()) == ["temp"]
        assert group["temp"].cf_data is self.temp
        # Check there are three coordinates.
        group = cf_group.coordinates
        assert len(group) == 3
        coordinates = ["height", "lat", "lon"]
        assert set(group.keys()) == set(coordinates)
        for name in coordinates:
            assert group[name].cf_data is getattr(self, name)
        # Check there are three auxiliary coordinates.
        group = cf_group.auxiliary_coordinates
        assert len(group) == 3
        aux_coordinates = ["delta", "sigma", "orography"]
        assert set(group.keys()) == set(aux_coordinates)
        for name in aux_coordinates:
            assert group[name].cf_data is getattr(self, name)
        # Check all the auxiliary coordinates are formula terms.
        formula_terms = cf_group.formula_terms
        assert set(group.items()) == set(formula_terms.items())
        # Check there are three bounds.
        group = cf_group.bounds
        assert len(group) == 3
        bounds = ["height_bnds", "delta_bnds", "sigma_bnds"]
        assert set(group.keys()) == set(bounds)
        for name in bounds:
            assert group[name].cf_data == getattr(self, name)


class Test_build_cf_groups__formula_terms:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.delta = netcdf_variable("delta", "height", np.float64, bounds="delta_bnds")
        self.delta_bnds = netcdf_variable("delta_bnds", "height bnds", np.float64)
        self.sigma = netcdf_variable("sigma", "height", np.float64, bounds="sigma_bnds")
        self.sigma_bnds = netcdf_variable("sigma_bnds", "height bnds", np.float64)
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
        mocker.patch("iris.fileformats.cf.CFReader._reset")
        mocker.patch(
            "iris.fileformats.netcdf._thread_safe_nc.DatasetWrapper",
            return_value=self.dataset,
        )

    def test_associate_formula_terms_with_data_variable(self, mocker):
        cf_group = CFReader("dummy").cf_group
        assert len(cf_group) == len(self.variables)
        # Check the cf-group associated with the data variable.
        temp_cf_group = cf_group["temp"].cf_group
        # Check the data variable is associated with eight variables.
        assert len(temp_cf_group) == 8
        # Check there are three coordinates.
        group = temp_cf_group.coordinates
        assert len(group) == 3
        coordinates = ["height", "lat", "lon"]
        assert set(group.keys()) == set(coordinates)
        for name in coordinates:
            assert group[name].cf_data is getattr(self, name)
        # Check the height coordinate is bounded.
        group = group["height"].cf_group
        assert len(group.bounds) == 1
        assert "height_bnds" in group.bounds
        assert group["height_bnds"].cf_data is self.height_bnds
        # Check there are five auxiliary coordinates.
        group = temp_cf_group.auxiliary_coordinates
        assert len(group) == 5
        aux_coordinates = ["delta", "sigma", "orography", "x", "y"]
        assert set(group.keys()) == set(aux_coordinates)
        for name in aux_coordinates:
            assert group[name].cf_data is getattr(self, name)
        # Check all the auxiliary coordinates are formula terms.
        formula_terms = cf_group.formula_terms
        assert set(formula_terms.items()).issubset(list(group.items()))
        # Check the terms by root.
        for name, term in zip(aux_coordinates, ["a", "b", "orog"]):
            assert formula_terms[name].cf_terms_by_root == dict(height=term)
        # Check the bounded auxiliary coordinates.
        for name, name_bnds in zip(["delta", "sigma"], ["delta_bnds", "sigma_bnds"]):
            aux_coord_group = group[name].cf_group
            assert len(aux_coord_group.bounds) == 1
            assert name_bnds in aux_coord_group.bounds
            assert aux_coord_group[name_bnds].cf_data is getattr(self, name_bnds)

    def test_promote_reference(self):
        cf_group = CFReader("dummy").cf_group
        assert len(cf_group) == len(self.variables)
        # Check the number of data variables.
        assert len(cf_group.data_variables) == 1
        assert list(cf_group.data_variables.keys()) == ["temp"]
        # Check the number of promoted variables.
        assert len(cf_group.promoted) == 1
        assert list(cf_group.promoted.keys()) == ["orography"]
        # Check the promoted variable dependencies.
        group = cf_group.promoted["orography"].cf_group.coordinates
        assert len(group) == 2
        coordinates = ("lat", "lon")
        assert set(group.keys()) == set(coordinates)
        for name in coordinates:
            assert group[name].cf_data is getattr(self, name)

    def test_formula_terms_ignore(self):
        self.orography.dimensions = ["lat", "wibble"]
        with pytest.warns(match="Ignoring formula terms variable"):
            cf_group = CFReader("dummy").cf_group
            group = cf_group.promoted
            assert list(group.keys()) == ["orography"]
            assert group["orography"].cf_data is self.orography

    def test_auxiliary_ignore(self):
        self.x.dimensions = ["lat", "wibble"]
        with pytest.warns(match="Ignoring variable 'x'"):
            cf_group = CFReader("dummy").cf_group
            promoted = ["x", "orography"]
            group = cf_group.promoted
            assert set(group.keys()) == set(promoted)
            for name in promoted:
                assert group[name].cf_data is getattr(self, name)

    def test_promoted_auxiliary_ignore(self):
        self.wibble = netcdf_variable("wibble", "lat wibble", np.float64)
        self.variables["wibble"] = self.wibble
        self.orography.coordinates = "wibble"
        with pytest.warns(match="Ignoring variable 'wibble'") as warns:
            cf_group = CFReader("dummy").cf_group.promoted
            promoted = ["wibble", "orography"]
            assert set(cf_group.keys()) == set(promoted)
            for name in promoted:
                assert cf_group[name].cf_data is getattr(self, name)
        # we should have got 2 warnings
        assert len(warns.list) == 2

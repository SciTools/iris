# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.cf.CFReader` class."""

import contextlib
import io

import numpy as np
import pytest

import iris
import iris.exceptions
from iris.fileformats import cf
from iris.fileformats.cf import (
    CFAuxiliaryCoordinateVariable,
    CFBoundaryVariable,
    CFCoordinateVariable,
    CFDataVariable,
    CFGridMappingVariable,
    CFGroup,
    CFReader,
    CFUGridAuxiliaryCoordinateVariable,
    CFUGridConnectivityVariable,
    CFUGridMeshVariable,
)
import iris.warnings


def netcdf_variable(
    mocker,
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

    ugrid_identities = (
        CFUGridAuxiliaryCoordinateVariable.cf_identities
        + CFUGridConnectivityVariable.cf_identities
        + [CFUGridMeshVariable.cf_identity]
    )
    ncvar = mocker.Mock(
        name=name,
        dimensions=dimensions,
        ncattrs=mocker.Mock(return_value=[]),
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
        **{name: None for name in ugrid_identities},
    )
    return ncvar


class Test_translate__global_attributes:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        ncvar = netcdf_variable(mocker, "ncvar", "height", np.float64)
        ncattrs = mocker.Mock(return_value=["dimensions"])
        getncattr = mocker.Mock(return_value="something something_else")
        dataset = mocker.Mock(
            file_format="NetCDF4",
            variables={"ncvar": ncvar},
            ncattrs=ncattrs,
            getncattr=getncattr,
        )
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=dataset,
        )

    def test_create_global_attributes(self, mocker):
        global_attrs = CFReader("dummy").cf_group.global_attributes
        assert global_attrs["dimensions"] == "something something_else"


class Test_translate__formula_terms:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.delta = netcdf_variable(
            mocker, "delta", "height", np.float64, bounds="delta_bnds"
        )
        self.delta_bnds = netcdf_variable(
            mocker, "delta_bnds", "height bnds", np.float64
        )
        self.sigma = netcdf_variable(
            mocker, "sigma", "height", np.float64, bounds="sigma_bnds"
        )
        self.sigma_bnds = netcdf_variable(
            mocker, "sigma_bnds", "height bnds", np.float64
        )
        self.orography = netcdf_variable(mocker, "orography", "lat lon", np.float64)
        formula_terms = "a: delta b: sigma orog: orography"
        standard_name = "atmosphere_hybrid_height_coordinate"
        self.height = netcdf_variable(
            mocker,
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
            mocker,
            "height_bnds",
            "height bnds",
            np.float64,
            formula_terms=formula_terms,
        )
        self.lat = netcdf_variable(mocker, "lat", "lat", np.float64)
        self.lon = netcdf_variable(mocker, "lon", "lon", np.float64)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable(
            mocker, "temp", "height lat lon", np.float64, coordinates="lat lon"
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
        ncattrs = mocker.Mock(return_value=[])
        self.dataset = mocker.Mock(
            file_format="NetCDF4", variables=self.variables, ncattrs=ncattrs
        )
        # Restrict the CFReader functionality to only performing translations.
        mocker.patch("iris.fileformats.cf.CFReader._build_cf_groups")
        mocker.patch("iris.fileformats.cf.CFReader._reset")
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
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
        self.delta = netcdf_variable(
            mocker, "delta", "height", np.float64, bounds="delta_bnds"
        )
        self.delta_bnds = netcdf_variable(
            mocker, "delta_bnds", "height bnds", np.float64
        )
        self.sigma = netcdf_variable(
            mocker, "sigma", "height", np.float64, bounds="sigma_bnds"
        )
        self.sigma_bnds = netcdf_variable(
            mocker, "sigma_bnds", "height bnds", np.float64
        )
        self.orography = netcdf_variable(mocker, "orography", "lat lon", np.float64)
        formula_terms = "a: delta b: sigma orog: orography"
        standard_name = "atmosphere_hybrid_height_coordinate"
        self.height = netcdf_variable(
            mocker,
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
            mocker,
            "height_bnds",
            "height bnds",
            np.float64,
            formula_terms=formula_terms,
        )
        self.lat = netcdf_variable(mocker, "lat", "lat", np.float64)
        self.lon = netcdf_variable(mocker, "lon", "lon", np.float64)
        self.x = netcdf_variable(mocker, "x", "lat lon", np.float64)
        self.y = netcdf_variable(mocker, "y", "lat lon", np.float64)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable(
            mocker, "temp", "height lat lon", np.float64, coordinates="x y"
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
        ncattrs = mocker.Mock(return_value=[])
        self.dataset = mocker.Mock(
            file_format="NetCDF4", variables=self.variables, ncattrs=ncattrs
        )
        # Restrict the CFReader functionality to only performing translations
        # and building first level cf-groups for variables.
        mocker.patch("iris.fileformats.cf.CFReader._reset")
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=self.dataset,
        )

        self.wibble = netcdf_variable(mocker, "wibble", "lat wibble", np.float64)

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
            assert group[name].cf_data == getattr(self, name)

    def test_formula_terms_ignore(self):
        self.orography.dimensions = ["lat", "wibble"]
        with pytest.warns(match="Ignoring formula terms variable"):
            cf_group = CFReader("dummy").cf_group
        group = cf_group.promoted
        assert list(group.keys()) == ["orography"]
        assert group["orography"].cf_data == self.orography

    def test_auxiliary_ignore(self):
        self.x.dimensions = ["lat", "wibble"]
        with pytest.warns(match=r"Ignoring variable x"):
            cf_group = CFReader("dummy").cf_group
        promoted = ["x", "orography"]
        group = cf_group.promoted
        assert set(group.keys()) == set(promoted)
        for name in promoted:
            assert group[name].cf_data == getattr(self, name)

    def test_promoted_auxiliary_ignore(self):
        self.variables["wibble"] = self.wibble
        self.orography.coordinates = "wibble"

        with pytest.warns(match="Ignoring variable wibble") as warns:
            cf_group = CFReader("dummy").cf_group.promoted

        promoted = ["wibble", "orography"]
        assert set(cf_group.keys()) == set(promoted)
        for name in promoted:
            assert cf_group[name].cf_data == getattr(self, name)
        # we should have got 2 warnings
        assert len(warns.list) == 2


class Test_build_cf_groups__ugrid:
    @pytest.fixture(autouse=True)
    def _setup_class(self, mocker):
        # Replicating syntax from test_CFReader.Test_build_cf_groups__formula_terms.
        self.mesh = netcdf_variable(mocker, "mesh", "", int)
        self.node_x = netcdf_variable(mocker, "node_x", "node", float)
        self.node_y = netcdf_variable(mocker, "node_y", "node", float)
        self.face_x = netcdf_variable(mocker, "face_x", "face", float)
        self.face_y = netcdf_variable(mocker, "face_y", "face", float)
        self.face_nodes = netcdf_variable(mocker, "face_nodes", "face vertex", int)
        self.levels = netcdf_variable(mocker, "levels", "levels", int)
        self.data = netcdf_variable(
            mocker, "data", "levels face", float, coordinates="face_x face_y"
        )

        # Add necessary attributes for mesh recognition.
        self.mesh.cf_role = "mesh_topology"
        self.mesh.node_coordinates = "node_x node_y"
        self.mesh.face_coordinates = "face_x face_y"
        self.mesh.face_node_connectivity = "face_nodes"
        self.face_nodes.cf_role = "face_node_connectivity"
        self.data.mesh = "mesh"

        self.variables = dict(
            mesh=self.mesh,
            node_x=self.node_x,
            node_y=self.node_y,
            face_x=self.face_x,
            face_y=self.face_y,
            face_nodes=self.face_nodes,
            levels=self.levels,
            data=self.data,
        )
        ncattrs = mocker.Mock(return_value=[])
        self.dataset = mocker.Mock(
            file_format="NetCDF4", variables=self.variables, ncattrs=ncattrs
        )

        # Restrict the CFReader functionality to only performing
        # translations and building first level cf-groups for variables.
        mocker.patch("iris.fileformats.cf.CFReader._reset")
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=self.dataset,
        )
        cf_reader = CFReader("dummy")
        self.cf_group = cf_reader.cf_group

    def test_inherited(self):
        for expected_var, collection in (
            [CFCoordinateVariable("levels", self.levels), "coordinates"],
            [CFDataVariable("data", self.data), "data_variables"],
        ):
            expected = {expected_var.cf_name: expected_var}
            assert getattr(self.cf_group, collection) == expected

    def test_connectivities(self):
        expected_var = CFUGridConnectivityVariable("face_nodes", self.face_nodes)
        expected = {expected_var.cf_name: expected_var}
        assert self.cf_group.connectivities == expected

    def test_mesh(self):
        expected_var = CFUGridMeshVariable("mesh", self.mesh)
        expected = {expected_var.cf_name: expected_var}
        assert self.cf_group.meshes == expected

    def test_ugrid_coords(self):
        names = [f"{loc}_{ax}" for loc in ("node", "face") for ax in ("x", "y")]
        expected = {
            name: CFUGridAuxiliaryCoordinateVariable(name, getattr(self, name))
            for name in names
        }
        assert self.cf_group.ugrid_coords == expected

    def test_is_cf_ugrid_group(self):
        assert isinstance(self.cf_group, CFGroup)


class Test_build_cf_groups__nczarr_scalar_grid_mapping:
    @pytest.fixture(autouse=True)
    def _setup_class(self, mocker):
        self.lat = netcdf_variable(mocker, "lat", "lat", np.float64)
        self.lon = netcdf_variable(mocker, "lon", "lon", np.float64)
        self.crs = netcdf_variable(mocker, "crs", "_scalar_", np.int32)
        self.crs.grid_mapping_name = "latitude_longitude"
        self.temp = netcdf_variable(
            mocker,
            "temp",
            "lat lon",
            np.float64,
            coordinates="lat lon",
            grid_mapping="crs",
        )
        self.lat.name = "lat"
        self.lon.name = "lon"
        self.crs.name = "crs"
        self.temp.name = "temp"

        self.variables = {
            "lat": self.lat,
            "lon": self.lon,
            "crs": self.crs,
            "temp": self.temp,
        }
        ncattrs = mocker.Mock(return_value=[])
        self.dataset = mocker.Mock(
            file_format="NetCDF4", variables=self.variables, ncattrs=ncattrs
        )
        mocker.patch("iris.fileformats.cf.CFReader._reset")
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=self.dataset,
        )
        self.cf_group = CFReader("dummy").cf_group

    def test_nczarr_scalar_grid_mapping_retains_type(self):
        expected_var = CFGridMappingVariable("crs", self.crs)
        assert self.cf_group.grid_mappings == {"crs": expected_var}
        assert "crs" not in self.cf_group.data_variables

    def test_nczarr_scalar_grid_mapping_spans_data_var(self):
        temp_cf_group = self.cf_group["temp"].cf_group
        assert "crs" in temp_cf_group.grid_mappings


def test_destructor(tmp_path):
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


class Test_init_and_lifecycle:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.variables = {"x": netcdf_variable(mocker, "x", "x", np.float64)}
        self.dataset = mocker.Mock(
            file_format="NetCDF4",
            variables=self.variables,
            ncattrs=mocker.Mock(return_value=[]),
            filepath=mocker.Mock(return_value="in-memory.nc"),
        )
        self.encoded_ds = mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=self.dataset,
        )
        mocker.patch("iris.fileformats.cf.CFReader._translate")
        mocker.patch("iris.fileformats.cf.CFReader._build_cf_groups")
        mocker.patch("iris.fileformats.cf.CFReader._reset")

    def test_init_with_url_source_preserves_filename_string(self):
        url = "https://example.com/some/file.nc"
        reader = CFReader(url)

        self.encoded_ds.assert_called_once_with(url, mode="r")
        assert reader.filename == url
        assert repr(reader) == f"CFReader('{url}')"

    def test_init_uses_dataset_wrapper_when_string_decode_disabled(self, mocker):
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.DECODE_TO_STRINGS_ON_READ",
            False,
        )
        wrapper_ds = mocker.patch(
            "iris.fileformats.cf._thread_safe_nc.DatasetWrapper",
            return_value=self.dataset,
        )

        reader = CFReader("dummy.nc")

        assert reader.filename.name == "dummy.nc"
        wrapper_ds.assert_called_once()
        self.encoded_ds.assert_not_called()

    def test_init_with_open_dataset_does_not_close_on_context_exit(self):
        with CFReader(self.dataset) as reader:
            assert reader is not None
            assert reader.filename == "in-memory.nc"

        self.dataset.close.assert_not_called()

    def test_init_warns_for_netcdf3_when_requested(self):
        self.dataset.file_format = "NETCDF3_CLASSIC"

        with pytest.warns(iris.warnings.IrisLoadWarning, match="Optimise CF-netCDF"):
            CFReader("dummy.nc", warn=True)

    def test_init_with_no_meshes_trims_ugrid_variable_types(self, mocker):
        self.dataset.variables = {"a": object(), "b": mocker.Mock(mesh=None)}

        reader = CFReader("dummy.nc")

        assert reader._with_ugrid is True

        mesh_free = mocker.Mock(
            file_format="NetCDF4",
            variables={},
            ncattrs=mocker.Mock(return_value=[]),
            filepath=mocker.Mock(return_value="in-memory.nc"),
        )
        self.encoded_ds.return_value = mesh_free
        reader = CFReader("dummy.nc")

        assert reader._with_ugrid is False
        assert CFUGridMeshVariable not in reader._variable_types


class Test_translate__grid_mapping_parse_errors:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.lat = netcdf_variable(mocker, "lat", "lat", np.float64)
        self.lon = netcdf_variable(mocker, "lon", "lon", np.float64)
        self.temp = netcdf_variable(
            mocker,
            "temp",
            "lat lon",
            np.float64,
            coordinates="lat lon",
            grid_mapping="bad syntax",
        )
        self.temp.name = "temp"
        variables = {"lat": self.lat, "lon": self.lon, "temp": self.temp}
        self.dataset = mocker.Mock(
            file_format="NetCDF4",
            variables=variables,
            ncattrs=mocker.Mock(return_value=[]),
        )
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=self.dataset,
        )
        mocker.patch(
            "iris.fileformats.cf.hh._parse_extended_grid_mapping",
            side_effect=iris.exceptions.CFParseError("failed to parse"),
        )

    def test_parse_failure_warns_and_reader_continues(self):
        with pytest.warns(
            iris.warnings.IrisCfWarning,
            match=r"Error parsing `grid_mapping` attribute for temp: failed to parse",
        ):
            cf_group = CFReader("dummy.nc").cf_group

        assert "temp" in cf_group.data_variables


class Test_translate__formula_terms_derived_bounds:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.term = netcdf_variable(mocker, "term", "z", np.float64)
        self.root = netcdf_variable(
            mocker,
            "z",
            "z",
            np.float64,
            formula_terms="a: term",
            bounds="z_bnds",
            standard_name="atmosphere_hybrid_height_coordinate",
        )
        self.root_bnds = netcdf_variable(
            mocker,
            "z_bnds",
            "z bnds",
            np.float64,
            formula_terms="a: term",
        )
        self.data = netcdf_variable(mocker, "temp", "z", np.float64)
        self.variables = {
            "z": self.root,
            "z_bnds": self.root_bnds,
            "term": self.term,
            "temp": self.data,
        }

    @staticmethod
    def _make_dataset(mocker, variables):
        return mocker.Mock(
            file_format="NetCDF4",
            variables=variables,
            ncattrs=mocker.Mock(return_value=[]),
        )

    def _patch_encoded_dataset(self, mocker):
        dataset = self._make_dataset(mocker, self.variables)
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=dataset,
        )

    @pytest.fixture(params=[True, False], ids=["FUTURE", "not_FUTURE"])
    def future_context(self, request):
        if request.param:
            result = iris.FUTURE.context(derived_bounds=True)
        else:
            result = contextlib.nullcontext()
        return result

    def test_derived_bounds_term(self, mocker, future_context):
        self._patch_encoded_dataset(mocker)

        with future_context:
            cf_group = CFReader("dummy.nc").cf_group

        assert "term" in cf_group.formula_terms
        assert isinstance(cf_group["term"], CFAuxiliaryCoordinateVariable)

    def test_derived_bounds_skips_when_term_missing(self, mocker, future_context):
        self.root_bnds.formula_terms = "a: missing_term"
        self._patch_encoded_dataset(mocker)

        with future_context:
            cf_group = CFReader("dummy.nc").cf_group

        assert "term" in cf_group.formula_terms
        assert isinstance(cf_group["term"], CFAuxiliaryCoordinateVariable)

    def test_promotes_non_formula_root_bounds_to_data(self, mocker):
        self.root_bnds = netcdf_variable(mocker, "z_bnds", "_scalar_", np.float64)
        # With valid formula terms, the variable would instead be recorded
        #  correctly as a bounds variable.
        del self.root_bnds.formula_terms
        self.variables["z_bnds"] = self.root_bnds
        self._patch_encoded_dataset(mocker)

        with iris.FUTURE.context(derived_bounds=True):
            cf_group = CFReader("dummy.nc").cf_group

        assert "z_bnds" in cf_group.promoted
        assert cf_group["z"].bounds is None

    def test_reclassifies_formula_term_bounds_variable(self, mocker, future_context):
        # If not referenced by any formula terms, the bounds variable is
        #  promoted to a data variable.
        # Referenced
        delta = netcdf_variable(
            mocker, "delta", "height", np.float64, bounds="delta_bnds"
        )
        # Referenced
        sigma = netcdf_variable(
            mocker, "sigma", "height", np.float64, bounds="sigma_bnds"
        )
        formula_terms = "a: delta b: sigma"
        height = netcdf_variable(
            mocker,
            "height",
            "height",
            np.float64,
            formula_terms=formula_terms,
            bounds="height_bnds",
            standard_name="atmosphere_hybrid_height_coordinate",
        )
        # Referenced
        formula_terms_bnds = "a: delta_bnds b: sigma_bnds"
        height_bnds = netcdf_variable(
            mocker,
            "height_bnds",
            "height bnds",
            np.float64,
            formula_terms=formula_terms_bnds,
        )
        delta_bnds = netcdf_variable(mocker, "delta_bnds", "height bnds", np.float64)
        sigma_bnds = netcdf_variable(mocker, "sigma_bnds", "height bnds", np.float64)
        data = netcdf_variable(mocker, "temp", "height", np.float64)
        variables = {
            "delta": delta,
            "sigma": sigma,
            "height": height,
            "height_bnds": height_bnds,
            "delta_bnds": delta_bnds,
            "sigma_bnds": sigma_bnds,
            "temp": data,
        }
        self.variables = variables
        self._patch_encoded_dataset(mocker)

        with future_context:
            cf_group = CFReader("dummy.nc").cf_group

        assert isinstance(cf_group["delta_bnds"], CFBoundaryVariable)
        assert isinstance(cf_group["sigma_bnds"], CFBoundaryVariable)
        assert cf_group["delta"].bounds == "delta_bnds"
        assert cf_group["sigma"].bounds == "sigma_bnds"

    def test_formula_term_already_in_group_uses_existing_variable(
        self, mocker, future_context
    ):
        self.root = netcdf_variable(
            mocker,
            "z",
            "z",
            np.float64,
            formula_terms="a: pressure",
            standard_name="atmosphere_hybrid_height_coordinate",
        )
        self.pressure = netcdf_variable(mocker, "pressure", "pressure", np.float64)
        self.variables = {"z": self.root, "pressure": self.pressure, "temp": self.data}
        self._patch_encoded_dataset(mocker)

        with future_context:
            cf_group = CFReader("dummy.nc").cf_group

        # When existing elsewhere, a variable referenced by a formula term is
        #  NOT created as an aux coord as it usually would. Instead the existing
        #  variable is re-used.
        assert cf_group.formula_terms["pressure"] is cf_group.coordinates["pressure"]
        assert "pressure" not in cf_group.auxiliary_coordinates

    @pytest.mark.parametrize(
        "standard_name", [False, True], ids=["no_standard_name", "with_standard_name"]
    )
    def test_derived_bounds_promotes_reference_terms(
        self, mocker, future_context, standard_name
    ):
        self.root = netcdf_variable(
            mocker,
            "z",
            "z",
            np.float64,
            formula_terms="a: pressure",
            standard_name="custom_reference",
        )
        if not standard_name:
            if isinstance(future_context, contextlib.nullcontext):
                pytest.skip("Test only applicable when FUTURE context is enabled.")
            else:
                del self.root.standard_name
        self.pressure = netcdf_variable(mocker, "pressure", "z", np.float64)
        self.variables = {"z": self.root, "pressure": self.pressure, "temp": self.data}
        self._patch_encoded_dataset(mocker)
        # reference_terms supports hybrid heights. Variables that are both
        #  named in formula_terms and referenced in reference_terms - "a" in
        #  this case - are always promoted.
        mocker.patch.dict(
            "iris.fileformats.cf.reference_terms",
            # Real world example: {"atmosphere_sigma_coordinate": ["ps"]}
            {"custom_reference": "a"},
            clear=False,
        )

        with future_context:
            cf_group = CFReader("dummy.nc").cf_group

        if standard_name:
            assert "pressure" in cf_group.promoted
        else:
            # Promotion step is skipped if standard_name is absent.
            assert "pressure" not in cf_group.promoted


class Test_translate__global_attributes_missing:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.var = netcdf_variable(mocker, "x", "x", np.float64)
        self.dataset = mocker.Mock(
            file_format="NetCDF4",
            variables={"x": self.var},
            ncattrs=mocker.Mock(return_value=["history"]),
            getncattr=mocker.Mock(side_effect=AttributeError),
        )
        mocker.patch(
            "iris.fileformats.netcdf._bytecoding_datasets.EncodedDataset",
            return_value=self.dataset,
        )
        mocker.patch("iris.fileformats.cf.CFReader._build_cf_groups")
        mocker.patch("iris.fileformats.cf.CFReader._reset")

    def test_translate_global_attr_missing_falls_back_to_default(self):
        cf_group = CFReader("dummy.nc").cf_group

        assert cf_group.global_attributes["history"] == ""


class Test_build_cf_groups__private_edge_cases:
    # Achieving full coverage requires a limited amount of internal state
    #  manipulation (to be avoided where possible as it hurts future refactoring).

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.reader = object.__new__(CFReader)
        self.reader._own_file = False
        self.reader._dataset = None
        self.reader._variable_types = ()
        self.reader._coord_system_mappings = {}
        self.reader.cf_group = CFGroup()

    def test_derived_bounds_relinks_spanning_bounds(self, mocker):
        data = netcdf_variable(
            mocker,
            "temp",
            "z",
            np.float64,
            bounds="temp_bnds",
            standard_name="atmosphere_hybrid_height_coordinate",
        )
        data.__len__ = mocker.Mock(return_value=1)
        data_var = CFDataVariable(
            "temp",
            data,
        )
        bnds_data = netcdf_variable(mocker, "temp_bnds", "z bnds", np.float64)
        bnds_data.__len__ = mocker.Mock(return_value=1)
        bnds_var = CFBoundaryVariable(
            "temp_bnds",
            bnds_data,
        )
        self.reader.cf_group["temp"] = data_var
        self.reader.cf_group["temp_bnds"] = bnds_var

        with iris.FUTURE.context(derived_bounds=True):
            self.reader._build_cf_groups({})

        assert "temp_bnds" in self.reader.cf_group["temp"].cf_group.bounds

    def test_derived_bounds_boundary_guard_continue_branch(self, mocker):
        # TODO: maybe the code itself is wrong and should be using isinstance?
        root = CFCoordinateVariable("z", netcdf_variable(mocker, "z", "z", np.float64))
        term = CFAuxiliaryCoordinateVariable(
            "term", netcdf_variable(mocker, "term", "z", np.float64)
        )
        term.add_formula_term("z", "a")
        self.reader.cf_group["z"] = root
        self.reader.cf_group["term"] = term

        # Force the exact identity check branch in CFReader._build_cf_groups.
        mocker.patch("iris.fileformats.cf.CFBoundaryVariable", term)
        with iris.FUTURE.context(derived_bounds=True):
            self.reader._build_cf_groups({})

        assert "term" in self.reader.cf_group.formula_terms

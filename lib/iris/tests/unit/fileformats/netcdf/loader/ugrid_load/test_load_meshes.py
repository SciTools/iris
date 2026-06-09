# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.mesh.load_meshes` function."""

from pathlib import Path
from uuid import uuid4

import pytest

from iris.common.mixin import CFVariableMixin
from iris.fileformats.netcdf.ugrid_load import load_meshes, logger
from iris.loading import LOAD_PROBLEMS
from iris.tests import _shared_utils
from iris.tests.stock.netcdf import ncgen_from_cdl


def cdl_to_nc(cdl, tmpdir: Path):
    cdl_path = str(tmpdir / "tst.cdl")
    nc_path = str(tmpdir / f"{uuid4()}.nc")
    # Use ncgen to convert this into an actual (temporary) netCDF file.
    ncgen_from_cdl(cdl_str=cdl, cdl_path=cdl_path, nc_path=nc_path)
    return nc_path


_TEST_CDL_HEAD = """
netcdf mesh_test {
    dimensions:
        node = 3 ;
        face = 1 ;
        vertex = 3 ;
        levels = 2 ;
    variables:
        int mesh ;
            mesh:cf_role = "mesh_topology" ;
            mesh:topology_dimension = 2 ;
            mesh:node_coordinates = "node_x node_y" ;
            mesh:face_node_connectivity = "face_nodes" ;
        float node_x(node) ;
            node_x:standard_name = "longitude" ;
        float node_y(node) ;
            node_y:standard_name = "latitude" ;
        int face_nodes(face, vertex) ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 0 ;
        int levels(levels) ;
        float node_data(levels, node) ;
            node_data:coordinates = "node_x node_y" ;
            node_data:location = "node" ;
            node_data:mesh = "mesh" ;
"""

_TEST_CDL_TAIL = """
data:
        mesh = 0;
        node_x = 0., 2., 1.;
        node_y = 0., 0., 1.;
        face_nodes = 0, 1, 2;
        levels = 1, 2;
        node_data = 0., 0., 0.;
    }
"""


class TestLoadErrors:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.ref_cdl = _TEST_CDL_HEAD + _TEST_CDL_TAIL
        self.nc_path = cdl_to_nc(self.ref_cdl, tmp_path)

    def add_second_mesh(self):
        second_name = "mesh2"
        cdl_extra = f"""
                     int {second_name} ;
                         {second_name}:cf_role = "mesh_topology" ;
                         {second_name}:topology_dimension = 2 ;
                         {second_name}:node_coordinates = "node_x node_y" ;
                         {second_name}:face_coordinates = "face_x face_y" ;
                         {second_name}:face_node_connectivity = "face_nodes" ;
             """
        vars_string = "variables:"
        vars_start = self.ref_cdl.index(vars_string) + len(vars_string)
        new_cdl = self.ref_cdl[:vars_start] + cdl_extra + self.ref_cdl[vars_start:]
        return new_cdl, second_name

    def test_with_data(self, tmp_path):
        nc_path = cdl_to_nc(self.ref_cdl, tmp_path)
        meshes = load_meshes(nc_path)

        files = list(meshes.keys())
        assert 1 == len(files)
        file_meshes = meshes[files[0]]
        assert 1 == len(file_meshes)
        mesh = file_meshes[0]
        assert "mesh" == mesh.var_name

    def test_no_data(self, tmp_path):
        cdl_lines = self.ref_cdl.split("\n")
        cdl_lines = filter(lambda line: ':mesh = "mesh"' not in line, cdl_lines)
        ref_cdl = "\n".join(cdl_lines)

        nc_path = cdl_to_nc(ref_cdl, tmp_path)
        meshes = load_meshes(nc_path)

        files = list(meshes.keys())
        assert 1 == len(files)
        file_meshes = meshes[files[0]]
        assert 1 == len(file_meshes)
        mesh = file_meshes[0]
        assert "mesh" == mesh.var_name

    def test_no_mesh(self, tmp_path):
        cdl_lines = self.ref_cdl.split("\n")
        cdl_lines = filter(
            lambda line: all(
                [s not in line for s in (':mesh = "mesh"', "mesh_topology")]
            ),
            cdl_lines,
        )
        ref_cdl = "\n".join(cdl_lines)

        nc_path = cdl_to_nc(ref_cdl, tmp_path)
        meshes = load_meshes(nc_path)

        assert {} == meshes

    def test_multi_files(self, tmp_path):
        files_count = 3
        nc_paths = [cdl_to_nc(self.ref_cdl, tmp_path) for _ in range(files_count)]
        meshes = load_meshes(nc_paths)
        assert files_count == len(meshes)

    def test_multi_meshes(self, tmp_path):
        ref_cdl, second_name = self.add_second_mesh()
        nc_path = cdl_to_nc(ref_cdl, tmp_path)
        meshes = load_meshes(nc_path)

        files = list(meshes.keys())
        assert 1 == len(files)
        file_meshes = meshes[files[0]]
        assert 2 == len(file_meshes)
        mesh_names = [mesh.var_name for mesh in file_meshes]
        assert "mesh" in mesh_names
        assert second_name in mesh_names

    def test_var_name(self, tmp_path):
        second_cdl, second_name = self.add_second_mesh()
        cdls = [self.ref_cdl, second_cdl]
        nc_paths = [cdl_to_nc(cdl, tmp_path) for cdl in cdls]
        meshes = load_meshes(nc_paths, second_name)

        files = list(meshes.keys())
        assert 1 == len(files)
        file_meshes = meshes[files[0]]
        assert 1 == len(file_meshes)
        assert second_name == file_meshes[0].var_name

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Iris cannot handle the URI scheme:.*"):
            _ = load_meshes("foo://bar")

    @_shared_utils.skip_data
    def test_non_nc(self, caplog):
        log_regex = r"Ignoring non-NetCDF file:.*"
        with _shared_utils.assert_logs(
            caplog, logger, level="INFO", msg_regex=log_regex
        ):
            meshes = load_meshes(
                _shared_utils.get_data_path(["PP", "simple_pp", "global.pp"])
            )
        assert {} == meshes

    def test_not_built(self, tmp_path):
        cdl = self.ref_cdl.replace("node_coordinates", "foo_coordinates")
        nc_path = cdl_to_nc(cdl, tmp_path)
        _ = load_meshes(nc_path)

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert "could not be identified from mesh node coordinates" in "".join(
            load_problem.stack_trace.format()
        )
        destination = load_problem.destination
        assert destination.iris_class is CFVariableMixin
        assert destination.identifier == "NOT_APPLICABLE"


class TestsHttp:
    # Tests of HTTP (OpenDAP) loading need mocking since we can't have tests
    #  that rely on 3rd party servers.
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        self.format_agent_mock = mocker.patch("iris.fileformats.FORMAT_AGENT.get_spec")

    def test_http(self):
        url = "https://foo"
        _ = load_meshes(url)
        self.format_agent_mock.assert_called_with(url, None)

    def test_mixed_sources(self, tmp_path):
        url = "https://foo"
        file = tmp_path / f"{uuid4()}.nc"
        file.touch()
        glob = f"{tmp_path}/*.nc"

        _ = load_meshes([url, glob])
        file_uris = [call[0][0] for call in self.format_agent_mock.call_args_list]
        for source in (url, Path(file).name):
            assert source in file_uris

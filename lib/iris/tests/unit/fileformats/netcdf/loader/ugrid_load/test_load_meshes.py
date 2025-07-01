# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.mesh.load_meshes` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
from shutil import rmtree
import tempfile
from uuid import uuid4

from iris.common.mixin import CFVariableMixin
from iris.fileformats.netcdf.ugrid_load import load_meshes, logger
from iris.loading import LOAD_PROBLEMS
from iris.tests.stock.netcdf import ncgen_from_cdl


def setUpModule():
    global TMP_DIR
    TMP_DIR = Path(tempfile.mkdtemp())


def tearDownModule():
    if TMP_DIR is not None:
        rmtree(TMP_DIR)


def cdl_to_nc(cdl, tmpdir=None):
    if tmpdir is None:
        tmpdir = TMP_DIR
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


class TestLoadErrors(tests.IrisTest):
    def setUp(self):
        self.ref_cdl = _TEST_CDL_HEAD + _TEST_CDL_TAIL
        self.nc_path = cdl_to_nc(self.ref_cdl)

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

    def test_with_data(self):
        nc_path = cdl_to_nc(self.ref_cdl)
        meshes = load_meshes(nc_path)

        files = list(meshes.keys())
        self.assertEqual(1, len(files))
        file_meshes = meshes[files[0]]
        self.assertEqual(1, len(file_meshes))
        mesh = file_meshes[0]
        self.assertEqual("mesh", mesh.var_name)

    def test_no_data(self):
        cdl_lines = self.ref_cdl.split("\n")
        cdl_lines = filter(lambda line: ':mesh = "mesh"' not in line, cdl_lines)
        ref_cdl = "\n".join(cdl_lines)

        nc_path = cdl_to_nc(ref_cdl)
        meshes = load_meshes(nc_path)

        files = list(meshes.keys())
        self.assertEqual(1, len(files))
        file_meshes = meshes[files[0]]
        self.assertEqual(1, len(file_meshes))
        mesh = file_meshes[0]
        self.assertEqual("mesh", mesh.var_name)

    def test_no_mesh(self):
        cdl_lines = self.ref_cdl.split("\n")
        cdl_lines = filter(
            lambda line: all(
                [s not in line for s in (':mesh = "mesh"', "mesh_topology")]
            ),
            cdl_lines,
        )
        ref_cdl = "\n".join(cdl_lines)

        nc_path = cdl_to_nc(ref_cdl)
        meshes = load_meshes(nc_path)

        self.assertDictEqual({}, meshes)

    def test_multi_files(self):
        files_count = 3
        nc_paths = [cdl_to_nc(self.ref_cdl) for _ in range(files_count)]
        meshes = load_meshes(nc_paths)
        self.assertEqual(files_count, len(meshes))

    def test_multi_meshes(self):
        ref_cdl, second_name = self.add_second_mesh()
        nc_path = cdl_to_nc(ref_cdl)
        meshes = load_meshes(nc_path)

        files = list(meshes.keys())
        self.assertEqual(1, len(files))
        file_meshes = meshes[files[0]]
        self.assertEqual(2, len(file_meshes))
        mesh_names = [mesh.var_name for mesh in file_meshes]
        self.assertIn("mesh", mesh_names)
        self.assertIn(second_name, mesh_names)

    def test_var_name(self):
        second_cdl, second_name = self.add_second_mesh()
        cdls = [self.ref_cdl, second_cdl]
        nc_paths = [cdl_to_nc(cdl) for cdl in cdls]
        meshes = load_meshes(nc_paths, second_name)

        files = list(meshes.keys())
        self.assertEqual(1, len(files))
        file_meshes = meshes[files[0]]
        self.assertEqual(1, len(file_meshes))
        self.assertEqual(second_name, file_meshes[0].var_name)

    def test_invalid_scheme(self):
        with self.assertRaisesRegex(ValueError, "Iris cannot handle the URI scheme:.*"):
            _ = load_meshes("foo://bar")

    @tests.skip_data
    def test_non_nc(self):
        log_regex = r"Ignoring non-NetCDF file:.*"
        with self.assertLogs(logger, level="INFO", msg_regex=log_regex):
            meshes = load_meshes(tests.get_data_path(["PP", "simple_pp", "global.pp"]))
        self.assertDictEqual({}, meshes)

    def test_not_built(self):
        cdl = self.ref_cdl.replace("node_coordinates", "foo_coordinates")
        nc_path = cdl_to_nc(cdl)
        _ = load_meshes(nc_path)

        load_problem = LOAD_PROBLEMS.problems[-1]
        self.assertIn(
            "could not be identified from mesh node coordinates",
            "".join(load_problem.stack_trace.format()),
        )
        destination = load_problem.destination
        self.assertIs(destination.iris_class, CFVariableMixin)
        self.assertEqual(destination.identifier, "NOT_APPLICABLE")


class TestsHttp(tests.IrisTest):
    # Tests of HTTP (OpenDAP) loading need mocking since we can't have tests
    #  that rely on 3rd party servers.
    def setUp(self):
        self.format_agent_mock = self.patch("iris.fileformats.FORMAT_AGENT.get_spec")

    def test_http(self):
        url = "https://foo"
        _ = load_meshes(url)
        self.format_agent_mock.assert_called_with(url, None)

    def test_mixed_sources(self):
        url = "https://foo"
        file = TMP_DIR / f"{uuid4()}.nc"
        file.touch()
        glob = f"{TMP_DIR}/*.nc"

        _ = load_meshes([url, glob])
        file_uris = [call[0][0] for call in self.format_agent_mock.call_args_list]
        for source in (url, Path(file).name):
            self.assertIn(source, file_uris)

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.fileformats.netcdf.load_cubes` function.

todo: migrate the remaining unit-esque tests from iris.tests.test_netcdf,
 switching to use netcdf.load_cubes() instead of iris.load()/load_cube().

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from pathlib import Path
from shutil import rmtree
import tempfile

from cf_units import as_unit
import numpy as np
import pytest

from iris.coords import AncillaryVariable, CellMeasure
from iris.cube import Cube
from iris.fileformats.netcdf import logger
from iris.fileformats.netcdf.loader import load_cubes
from iris.loading import LOAD_PROBLEMS
from iris.mesh import MeshCoord
from iris.tests.stock.netcdf import ncgen_from_cdl


def setUpModule():
    global TMP_DIR
    TMP_DIR = Path(tempfile.mkdtemp())


def tearDownModule():
    if TMP_DIR is not None:
        rmtree(TMP_DIR)


def cdl_to_nc(cdl):
    cdl_path = TMP_DIR / "tst.cdl"
    nc_path = TMP_DIR / "tst.nc"
    ncgen_from_cdl(cdl, cdl_path, nc_path)
    return str(nc_path)


class Tests(tests.IrisTest):
    def test_ancillary_variables(self):
        # Note: using a CDL string as a test data reference, rather than a
        # binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
            variables:
                int64 qqv(axv) ;
                    qqv:long_name = "qq" ;
                    qqv:units = "1" ;
                    qqv:ancillary_variables = "my_av" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_av(axv) ;
                    my_av:units = "1" ;
                    my_av:long_name = "refs" ;
                    my_av:custom = "extra-attribute";
            data:
                axv = 1, 2, 3;
                my_av = 11., 12., 13.;
            }
            """
        nc_path = cdl_to_nc(ref_cdl)

        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        avs = cubes[0].ancillary_variables()
        self.assertEqual(len(avs), 1)
        expected = AncillaryVariable(
            np.ma.array([11.0, 12.0, 13.0]),
            long_name="refs",
            var_name="my_av",
            units="1",
            attributes={"custom": "extra-attribute"},
        )
        self.assertEqual(avs[0], expected)

    def test_status_flags(self):
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
            variables:
                int64 qqv(axv) ;
                    qqv:long_name = "qq" ;
                    qqv:units = "1" ;
                    qqv:ancillary_variables = "my_av" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                byte my_av(axv) ;
                    my_av:long_name = "qq status_flag" ;
                    my_av:flag_values = 1b, 2b ;
                    my_av:flag_meanings = "a b" ;
            data:
                axv = 11, 21, 31;
                my_av = 1b, 1b, 2b;
            }
            """
        nc_path = cdl_to_nc(ref_cdl)

        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        avs = cubes[0].ancillary_variables()
        self.assertEqual(len(avs), 1)
        expected = AncillaryVariable(
            np.ma.array([1, 1, 2], dtype=np.int8),
            long_name="qq status_flag",
            var_name="my_av",
            units="no_unit",
            attributes={
                "flag_values": np.array([1, 2], dtype=np.int8),
                "flag_meanings": "a b",
            },
        )
        self.assertEqual(avs[0], expected)

    def test_cell_measures(self):
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
                ayv = 2 ;
            variables:
                int64 qqv(ayv, axv) ;
                    qqv:long_name = "qq" ;
                    qqv:units = "1" ;
                    qqv:cell_measures = "area: my_areas" ;
                int64 ayv(ayv) ;
                    ayv:units = "1" ;
                    ayv:long_name = "y" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_areas(ayv, axv) ;
                    my_areas:units = "m2" ;
                    my_areas:long_name = "standardised cell areas" ;
                    my_areas:custom = "extra-attribute";
            data:
                axv = 11, 12, 13;
                ayv = 21, 22;
                my_areas = 110., 120., 130., 221., 231., 241.;
            }
            """
        nc_path = cdl_to_nc(ref_cdl)

        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        cms = cubes[0].cell_measures()
        self.assertEqual(len(cms), 1)
        expected = CellMeasure(
            np.ma.array([[110.0, 120.0, 130.0], [221.0, 231.0, 241.0]]),
            measure="area",
            var_name="my_areas",
            long_name="standardised cell areas",
            units="m2",
            attributes={"custom": "extra-attribute"},
        )
        self.assertEqual(cms[0], expected)

    def test_default_units(self):
        # Note: using a CDL string as a test data reference, rather than a binary file.
        ref_cdl = """
            netcdf cm_attr {
            dimensions:
                axv = 3 ;
                ayv = 2 ;
            variables:
                int64 qqv(ayv, axv) ;
                    qqv:long_name = "qq" ;
                    qqv:ancillary_variables = "my_av" ;
                    qqv:cell_measures = "area: my_areas" ;
                int64 ayv(ayv) ;
                    ayv:long_name = "y" ;
                int64 axv(axv) ;
                    axv:units = "1" ;
                    axv:long_name = "x" ;
                double my_av(axv) ;
                    my_av:long_name = "refs" ;
                double my_areas(ayv, axv) ;
                    my_areas:long_name = "areas" ;
            data:
                axv = 11, 12, 13;
                ayv = 21, 22;
                my_areas = 110., 120., 130., 221., 231., 241.;
            }
            """
        nc_path = cdl_to_nc(ref_cdl)

        # Load with iris.fileformats.netcdf.load_cubes, and check expected content.
        cubes = list(load_cubes(nc_path))
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0].units, as_unit("unknown"))
        self.assertEqual(cubes[0].coord("y").units, as_unit("unknown"))
        self.assertEqual(cubes[0].coord("x").units, as_unit(1))
        self.assertEqual(cubes[0].ancillary_variable("refs").units, as_unit("unknown"))
        self.assertEqual(cubes[0].cell_measure("areas").units, as_unit("unknown"))


class TestsMesh(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        cls.ref_cdl = """
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
                        mesh:face_coordinates = "face_x face_y" ;
                        mesh:face_node_connectivity = "face_nodes" ;
                    float node_x(node) ;
                        node_x:standard_name = "longitude" ;
                    float node_y(node) ;
                        node_y:standard_name = "latitude" ;
                    float face_x(face) ;
                        face_x:standard_name = "longitude" ;
                    float face_y(face) ;
                        face_y:standard_name = "latitude" ;
                    int face_nodes(face, vertex) ;
                        face_nodes:cf_role = "face_node_connectivity" ;
                        face_nodes:start_index = 0 ;
                    int levels(levels) ;
                    float node_data(levels, node) ;
                        node_data:coordinates = "node_x node_y" ;
                        node_data:location = "node" ;
                        node_data:mesh = "mesh" ;
                    float face_data(levels, face) ;
                        face_data:coordinates = "face_x face_y" ;
                        face_data:location = "face" ;
                        face_data:mesh = "mesh" ;
                data:
                    mesh = 0;
                    node_x = 0., 2., 1.;
                    node_y = 0., 0., 1.;
                    face_x = 0.5;
                    face_y = 0.5;
                    face_nodes = 0, 1, 2;
                    levels = 1, 2;
                    node_data = 0., 0., 0.;
                    face_data = 0.;
                }
            """
        cls.nc_path = cdl_to_nc(cls.ref_cdl)
        cls.mesh_cubes = list(load_cubes(cls.nc_path))

    def setUp(self):
        # Interim measure to allow pytest-style patching in the absence of
        #  full-scale pytest conversion.
        self.monkeypatch = pytest.MonkeyPatch()

    def test_standard_dims(self):
        for cube in self.mesh_cubes:
            self.assertIsNotNone(cube.coords("levels"))

    def test_mesh_coord(self):
        cube = [cube for cube in self.mesh_cubes if cube.var_name == "face_data"][0]
        face_x = cube.coord("longitude")
        face_y = cube.coord("latitude")

        for coord in (face_x, face_y):
            self.assertIsInstance(coord, MeshCoord)
            self.assertEqual("face", coord.location)
            self.assertArrayEqual(np.ma.array([0.5]), coord.points)

        self.assertEqual("x", face_x.axis)
        self.assertEqual("y", face_y.axis)
        self.assertEqual(face_x.mesh, face_y.mesh)
        self.assertArrayEqual(np.ma.array([[0.0, 2.0, 1.0]]), face_x.bounds)
        self.assertArrayEqual(np.ma.array([[0.0, 0.0, 1.0]]), face_y.bounds)

    def test_shared_mesh(self):
        cube_meshes = [cube.coord("latitude").mesh for cube in self.mesh_cubes]
        self.assertEqual(cube_meshes[0], cube_meshes[1])

    def test_missing_mesh(self):
        ref_cdl = self.ref_cdl.replace(
            'face_data:mesh = "mesh"', 'face_data:mesh = "mesh2"'
        )
        nc_path = cdl_to_nc(ref_cdl)

        # No error when mesh handling not activated.
        _ = list(load_cubes(nc_path))

        log_regex = r".*could not be found in file."
        with self.assertLogs(logger, level="DEBUG", msg_regex=log_regex):
            _ = list(load_cubes(nc_path))

    def test_mesh_coord_not_built(self):
        def mock_build_mesh_coords(mesh, cf_var):
            raise RuntimeError("Mesh coords not built")

        with self.monkeypatch.context() as m:
            m.setattr(
                "iris.fileformats.netcdf.ugrid_load._build_mesh_coords",
                mock_build_mesh_coords,
            )
            _ = list(load_cubes(self.nc_path))

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert "Mesh coords not built" in "".join(load_problem.stack_trace.format())

    def test_mesh_coord_not_added(self):
        def mock_add_aux_coord(self, coord, data_dims=None):
            raise RuntimeError("Mesh coord not added")

        with self.monkeypatch.context() as m:
            m.setattr("iris.cube.Cube.add_aux_coord", mock_add_aux_coord)
            _ = list(load_cubes(self.nc_path))

        load_problem = LOAD_PROBLEMS.problems[-1]
        assert "Mesh coord not added" in "".join(load_problem.stack_trace.format())

    def test_mesh_coord_capture_destination(self):
        def mock_build_mesh_coords(mesh, cf_var):
            raise RuntimeError("Mesh coords not built")

        with self.monkeypatch.context() as m:
            m.setattr(
                "iris.fileformats.netcdf.ugrid_load._build_mesh_coords",
                mock_build_mesh_coords,
            )
            _ = list(load_cubes(self.nc_path))

        load_problem = LOAD_PROBLEMS.problems[-1]
        destination = load_problem.destination
        assert destination.iris_class is Cube
        assert destination.identifier in ("node_data", "face_data")

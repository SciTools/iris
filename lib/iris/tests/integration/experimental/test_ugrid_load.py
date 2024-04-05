# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for NetCDF-UGRID file loading.

todo: fold these tests into netcdf tests when experimental.ugrid is folded into
 standard behaviour.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from collections.abc import Iterable

import pytest

from iris import Constraint, load
from iris.experimental.ugrid.load import PARSE_UGRID_ON_LOAD, load_mesh, load_meshes
from iris.experimental.ugrid.mesh import Mesh
from iris.tests.stock.netcdf import (
    _file_from_cdl_template as create_file_from_cdl_template,
)
from iris.tests.unit.tests.stock.test_netcdf import XIOSFileMixin
from iris.warnings import IrisCfWarning


def ugrid_load(uris, constraints=None, callback=None):
    # TODO: remove constraint once files no longer have orphan connectivities.
    orphan_connectivities = (
        "Mesh2d_half_levels_edge_face_links",
        "Mesh2d_half_levels_face_links",
        "Mesh2d_half_levels_face_edges",
        "Mesh2d_full_levels_edge_face_links",
        "Mesh2d_full_levels_face_links",
        "Mesh2d_full_levels_face_edges",
    )
    filter_orphan_connectivities = Constraint(
        cube_func=lambda cube: cube.var_name not in orphan_connectivities
    )
    if constraints is None:
        constraints = filter_orphan_connectivities
    else:
        if not isinstance(constraints, Iterable):
            constraints = [constraints]
        constraints.append(filter_orphan_connectivities)

    with PARSE_UGRID_ON_LOAD.context():
        return load(uris, constraints, callback)


@tests.skip_data
class TestBasic(tests.IrisTest):
    def common_test(self, load_filename, assert_filename):
        cube_list = ugrid_load(
            tests.get_data_path(["NetCDF", "unstructured_grid", load_filename]),
        )
        self.assertEqual(1, len(cube_list))
        cube = cube_list[0]
        self.assertCML(cube, ["experimental", "ugrid", assert_filename])

    def test_2D_1t_face_half_levels(self):
        self.common_test(
            "lfric_ngvat_2D_1t_face_half_levels_main_conv_rain.nc",
            "2D_1t_face_half_levels.cml",
        )

    def test_3D_1t_face_half_levels(self):
        self.common_test(
            "lfric_ngvat_3D_1t_half_level_face_grid_derived_theta_in_w3.nc",
            "3D_1t_face_half_levels.cml",
        )

    def test_3D_1t_face_full_levels(self):
        self.common_test(
            "lfric_ngvat_3D_1t_full_level_face_grid_main_area_fraction_unit1.nc",
            "3D_1t_face_full_levels.cml",
        )

    def test_2D_72t_face_half_levels(self):
        self.common_test(
            "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
            "2D_72t_face_half_levels.cml",
        )

    def test_3D_snow_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_snow_pseudo_levels_1t_face_half_levels_main_snow_layer_temp.nc",
            "3D_snow_pseudo_levels.cml",
        )

    def test_3D_soil_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_soil_pseudo_levels_1t_face_half_levels_main_soil_temperature.nc",
            "3D_soil_pseudo_levels.cml",
        )

    def test_3D_tile_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_tile_pseudo_levels_1t_face_half_levels_main_sw_up_tile.nc",
            "3D_tile_pseudo_levels.cml",
        )

    def test_3D_veg_pseudo_levels(self):
        self.common_test(
            "lfric_ngvat_3D_veg_pseudo_levels_1t_face_half_levels_main_snowpack_density.nc",
            "3D_veg_pseudo_levels.cml",
        )

    def test_no_mesh(self):
        with PARSE_UGRID_ON_LOAD.context():
            cube_list = load(
                tests.get_data_path(
                    ["NetCDF", "unstructured_grid", "theta_nodal_not_ugrid.nc"]
                )
            )
        self.assertTrue(all([cube.mesh is None for cube in cube_list]))


@tests.skip_data
class TestMultiplePhenomena(tests.IrisTest):
    def test_multiple_phenomena(self):
        cube_list = ugrid_load(
            tests.get_data_path(
                ["NetCDF", "unstructured_grid", "lfric_surface_mean.nc"]
            ),
        )
        self.assertCML(cube_list, ("experimental", "ugrid", "surface_mean.cml"))


class TestTolerantLoading(XIOSFileMixin):
    # N.B. using parts of the XIOS-like file integration testing, to make
    # temporary netcdf files from stored CDL templates.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # create cls.temp_dir = dir for test files

    @classmethod
    def tearDownClass(cls):
        super().setUpClass()  # destroy temp dir

    # Create a testfile according to testcase-specific arguments.
    # NOTE: with this, parent "create_synthetic_test_cube" can load a cube.
    def create_synthetic_file(self, **create_kwargs):
        template_name = create_kwargs["template"]  # required kwarg
        testfile_name = "tmp_netcdf"
        template_subs = dict(NUM_NODES=7, NUM_FACES=3, DATASET_NAME=testfile_name)
        kwarg_subs = create_kwargs.get("subs", {})  # optional kwarg
        template_subs.update(kwarg_subs)
        filepath = create_file_from_cdl_template(
            temp_file_dir=self.temp_dir,
            dataset_name=testfile_name,
            dataset_type=template_name,
            template_subs=template_subs,
        )
        return str(filepath)  # N.B. Path object not usable in iris.load

    def test_mesh_bad_topology_dimension(self):
        # Check that the load generates a suitable warning.
        warn_regex = r"topology_dimension.* ignoring"
        with pytest.warns(IrisCfWarning, match=warn_regex):
            template = "minimal_bad_topology_dim"
            dim_line = "mesh_var:topology_dimension = 1 ;"  # which is wrong !
            cube = self.create_synthetic_test_cube(
                template=template, subs=dict(TOPOLOGY_DIM_DEFINITION=dim_line)
            )

        # Check that the result has topology-dimension of 2 (not 1).
        self.assertEqual(cube.mesh.topology_dimension, 2)

    def test_mesh_no_topology_dimension(self):
        # Check that the load generates a suitable warning.
        warn_regex = r"Mesh variable.* has no 'topology_dimension'"
        with pytest.warns(IrisCfWarning, match=warn_regex):
            template = "minimal_bad_topology_dim"
            dim_line = ""  # don't create ANY topology_dimension property
            cube = self.create_synthetic_test_cube(
                template=template, subs=dict(TOPOLOGY_DIM_DEFINITION=dim_line)
            )

        # Check that the result has the correct topology-dimension value.
        self.assertEqual(cube.mesh.topology_dimension, 2)

    def test_mesh_bad_cf_role(self):
        # Check that the load generates a suitable warning.
        warn_regex = r"inappropriate cf_role"
        with pytest.warns(IrisCfWarning, match=warn_regex):
            template = "minimal_bad_mesh_cf_role"
            dim_line = 'mesh_var:cf_role = "foo" ;'
            _ = self.create_synthetic_test_cube(
                template=template, subs=dict(CF_ROLE_DEFINITION=dim_line)
            )

    def test_mesh_no_cf_role(self):
        # Check that the load generates a suitable warning.
        warn_regex = r"no cf_role attribute"
        with pytest.warns(IrisCfWarning, match=warn_regex):
            template = "minimal_bad_mesh_cf_role"
            dim_line = ""
            _ = self.create_synthetic_test_cube(
                template=template, subs=dict(CF_ROLE_DEFINITION=dim_line)
            )


@tests.skip_data
class Test_load_mesh(tests.IrisTest):
    def common_test(self, file_name, mesh_var_name):
        with PARSE_UGRID_ON_LOAD.context():
            mesh = load_mesh(
                tests.get_data_path(["NetCDF", "unstructured_grid", file_name])
            )
        # NOTE: cannot use CML tests as this isn't supported for non-Cubes.
        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(mesh.var_name, mesh_var_name)

    def test_full_file(self):
        self.common_test(
            "lfric_ngvat_2D_1t_face_half_levels_main_conv_rain.nc",
            "Mesh2d_half_levels",
        )

    def test_mesh_file(self):
        self.common_test("mesh_C12.nc", "dynamics")

    def test_no_mesh(self):
        with PARSE_UGRID_ON_LOAD.context():
            meshes = load_meshes(
                tests.get_data_path(
                    ["NetCDF", "unstructured_grid", "theta_nodal_not_ugrid.nc"]
                )
            )
        self.assertDictEqual({}, meshes)


if __name__ == "__main__":
    tests.main()

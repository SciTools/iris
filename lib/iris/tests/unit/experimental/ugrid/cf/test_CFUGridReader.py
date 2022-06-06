# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.cf.CFUGridGroup` class.

todo: fold these tests into cf tests when experimental.ugrid is folded into
 standard behaviour.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

from iris.experimental.ugrid.cf import (
    CFUGridAuxiliaryCoordinateVariable,
    CFUGridConnectivityVariable,
    CFUGridGroup,
    CFUGridMeshVariable,
    CFUGridReader,
)
from iris.fileformats.cf import CFCoordinateVariable, CFDataVariable
from iris.tests.unit.fileformats.cf.test_CFReader import netcdf_variable


def netcdf_ugrid_variable(
    name,
    dimensions,
    dtype,
    coordinates=None,
):
    ncvar = netcdf_variable(
        name=name, dimensions=dimensions, dtype=dtype, coordinates=coordinates
    )

    # Fill in all the extra UGRID attributes to prevent problems with getattr
    # and Mock. Any attribute can be replaced in downstream setUp if present.
    ugrid_attrs = (
        CFUGridAuxiliaryCoordinateVariable.cf_identities
        + CFUGridConnectivityVariable.cf_identities
        + [CFUGridMeshVariable.cf_identity]
    )
    for attr in ugrid_attrs:
        setattr(ncvar, attr, None)

    return ncvar


class Test_build_cf_groups(tests.IrisTest):
    @classmethod
    def setUpClass(cls):
        # Replicating syntax from test_CFReader.Test_build_cf_groups__formula_terms.
        cls.mesh = netcdf_ugrid_variable("mesh", "", int)
        cls.node_x = netcdf_ugrid_variable("node_x", "node", float)
        cls.node_y = netcdf_ugrid_variable("node_y", "node", float)
        cls.face_x = netcdf_ugrid_variable("face_x", "face", float)
        cls.face_y = netcdf_ugrid_variable("face_y", "face", float)
        cls.face_nodes = netcdf_ugrid_variable(
            "face_nodes", "face vertex", int
        )
        cls.levels = netcdf_ugrid_variable("levels", "levels", int)
        cls.data = netcdf_ugrid_variable(
            "data", "levels face", float, coordinates="face_x face_y"
        )

        # Add necessary attributes for mesh recognition.
        cls.mesh.cf_role = "mesh_topology"
        cls.mesh.node_coordinates = "node_x node_y"
        cls.mesh.face_coordinates = "face_x face_y"
        cls.mesh.face_node_connectivity = "face_nodes"
        cls.face_nodes.cf_role = "face_node_connectivity"
        cls.data.mesh = "mesh"

        cls.variables = dict(
            mesh=cls.mesh,
            node_x=cls.node_x,
            node_y=cls.node_y,
            face_x=cls.face_x,
            face_y=cls.face_y,
            face_nodes=cls.face_nodes,
            levels=cls.levels,
            data=cls.data,
        )
        ncattrs = mock.Mock(return_value=[])
        cls.dataset = mock.Mock(
            file_format="NetCDF4", variables=cls.variables, ncattrs=ncattrs
        )

    def setUp(self):
        # Restrict the CFUGridReader functionality to only performing
        # translations and building first level cf-groups for variables.
        self.patch("iris.experimental.ugrid.cf.CFUGridReader._reset")
        self.patch("netCDF4.Dataset", return_value=self.dataset)
        cf_reader = CFUGridReader("dummy")
        self.cf_group = cf_reader.cf_group

    def test_inherited(self):
        for expected_var, collection in (
            [CFCoordinateVariable("levels", self.levels), "coordinates"],
            [CFDataVariable("data", self.data), "data_variables"],
        ):
            expected = {expected_var.cf_name: expected_var}
            self.assertDictEqual(expected, getattr(self.cf_group, collection))

    def test_connectivities(self):
        expected_var = CFUGridConnectivityVariable(
            "face_nodes", self.face_nodes
        )
        expected = {expected_var.cf_name: expected_var}
        self.assertDictEqual(expected, self.cf_group.connectivities)

    def test_mesh(self):
        expected_var = CFUGridMeshVariable("mesh", self.mesh)
        expected = {expected_var.cf_name: expected_var}
        self.assertDictEqual(expected, self.cf_group.meshes)

    def test_ugrid_coords(self):
        names = [
            f"{loc}_{ax}" for loc in ("node", "face") for ax in ("x", "y")
        ]
        expected = {
            name: CFUGridAuxiliaryCoordinateVariable(name, getattr(self, name))
            for name in names
        }
        self.assertDictEqual(expected, self.cf_group.ugrid_coords)

    def test_is_cf_ugrid_group(self):
        self.assertIsInstance(self.cf_group, CFUGridGroup)

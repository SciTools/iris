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

from unittest.mock import MagicMock

from iris.experimental.ugrid.cf import (
    CFUGridAuxiliaryCoordinateVariable,
    CFUGridConnectivityVariable,
    CFUGridGroup,
    CFUGridMeshVariable,
)
from iris.fileformats.cf import CFCoordinateVariable, CFDataVariable


class Tests(tests.IrisTest):
    def setUp(self):
        self.cf_group = CFUGridGroup()

    def test_inherited(self):
        coord_var = MagicMock(spec=CFCoordinateVariable, cf_name="coord_var")
        self.cf_group[coord_var.cf_name] = coord_var
        self.assertEqual(
            coord_var, self.cf_group.coordinates[coord_var.cf_name]
        )

    def test_connectivities(self):
        conn_var = MagicMock(
            spec=CFUGridConnectivityVariable, cf_name="conn_var"
        )
        self.cf_group[conn_var.cf_name] = conn_var
        self.assertEqual(
            conn_var, self.cf_group.connectivities[conn_var.cf_name]
        )

    def test_ugrid_coords(self):
        coord_var = MagicMock(
            spec=CFUGridAuxiliaryCoordinateVariable, cf_name="coord_var"
        )
        self.cf_group[coord_var.cf_name] = coord_var
        self.assertEqual(
            coord_var, self.cf_group.ugrid_coords[coord_var.cf_name]
        )

    def test_meshes(self):
        mesh_var = MagicMock(spec=CFUGridMeshVariable, cf_name="mesh_var")
        self.cf_group[mesh_var.cf_name] = mesh_var
        self.assertEqual(mesh_var, self.cf_group.meshes[mesh_var.cf_name])

    def test_non_data_names(self):
        data_var = MagicMock(spec=CFDataVariable, cf_name="data_var")
        coord_var = MagicMock(spec=CFCoordinateVariable, cf_name="coord_var")
        conn_var = MagicMock(
            spec=CFUGridConnectivityVariable, cf_name="conn_var"
        )
        ugrid_coord_var = MagicMock(
            spec=CFUGridAuxiliaryCoordinateVariable, cf_name="ugrid_coord_var"
        )
        mesh_var = MagicMock(spec=CFUGridMeshVariable, cf_name="mesh_var")
        mesh_var2 = MagicMock(spec=CFUGridMeshVariable, cf_name="mesh_var2")
        duplicate_name_var = MagicMock(
            spec=CFUGridMeshVariable, cf_name="coord_var"
        )

        for var in (
            data_var,
            coord_var,
            conn_var,
            ugrid_coord_var,
            mesh_var,
            mesh_var2,
            duplicate_name_var,
        ):
            self.cf_group[var.cf_name] = var

        expected_names = [
            var.cf_name
            for var in (
                coord_var,
                conn_var,
                ugrid_coord_var,
                mesh_var,
                mesh_var2,
            )
        ]
        expected = set(expected_names)
        self.assertEqual(expected, self.cf_group.non_data_variable_names)

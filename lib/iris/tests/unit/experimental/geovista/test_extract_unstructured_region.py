# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.experimental.geovista.extract_unstructured_region` function."""

from unittest.mock import MagicMock, Mock

from geovista.common import VTK_CELL_IDS, VTK_POINT_IDS
import numpy as np
import pytest

from iris.experimental.geovista import extract_unstructured_region
from iris.tests.stock import sample_2d_latlons
from iris.tests.stock.mesh import sample_mesh_cube


class TestRegionExtraction:
    @pytest.fixture()
    def cube_2d(self):
        return sample_2d_latlons()

    @pytest.fixture(params=["face", "node"], autouse=True)
    def cube_mesh(self, request):
        self.cube_mesh = sample_mesh_cube(location=request.param, n_z=10)

    @pytest.fixture()
    def cube_mesh_edge(self):
        return sample_mesh_cube(location="edge")

    @pytest.fixture(autouse=True)
    def mocked_polydata(self):
        mock_polydata_scalars = {
            VTK_CELL_IDS: np.arange(2),
            VTK_POINT_IDS: np.arange(14),
        }
        polydata = MagicMock()
        polydata.__getitem__.side_effect = mock_polydata_scalars.__getitem__
        polydata.GetNumberOfCells.return_value = 3
        polydata.GetNumberOfPoints.return_value = 15
        self.mocked_polydata = polydata

    @pytest.fixture(autouse=True)
    def mocked_region(self):
        region = Mock()
        region.enclosed.return_value = self.mocked_polydata
        self.mocked_region = region

    def test_called_with(self):
        extract_unstructured_region(
            self.cube_mesh, self.mocked_polydata, self.mocked_region
        )
        self.mocked_region.enclosed.assert_called_with(self.mocked_polydata)

    def test_kwarg(self):
        extract_unstructured_region(
            self.cube_mesh,
            self.mocked_polydata,
            self.mocked_region,
            test="test",
        )
        actual = self.mocked_region.enclosed.call_args.kwargs
        assert actual["test"] == "test"

    @pytest.mark.parametrize(
        "transpose_cube", [True, False], ids=["Transposed", "Not Transposed"]
    )
    def test_indices(self, transpose_cube):
        if transpose_cube:
            self.cube_mesh.transpose()
        extracted_region = extract_unstructured_region(
            self.cube_mesh, self.mocked_polydata, self.mocked_region
        )
        if self.cube_mesh.location == "face":
            expected_length = len(self.mocked_polydata[VTK_CELL_IDS])
        else:
            assert self.cube_mesh.location == "node"
            expected_length = len(self.mocked_polydata[VTK_POINT_IDS])
        mesh_dim = self.cube_mesh.mesh_dim()
        assert extracted_region.shape[mesh_dim] == expected_length

    def test_empty_indices(self):
        mock_polydata_scalars = {
            VTK_CELL_IDS: np.arange(0),
            VTK_POINT_IDS: np.arange(0),
        }
        self.mocked_polydata.__getitem__.side_effect = mock_polydata_scalars.__getitem__
        with pytest.raises(
            IndexError, match="No part of `polydata` falls within `region`."
        ):
            extract_unstructured_region(
                self.cube_mesh, self.mocked_polydata, self.mocked_region
            )

    def test_recreate_mesh(self):
        extracted_region = extract_unstructured_region(
            self.cube_mesh, self.mocked_polydata, self.mocked_region
        )
        if self.cube_mesh.location == "face":
            assert extracted_region.mesh is not None
        else:
            assert extracted_region.mesh is None

    def test_new_mesh_coords(self):
        extracted_region = extract_unstructured_region(
            self.cube_mesh, self.mocked_polydata, self.mocked_region
        )
        if self.cube_mesh.location == "face":
            mesh_coords = extracted_region.coords(mesh_coords=True)
            np.testing.assert_array_equal(
                mesh_coords[0].bounds,
                [[1200, 1201, 1202, 1203], [1204, 1205, 1206, 1207]],
            )
            np.testing.assert_array_equal(mesh_coords[0].points, [3200, 3201])
            np.testing.assert_array_equal(
                mesh_coords[1].bounds,
                [[1100, 1101, 1102, 1103], [1104, 1105, 1106, 1107]],
            )
            np.testing.assert_array_equal(mesh_coords[1].points, [3100, 3101])

    def test_no_mesh(self, cube_2d):
        cube = cube_2d
        with pytest.raises(ValueError, match="Cube must have a mesh"):
            extract_unstructured_region(cube, self.mocked_polydata, self.mocked_region)

    def test_edge_location(self, cube_mesh_edge):
        with pytest.raises(
            NotImplementedError,
            match=f"cube.location must be `face` or `node`."
            f" Found: {cube_mesh_edge.location}.",
        ):
            extract_unstructured_region(
                cube_mesh_edge, self.mocked_polydata, self.mocked_region
            )

    def test_cube_and_poly_shapes_mismatch(self):
        self.mocked_polydata.GetNumberOfCells.return_value = 4
        self.mocked_polydata.GetNumberOfPoints.return_value = 16
        polydata_length = ()
        if self.cube_mesh.location == "face":
            polydata_length = self.mocked_polydata.GetNumberOfCells()
        elif self.cube_mesh.location == "node":
            polydata_length = self.mocked_polydata.GetNumberOfPoints()
        with pytest.raises(
            ValueError,
            match=f"The mesh on the cube and the polydata"
            f"must have the same shape."
            f" Found Mesh: {self.cube_mesh.shape[self.cube_mesh.mesh_dim()]},"
            f" Polydata: {polydata_length}.",
        ):
            extract_unstructured_region(
                self.cube_mesh, self.mocked_polydata, self.mocked_region
            )

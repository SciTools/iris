import numpy as np
import pytest
import shapely

from iris.coords import DimCoord
import iris.cube
from iris.exceptions import IrisUserWarning
import iris.tests as tests
from iris.util import mask_cube_from_shapefile


class TestBasicCubeMasking(tests.IrisTest):
    """Unit tests for mask_cube_from_shapefile function"""

    basic_data = np.array([[1, 2], [4, 8]])
    basic_cube = iris.cube.Cube(basic_data)
    coord = DimCoord(
        np.array([0, 1]),
        standard_name="projection_y_coordinate",
        bounds=[[0, 0.5], [0.5, 1]],
        units="1",
    )
    basic_cube.add_dim_coord(coord, 0)
    coord = DimCoord(
        np.array([0, 1]),
        standard_name="projection_x_coordinate",
        bounds=[[0, 0.5], [0.5, 1]],
        units="1",
    )
    basic_cube.add_dim_coord(coord, 1)

    def testBasicCubeIntersect(self):
        shape = shapely.geometry.box(0.6, 0.6, 1, 1)
        masked_cube = mask_cube_from_shapefile(shape, self.basic_cube)
        assert (
            np.sum(masked_cube.data) == 8
        ), f"basic cube masking failed test - expected 8 got {np.sum(masked_cube.data)}"

    def testBasicCubeIntersectLowWeight(self):
        shape = shapely.geometry.box(0.1, 0.6, 1, 1)
        masked_cube = mask_cube_from_shapefile(
            shape, self.basic_cube, minimum_weight=0.2
        )
        assert (
            np.sum(masked_cube.data) == 12
        ), f"basic cube masking weighting failed test - expected 12 got {np.sum(masked_cube.data)}"

    def testBasicCubeIntersectHighWeight(self):
        shape = shapely.geometry.box(0.1, 0.6, 1, 1)
        masked_cube = mask_cube_from_shapefile(
            shape, self.basic_cube, minimum_weight=0.7
        )
        assert (
            np.sum(masked_cube.data) == 8
        ), f"basic cube masking weighting failed test- expected 8 got {np.sum(masked_cube.data)}"

    def testCubeListError(self):
        cubelist = iris.cube.CubeList([self.basic_cube])
        shape = shapely.geometry.box(1, 1, 2, 2)
        with pytest.raises(
            TypeError, match="CubeList object rather than Cube"
        ):
            mask_cube_from_shapefile(shape, cubelist)

    def testNonCubeError(self):
        fake = None
        shape = shapely.geometry.box(1, 1, 2, 2)
        with pytest.raises(TypeError, match="Received non-Cube object"):
            mask_cube_from_shapefile(shape, fake)

    def testLineShapeWarning(self):
        shape = shapely.geometry.LineString([(0, 0.75), (2, 0.75)])
        with pytest.warns(IrisUserWarning, match="invalid type"):
            masked_cube = mask_cube_from_shapefile(
                shape, self.basic_cube, minimum_weight=0.1
            )
        assert (
            np.sum(masked_cube.data) == 12
        ), f"basic cube masking against line failed test - expected 7 got {np.sum(masked_cube.data)}"

    def testShapeInvalid(self):
        shape = None
        with pytest.raises(TypeError, match="valid Shapely"):
            mask_cube_from_shapefile(shape, self.basic_cube)

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for merging with dataless cubes."""

import dask.array as da
import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList


class TestMergeDataless:
    def _testcube(self, z=1, name="this", dataless=False, lazy=False):
        # Create a testcube with a scalar Z coord, for merge testing.
        data = da.arange(3) if lazy else np.arange(3)
        cube = Cube(
            data,
            long_name=name,
            dim_coords_and_dims=[(DimCoord([0.0, 1.0, 2], long_name="x"), 0)],
            aux_coords_and_dims=[(AuxCoord([z], long_name="z"), ())],
        )
        if dataless:
            cube.data = None
        return cube

    def test_mixed_passthrough(self):
        # Check that normal merge can handle dataless alongside dataful cubes.
        cube_normal = self._testcube(name="this", dataless=False)
        cube_dataless = self._testcube(name="that", dataless=True)
        cubes = CubeList([cube_normal, cube_dataless])

        result = cubes.merge()

        assert len(result) == 2
        cube1, cube2 = [result.extract_cube(name) for name in ("this", "that")]
        assert not cube1.is_dataless()
        assert cube2.is_dataless()

    def test_dataless_merge(self):
        # Check that dataless cubes can be merged.
        cube_1 = self._testcube(z=1, dataless=True)
        cube_2 = self._testcube(z=2, dataless=True)
        cubes = CubeList([cube_1, cube_2])

        cube = cubes.merge_cube()

        assert cube.is_dataless()
        assert np.all(cube.coord("z").points == [1, 2])

    def test_dataless_dataful_merge(self):
        # Check that dataless cubes can merge **with** regular ones.
        # Include checking that laziness is preserved.
        cube_normal = self._testcube(z=1, dataless=False, lazy=True)
        cube_dataless = self._testcube(z=2, dataless=True)
        cubes = CubeList([cube_normal, cube_dataless])

        cube = cubes.merge_cube()

        assert not cube.is_dataless()
        assert cube.has_lazy_data()
        data_z1, data_z2 = cube[0].data, cube[1].data
        assert np.all(data_z1 == [0, 1, 2])
        assert np.all(np.ma.getmaskarray(data_z2) == True)  # noqa: E712

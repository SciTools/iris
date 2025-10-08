# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for merging with dataless cubes."""

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList


class TestMergeDataless:
    def _testcube(self, z=1, name="this", dataless=False):
        # Create a testcube with a scalar Z coord, for merge testing.
        cube = Cube(
            [1, 2, 3],
            long_name=name,
            dim_coords_and_dims=[(DimCoord([0.0, 1.0, 2], long_name="x"), 0)],
            aux_coords_and_dims=[(AuxCoord([z], long_name="z"), ())],
        )
        if dataless:
            cube.data = None
        return cube

    def test_mixed_passthrough(self):
        # Check that normal merge can handle dataless alongside dataful cubes.
        cubes = CubeList(
            [
                self._testcube(name="this", dataless=False),
                self._testcube(name="that", dataless=True),
            ]
        )
        result = cubes.merge()
        assert len(result) == 2
        cube1, cube2 = [result.extract_cube(name) for name in ("this", "that")]
        assert not cube1.is_dataless()
        assert cube2.is_dataless()

    def test_dataless_merge(self):
        # Check that dataless cubes can be merged.
        cubes = CubeList(
            [
                self._testcube(z=1, dataless=True),
                self._testcube(z=2, dataless=True),
            ]
        )
        cube = cubes.merge_cube()
        assert cube.is_dataless()
        assert np.all(cube.coord("z").points == [1, 2])

    def test_dataless_dataful_merge(self):
        # Check that dataless cubes can merge **with** regular ones.
        # Check that dataless cubes can be merged correctly.
        cubes = CubeList(
            [
                self._testcube(z=1, dataless=False),
                self._testcube(z=2, dataless=True),
            ]
        )
        cube = cubes.merge_cube()
        assert not cube.is_dataless()
        data_z1, data_z2 = cube[0].data, cube[1].data
        assert np.all(data_z1 == [1, 2, 3])
        assert np.all(np.ma.getmaskarray(data_z2) == True)  # noqa: E712

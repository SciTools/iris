# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.analysis.trajectory.Trajectory`."""

import numpy as np

from iris.analysis.trajectory import Trajectory
from iris.tests.stock import simple_3d, simple_4d_with_hybrid_height


class Test___init__:
    def test_2_points(self):
        # basic 2-seg line along x
        waypoints = [{"lat": 0, "lon": 0}, {"lat": 1, "lon": 2}]
        trajectory = Trajectory(waypoints, sample_count=5)

        assert trajectory.length == np.sqrt(5)
        assert trajectory.sample_count == 5
        assert trajectory.sampled_points == [
            {"lat": 0.0, "lon": 0.0},
            {"lat": 0.25, "lon": 0.5},
            {"lat": 0.5, "lon": 1.0},
            {"lat": 0.75, "lon": 1.5},
            {"lat": 1.0, "lon": 2.0},
        ]

    def test_3_points(self):
        # basic 2-seg line along x
        waypoints = [
            {"lat": 0, "lon": 0},
            {"lat": 0, "lon": 1},
            {"lat": 0, "lon": 2},
        ]
        trajectory = Trajectory(waypoints, sample_count=21)

        assert trajectory.length == 2.0
        assert trajectory.sample_count == 21
        assert trajectory.sampled_points[19] == {"lat": 0.0, "lon": 1.9000000000000001}

    def test_zigzag(self):
        # 4-seg m-shape
        waypoints = [
            {"lat": 0, "lon": 0},
            {"lat": 1, "lon": 1},
            {"lat": 0, "lon": 2},
            {"lat": 1, "lon": 3},
            {"lat": 0, "lon": 4},
        ]
        trajectory = Trajectory(waypoints, sample_count=33)

        assert trajectory.length == 5.6568542494923806
        assert trajectory.sample_count == 33
        assert trajectory.sampled_points[31] == {
            "lat": 0.12499999999999989,
            "lon": 3.875,
        }


class Test__get_interp_points:
    def test_basic(self):
        dim_names = "lat"
        waypoints = [{dim_names: 0}, {dim_names: 1}]
        sample_count = 5
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory._get_interp_points()
        expected_points = list(np.linspace(0, 1, sample_count))

        assert len(result) == len(waypoints[0])
        assert len(result[0][1]) == sample_count
        assert result[0][1] == expected_points
        assert result[0][0] == dim_names

    def test_2d(self):
        dim_names = ["lat", "lon"]
        waypoints = [
            {dim_names[0]: 0, dim_names[1]: 0},
            {dim_names[0]: 1, dim_names[1]: 2},
        ]
        sample_count = 5
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory._get_interp_points()

        assert len(result) == len(waypoints[0])
        assert len(result[0][1]) == sample_count
        assert len(result[1][1]) == sample_count
        assert result[0][0] in dim_names
        assert result[1][0] in dim_names

    def test_3d(self):
        dim_names = ["y", "x", "z"]
        waypoints = [
            {dim_names[0]: 0, dim_names[1]: 0, dim_names[2]: 2},
            {dim_names[0]: 1, dim_names[1]: 2, dim_names[2]: 10},
        ]
        sample_count = 5
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory._get_interp_points()

        assert len(result) == len(waypoints[0])
        assert len(result[0][1]) == sample_count
        assert len(result[1][1]) == sample_count
        assert len(result[2][1]) == sample_count
        assert result[0][0] in dim_names
        assert result[1][0] in dim_names
        assert result[2][0] in dim_names


class Test_interpolate:
    def _result_cube_metadata(self, res_cube):
        dim_names = [c.name() for c in res_cube.dim_coords]
        named_dims = [res_cube.coord_dims(c)[0] for c in res_cube.dim_coords]
        anon_dims = list(set(range(res_cube.ndim)) - set(named_dims))
        anon_dims = None if not len(anon_dims) else anon_dims
        return dim_names, named_dims, anon_dims

    def test_cube__simple_3d(self):
        # Test that an 'index' coord is added to the resultant cube.
        cube = simple_3d()
        waypoints = [
            {"latitude": 40, "longitude": 40},
            {"latitude": 0, "longitude": 0},
        ]
        sample_count = 3
        new_coord_name = "index"
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory.interpolate(cube)

        dim_names, named_dims, anon_dims = self._result_cube_metadata(result)
        new_coord = result.coord(new_coord_name)
        exp_named_dims = [0, 1]

        assert result.ndim == cube.ndim - 1
        assert new_coord_name in dim_names
        assert named_dims == exp_named_dims
        assert anon_dims is None
        assert len(new_coord.points) == sample_count

    def test_cube__anon_dim(self):
        cube = simple_4d_with_hybrid_height()
        cube.remove_coord("model_level_number")  # Make cube dim 1 anonymous.
        waypoints = [
            {"grid_latitude": 21, "grid_longitude": 31},
            {"grid_latitude": 23, "grid_longitude": 33},
        ]
        sample_count = 4
        new_coord_name = "index"
        trajectory = Trajectory(waypoints, sample_count=sample_count)
        result = trajectory.interpolate(cube)

        dim_names, named_dims, anon_dims = self._result_cube_metadata(result)
        new_coord = result.coord(new_coord_name)
        exp_named_dims = [0, 2]
        exp_anon_dims = [1]

        assert result.ndim == cube.ndim - 1
        assert new_coord_name in dim_names
        assert named_dims == exp_named_dims
        assert anon_dims == exp_anon_dims
        assert len(new_coord.points) == sample_count

    def test_call(self, mocker):
        # Test that :func:`iris.analysis.trajectory.interpolate` is called by
        # `Trajectory.interpolate`.
        cube = simple_3d()
        to_patch = "iris.analysis.trajectory.interpolate"
        waypoints = [
            {"latitude": 40, "longitude": 40},
            {"latitude": 0, "longitude": 0},
        ]
        sample_count = 3
        trajectory = Trajectory(waypoints, sample_count=sample_count)

        mock_interpolate = mocker.patch(to_patch, return_value=cube)
        trajectory.interpolate(cube)

        mock_interpolate.assert_called_once()

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.fill_betweenx` function."""

import numpy as np
import pytest

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import _shared_utils

if _shared_utils.MPL_AVAILABLE:
    import cartopy.mpl.geoaxes
    import matplotlib.pyplot as plt

    import iris.plot as iplt


def _profile(positive=None):
    # A vertical profile: two temperature bounds against depth.
    attributes = {} if positive is None else {"positive": positive}
    depth = DimCoord(
        [0.0, 10.0, 20.0], long_name="depth", units="m", attributes=attributes
    )
    x1 = Cube([1.0, 2.0, 3.0], long_name="tmin", units="K")
    x1.add_dim_coord(depth, 0)
    x2 = Cube([4.0, 5.0, 6.0], long_name="tmax", units="K")
    x2.add_dim_coord(depth.copy(), 0)
    return depth, x1, x2


@_shared_utils.skip_plot
class TestFillBetweenx(_shared_utils.GraphicsTest):
    def test_fills_horizontally(self):
        depth, x1, x2 = _profile()
        result = iplt.fill_betweenx(depth, x1, x2)
        vertices = result.get_paths()[0].vertices
        # The coordinate is on the y-axis and the cubes are on the x-axis.
        assert set(vertices[:, 1]) == {0.0, 10.0, 20.0}
        assert set(vertices[:, 0]) == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

    def test_coord_coord(self):
        depth, x1, x2 = _profile()
        x1 = AuxCoord(x1.data, long_name="tmin", units="K")
        x2 = AuxCoord(x2.data, long_name="tmax", units="K")
        result = iplt.fill_betweenx(depth, x1, x2)
        vertices = result.get_paths()[0].vertices
        assert set(vertices[:, 0]) == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

    def test_axes_keyword(self):
        depth, x1, x2 = _profile()
        _, ax = plt.subplots()
        result = iplt.fill_betweenx(depth, x1, x2, axes=ax)
        assert result in ax.collections

    def test_positive_down_inverts_yaxis(self):
        depth, x1, x2 = _profile(positive="down")
        iplt.fill_betweenx(depth, x1, x2)
        assert plt.gca().yaxis_inverted()

    def test_positive_up_keeps_yaxis(self):
        depth, x1, x2 = _profile(positive="up")
        iplt.fill_betweenx(depth, x1, x2)
        assert not plt.gca().yaxis_inverted()

    def test_lat_lon_draws_map(self):
        lat = AuxCoord([0.0, 10.0, 20.0], "latitude", units="degrees")
        lon1 = AuxCoord([0.0, 5.0, 10.0], "longitude", units="degrees")
        lon2 = AuxCoord([20.0, 25.0, 30.0], "longitude", units="degrees")
        iplt.fill_betweenx(lat, lon1, lon2)
        assert isinstance(plt.gca(), cartopy.mpl.geoaxes.GeoAxes)

    def test_incompatible_sizes(self):
        depth, x1, x2 = _profile()
        with pytest.raises(ValueError, match="are not all compatible"):
            iplt.fill_betweenx(depth, x1[:-1], x2)

    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_not_cube_or_coord(self, position):
        args = list(_profile())
        args[position] = np.arange(3)
        with pytest.raises(TypeError, match="must be a cube or a coordinate"):
            iplt.fill_betweenx(*args)

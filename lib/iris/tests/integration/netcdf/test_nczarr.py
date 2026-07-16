# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading and saving zarr files via NcZarr."""

import pytest

import iris
from iris.tests import stock


@pytest.fixture
def realistic_4d_w_everything():
    """Return a realistic 4D cube with all the bells and whistles."""
    return stock.realistic_4d_w_everything()


def test_roundtrip(tmp_path, realistic_4d_w_everything):
    output_path = tmp_path / "output.zarr"
    output_uri = output_path.as_uri() + "#mode=nczarr,file"
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(realistic_4d_w_everything, output_uri)

    # Confirm a Zarr structure.
    assert output_path.is_dir()
    assert (output_path / ".zattrs").is_file()
    assert (output_path / ".zgroup").is_file()

    loaded = iris.load(output_uri)[0]
    # NcZarr (netCDF4-C library) has a precision loss for floating-point
    # scalar variable attributes: earth_radius 6371229.0 round-trips as
    # 6371230.0.  Work around this by replacing the loaded coord-system with
    # the original one so that the comparison focuses on everything else.
    original_cs = realistic_4d_w_everything.coord("grid_latitude").coord_system
    for coord_name in ("grid_latitude", "grid_longitude"):
        loaded.coord(coord_name).coord_system = original_cs

    assert loaded == realistic_4d_w_everything

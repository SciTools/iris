# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading and saving zarr files via NcZarr."""

import json
from pathlib import Path

import pytest

import iris
from iris.coord_systems import GeogCS
from iris.tests import stock


def nczarr_uri(path: Path) -> str:
    return path.as_uri() + "#mode=nczarr,file"


@pytest.fixture
def realistic_4d_w_everything():
    """Return a realistic 4D cube with all the bells and whistles."""
    cube = stock.realistic_4d_w_everything()
    # Precision loss for floating-point scalar variable attributes:
    #  earth_radius 6371229.0 round-trips as 6371230.0.
    for coord_name in ("grid_latitude", "grid_longitude"):
        cube.coord(coord_name).coord_system.ellipsoid = GeogCS(6371230.0)
    return cube


@pytest.fixture
def zarr_file(tmp_path, realistic_4d_w_everything):
    """Return a Zarr file path containing a realistic 4D cube with all the bells and whistles."""
    output_path = tmp_path / "sample.zarr"
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(realistic_4d_w_everything, nczarr_uri(output_path))
    return output_path


def test_roundtrip_cube(tmp_path, realistic_4d_w_everything):
    output_path = tmp_path / "output.zarr"
    output_uri = nczarr_uri(output_path)
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(realistic_4d_w_everything, output_uri)

    # Confirm a Zarr structure.
    assert output_path.is_dir()
    assert (output_path / ".zattrs").is_file()
    assert (output_path / ".zgroup").is_file()

    loaded = iris.load(output_uri)[0]
    assert loaded == realistic_4d_w_everything


def test_roundtrip_file(tmp_path, zarr_file):
    def _get_json(path: Path):
        return sorted(json.load(path.open()))

    loaded = iris.load(nczarr_uri(zarr_file))
    output_path = tmp_path / "output.zarr"
    output_uri = nczarr_uri(output_path)
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(loaded, output_uri)

    for original_file in zarr_file.rglob("*"):
        relative_path = original_file.relative_to(zarr_file)
        output_file = output_path / relative_path
        assert output_file.exists()
        if original_file.name in (".zattrs", ".zgroup"):
            assert _get_json(original_file) == _get_json(output_file)
        elif original_file.is_file():
            assert original_file.read_bytes() == output_file.read_bytes()

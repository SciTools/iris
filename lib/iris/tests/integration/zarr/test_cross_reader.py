# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iris and xarray read each other's Zarr stores.

The check that justifies two decisions in the design. Attributes are written
as JSON-native values, losing their NumPy dtype, *so that* every other Zarr
consumer can read them; and a floating ``_FillValue`` is written as base64,
against CF section 2.5.1, because released xarray fails the whole dataset open
on a plain JSON number.

Both are only worth anything if someone checks, and xarray is the reader to
check against: it is what most of the Zarr world reads with.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")

import iris
from iris.coords import DimCoord
from iris.cube import Cube


def small_cube(dtype="f4", masked=False):
    values = np.arange(6, dtype=dtype).reshape(2, 3)
    if masked:
        values = np.ma.masked_array(values, mask=[[False, True, False]] * 2)
    cube = Cube(values, standard_name="air_temperature", units="K")
    cube.var_name = "air_temperature"
    cube.add_dim_coord(
        DimCoord(
            np.array([0.0, 1.0], dtype="f4"),
            standard_name="latitude",
            units="degrees",
            var_name="latitude",
        ),
        0,
    )
    cube.add_dim_coord(
        DimCoord(
            np.array([0.0, 1.0, 2.0], dtype="f4"),
            standard_name="longitude",
            units="degrees",
            var_name="longitude",
        ),
        1,
    )
    return cube


class TestXarrayReadsIris:
    @pytest.mark.parametrize("dtype", ["f4", "f8", "i2", "i4"])
    def test_a_store_opens(self, tmp_path, dtype):
        # Asserted on the open, not on the values: the failure mode is a
        # TypeError during open that takes the whole dataset with it.
        path = tmp_path / f"out_{dtype}.zarr"
        iris.save(small_cube(dtype=dtype), path)
        dataset = xr.open_zarr(path)
        assert "air_temperature" in dataset.data_vars

    @pytest.mark.parametrize("dtype", ["f4", "f8", "i2"])
    def test_a_masked_store_opens_and_masks(self, tmp_path, dtype):
        path = tmp_path / f"masked_{dtype}.zarr"
        iris.save(small_cube(dtype=dtype, masked=True), path)
        values = xr.open_zarr(path)["air_temperature"].values
        assert np.isnan(values[0, 1])
        assert not np.isnan(values[0, 0])

    def test_an_explicit_finite_sentinel_masks(self, tmp_path):
        path = tmp_path / "sentinel.zarr"
        iris.save(small_cube(masked=True), path, fill_value=np.float32(-99.0))
        values = xr.open_zarr(path)["air_temperature"].values
        assert np.isnan(values[0, 1])

    def test_the_dimensions_and_coordinates_are_xarrays_too(self, tmp_path):
        path = tmp_path / "dims.zarr"
        iris.save(small_cube(), path)
        dataset = xr.open_zarr(path)
        assert dataset["air_temperature"].dims == ("latitude", "longitude")
        np.testing.assert_allclose(dataset["latitude"].values, [0.0, 1.0])

    def test_attributes_survive_as_values(self, tmp_path):
        # A cube attribute not reserved for a data variable is written as a
        # Zarr group attribute, the same split the netCDF saver makes by
        # default, so this reads the dataset's own attrs rather than the
        # data variable's.
        path = tmp_path / "attrs.zarr"
        cube = small_cube()
        cube.attributes["a_number"] = np.float32(1.5)
        cube.attributes["a_list"] = np.array([1, 2], dtype=np.int16)
        cube.attributes["text"] = "words"
        iris.save(cube, path)
        attributes = xr.open_zarr(path).attrs
        assert attributes["a_number"] == 1.5
        assert list(attributes["a_list"]) == [1, 2]
        assert attributes["text"] == "words"

    def test_a_consolidated_store_opens_consolidated(self, tmp_path):
        path = tmp_path / "consolidated.zarr"
        iris.save(small_cube(), path)
        # The default: xarray uses the consolidated document when it is there.
        assert "air_temperature" in xr.open_zarr(path).data_vars


class TestIrisReadsXarray:
    @pytest.mark.parametrize(
        ("fill", "dtype"),
        [(np.nan, "f4"), (-99.0, "f4"), (-9, "i2")],
        ids=["float32-nan", "float32-sentinel", "int16-sentinel"],
    )
    def test_a_store_xarray_wrote_loads_masked(self, tmp_path, fill, dtype):
        path = tmp_path / f"xarray_{dtype}_{fill}.zarr"
        values = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        values[0, 1] = fill if fill == fill else values[0, 1]
        dataset = xr.Dataset(
            {
                "air_temperature": (
                    ("latitude", "longitude"),
                    np.ma.masked_equal(values, fill)
                    if fill == fill
                    else np.where(np.arange(6).reshape(2, 3) == 1, np.nan, values),
                )
            },
            coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0, 2.0]},
        )
        dataset["air_temperature"].encoding["_FillValue"] = fill
        # Consolidation is not what this test is about, and xarray's default
        # warns that it is not yet part of the Zarr version 3 specification.
        dataset.to_zarr(path, zarr_format=3, consolidated=False)
        cube = iris.load_cube(path)
        assert cube.data.mask[0, 1]

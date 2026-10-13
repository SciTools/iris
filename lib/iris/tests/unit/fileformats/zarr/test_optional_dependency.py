# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iris must import, and load netCDF, without zarr-python installed."""

import subprocess
import sys


def _imports_zarr(statement):
    """Run `statement` in a fresh interpreter; report whether zarr was imported."""
    probe = f"{statement}\nimport sys\nprint('zarr' in sys.modules)\n"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip() == "True"


def test_importing_iris_does_not_import_zarr():
    assert not _imports_zarr("import iris")


def test_importing_the_format_registry_does_not_import_zarr():
    # iris.fileformats imports every format module it registers a handler
    # from. If the Zarr loader joined that list, Iris would stop installing
    # without zarr-python.
    assert not _imports_zarr("import iris.fileformats")


def test_importing_the_zarr_package_does_not_import_zarr():
    # The package docstring is reachable without the dependency; only the
    # modules under it need zarr itself.
    assert not _imports_zarr("import iris.fileformats.zarr")


import zarr

from . import stores


def test_make_store_version_3(tmp_path):
    location = stores.make_store(tmp_path / "v3.zarr")
    group = zarr.open_group(store=location, mode="r")
    assert group["air_temperature"].metadata.zarr_format == 3
    assert group["air_temperature"].metadata.dimension_names == ("y", "x")


def test_make_store_version_2(tmp_path):
    location = stores.make_store(tmp_path / "v2.zarr", zarr_format=2)
    group = zarr.open_group(store=location, mode="r")
    array = group["air_temperature"]
    assert array.metadata.zarr_format == 2
    assert array.attrs[stores.ARRAY_DIMENSIONS] == ["y", "x"]

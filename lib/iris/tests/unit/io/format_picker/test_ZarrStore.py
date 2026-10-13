# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Recognising a Zarr store, locally by its marker file and remotely by its URL."""

import pytest

from iris.fileformats import _is_zarr_store
from iris.io.format_picker import ZarrStore
from iris.tests import _shared_utils


def test_the_element_carries_the_whole_target():
    # Not the basename: recognising a local store means reading a file inside
    # it, so the element needs the path, not the last component of it.
    element = ZarrStore()
    assert element.requires_fh is False
    assert element.get_element("/data/store.zarr", None) == "/data/store.zarr"


def test_a_version_3_store_is_recognised(tmp_path):
    store = tmp_path / "v3.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    assert _is_zarr_store(str(store)) is True


def test_a_version_2_store_is_recognised(tmp_path):
    store = tmp_path / "v2.zarr"
    store.mkdir()
    (store / ".zgroup").write_text('{"zarr_format": 2}')
    assert _is_zarr_store(str(store)) is True


def test_a_directory_named_zarr_with_no_marker_is_not_a_store(tmp_path):
    """The suffix alone is not the local test, and must not become one.

    Reading the marker file is what distinguishes a real store from a
    directory someone happened to name `.zarr`, and locally that costs one
    stat.
    """
    store = tmp_path / "pretend.zarr"
    store.mkdir()
    assert _is_zarr_store(str(store)) is False


def test_a_plain_file_is_not_a_store(tmp_path):
    path = tmp_path / "data.nc"
    path.write_bytes(b"CDF\x01")
    assert _is_zarr_store(str(path)) is False


@pytest.mark.parametrize("scheme", ["s3", "gs", "az", "abfs"])
def test_a_remote_store_is_recognised_by_scheme_and_suffix(scheme):
    assert _is_zarr_store(f"{scheme}://bucket/path/data.zarr") is True
    assert _is_zarr_store(f"{scheme}://bucket/path/data.zarr/") is True


def test_a_remote_url_without_the_suffix_is_not_a_store():
    # Remotely the suffix is all there is to go on: reading a marker file
    # would cost a network round trip before Iris knows it wants the target.
    assert _is_zarr_store("s3://bucket/path/data") is False


def test_an_http_url_is_not_a_store():
    # OPeNDAP, which the netCDF loader already claims.
    assert _is_zarr_store("https://example.invalid/data.zarr") is False


def test_an_nczarr_url_is_not_a_native_store(tmp_path):
    """A user who writes `#mode=nczarr` has asked for NCZarr by name.

    The store on disk is a real one, so the marker-file test says yes unless
    the fragment is checked first - and `load_http` passes the fragment
    through to the format agent.
    """
    store = tmp_path / "s.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    assert _is_zarr_store(f"file://{store}#mode=nczarr,file") is False


def test_a_non_string_target_is_not_a_store():
    # `load_data_objects` passes open datasets, not paths.
    assert _is_zarr_store(object()) is False


@_shared_utils.skip_data
def test_the_agent_still_picks_netcdf_for_a_netcdf_file():
    """A `requires_fh=False` spec is consulted for every target with no buffer.

    Adding one risks claiming a target another format already handles, and
    the format agent returns the first match rather than the best.
    """
    from iris.fileformats import FORMAT_AGENT

    path = _shared_utils.get_data_path(
        ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
    )
    with open(path, "rb") as handle:
        spec = FORMAT_AGENT.get_spec("SMALL_hires_wind_u_for_ipcc4.nc", handle)
    assert spec.name.startswith("NetCDF")


def test_the_agent_picks_zarr_for_a_store(tmp_path):
    from iris.fileformats import FORMAT_AGENT

    store = tmp_path / "s.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    assert FORMAT_AGENT.get_spec(str(store), None).name == "Zarr"


def test_the_agent_still_picks_nczarr_for_an_nczarr_url(tmp_path):
    from iris.fileformats import FORMAT_AGENT

    store = tmp_path / "s.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    uri = f"file://{store}#mode=nczarr,file"
    assert FORMAT_AGENT.get_spec(uri, None).name == "NcZarr"

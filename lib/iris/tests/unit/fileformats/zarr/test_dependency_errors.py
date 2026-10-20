# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Both Zarr load routes report a clear error when ``zarr`` cannot be used.

``iris.load`` reaches :mod:`iris.fileformats.zarr` through
:func:`iris.fileformats._load_zarr`; the Zarr page of the user manual also
tells a reader to call :func:`iris.fileformats.zarr.load_cubes` directly, for
the ``group=`` keyword. Both must raise the same ``ImportError``, naming the
package and the version floor, whether ``zarr`` is missing entirely or merely
too old.
"""

import subprocess
import sys

import pytest

import iris
import iris.fileformats.zarr as zarr_package
from iris.fileformats.zarr import _MISSING_MESSAGE

#: Blocks `import zarr` in the interpreter this is run in, without touching
#: the installed distribution's metadata - the only way to simulate "not
#: installed" for the import system while leaving `importlib.metadata`
#: answering as it does on a real machine with zarr present.
_BLOCK_ZARR = """
import sys


class _BlockZarr:
    def find_spec(self, name, path, target=None):
        if name == "zarr" or name.startswith("zarr."):
            raise ImportError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, _BlockZarr())
"""


def _run(statement):
    """Run `statement` in a fresh interpreter; return what it printed."""
    completed = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _report_old_zarr(name):
    """Stand in for `importlib.metadata.version`, answering below the floor."""
    return "2.18.0"


def test_message_names_the_package_and_the_version_floor():
    assert "zarr" in _MISSING_MESSAGE
    assert "3.0.8" in _MISSING_MESSAGE


def test_iris_load_reports_it_when_zarr_is_missing(tmp_path):
    store = tmp_path / "s.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")

    probe = (
        _BLOCK_ZARR
        + f"""
import iris
try:
    iris.load({str(store)!r})
except ImportError as exception:
    print(exception)
"""
    )
    assert _run(probe) == _MISSING_MESSAGE


def test_load_cubes_reports_it_when_zarr_is_missing(tmp_path):
    store = tmp_path / "s.zarr"
    store.mkdir()

    probe = (
        _BLOCK_ZARR
        + f"""
import iris.fileformats.zarr
try:
    iris.fileformats.zarr.load_cubes({str(store)!r})
except ImportError as exception:
    print(exception)
"""
    )
    assert _run(probe) == _MISSING_MESSAGE


def test_iris_load_reports_it_when_zarr_is_too_old(mocker, tmp_path):
    """Zarr 2.x imports fine, so only a version check catches it.

    Without this, `ZarrDataset.__init__` reaches
    `self._group.metadata.zarr_format` against a zarr 2 `Group`, which has no
    `.metadata`, and the user sees a bare `AttributeError` instead.
    """
    mocker.patch.object(zarr_package.importlib.metadata, "version", _report_old_zarr)
    store = tmp_path / "s.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")

    with pytest.raises(ImportError) as excinfo:
        iris.load(str(store))
    assert str(excinfo.value) == _MISSING_MESSAGE


def test_load_cubes_reports_it_when_zarr_is_too_old(mocker):
    mocker.patch.object(zarr_package.importlib.metadata, "version", _report_old_zarr)

    with pytest.raises(ImportError) as excinfo:
        zarr_package.load_cubes("irrelevant")
    assert str(excinfo.value) == _MISSING_MESSAGE

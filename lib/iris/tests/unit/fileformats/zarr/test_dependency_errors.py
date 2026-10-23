# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Every Zarr route reports a clear error when ``zarr`` cannot be used.

``iris.load`` reaches :mod:`iris.fileformats.zarr` through
:func:`iris.fileformats._load_zarr`; the Zarr page of the user manual also
tells a reader to call :func:`iris.fileformats.zarr.load_cubes` directly, for
the ``group=`` keyword; and ``iris.save`` reaches it through the
:func:`iris.io._zarr_save` wrapper. All three must raise the same
``ImportError``, naming the package and the version floor, whether ``zarr``
is missing entirely or merely too old.
"""

import subprocess
import sys

import numpy as np
import pytest

import iris
from iris.cube import Cube
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


#: What the probe subprocess runs, once `_BLOCK_ZARR` has taken zarr away.
#:
#: Kept at module level, and so at column zero, because it is source for
#: another interpreter: indenting it to sit inside a test body would indent
#: the code the subprocess has to compile.
_LOAD_PROBE = """
import iris
try:
    iris.load({location!r})
except ImportError as exception:
    print(exception)
"""

_LOAD_CUBES_PROBE = """
import iris.fileformats.zarr
try:
    iris.fileformats.zarr.load_cubes({location!r})
except ImportError as exception:
    print(exception)
"""


_SAVE_PROBE = """
import numpy as np
import iris
from iris.cube import Cube

cube = Cube(np.arange(4, dtype=np.float32), var_name="a")
try:
    iris.save(cube, {location!r})
except ImportError as exception:
    print(exception)
"""


class Test__MISSING_MESSAGE:
    def test_it_names_the_package_and_the_version_floor(self):
        assert "zarr" in _MISSING_MESSAGE
        assert "3.0.8" in _MISSING_MESSAGE

    def test_it_covers_saving_as_well_as_loading(self):
        # `iris.save` raises this one too, so it cannot be worded as a
        # loading failure: a user who was saving would be told about the
        # wrong half of the package.
        assert "load" not in _MISSING_MESSAGE


class Test_zarr_is_missing:
    def test_iris_load_reports_it(self, tmp_path):
        store = tmp_path / "s.zarr"
        store.mkdir()
        (store / "zarr.json").write_text("{}")

        probe = _BLOCK_ZARR + _LOAD_PROBE.format(location=str(store))
        assert _run(probe) == _MISSING_MESSAGE

    def test_load_cubes_reports_it(self, tmp_path):
        store = tmp_path / "s.zarr"
        store.mkdir()

        probe = _BLOCK_ZARR + _LOAD_CUBES_PROBE.format(location=str(store))
        assert _run(probe) == _MISSING_MESSAGE

    def test_iris_save_reports_it(self, tmp_path):
        probe = _BLOCK_ZARR + _SAVE_PROBE.format(location=str(tmp_path / "out.zarr"))
        assert _run(probe) == _MISSING_MESSAGE


class Test_zarr_is_too_old:
    def test_iris_load_reports_it(self, mocker, tmp_path):
        """Zarr 2.x imports fine, so only a version check catches it.

        Without this, `ZarrDataset.__init__` reaches
        `self._group.metadata.zarr_format` against a zarr 2 `Group`, which
        has no `.metadata`, and the user sees a bare `AttributeError`
        instead.
        """
        mocker.patch.object(
            zarr_package.importlib.metadata, "version", _report_old_zarr
        )
        store = tmp_path / "s.zarr"
        store.mkdir()
        (store / "zarr.json").write_text("{}")

        with pytest.raises(ImportError) as excinfo:
            iris.load(str(store))
        assert str(excinfo.value) == _MISSING_MESSAGE

    def test_load_cubes_reports_it(self, mocker):
        mocker.patch.object(
            zarr_package.importlib.metadata, "version", _report_old_zarr
        )

        with pytest.raises(ImportError) as excinfo:
            zarr_package.load_cubes("irrelevant")
        assert str(excinfo.value) == _MISSING_MESSAGE

    def test_iris_save_reports_it(self, mocker, tmp_path):
        """The save path must use the same version gate as the load path.

        :func:`iris.io._zarr_save` gets there with ``from
        iris.fileformats.zarr import save``, which runs this package's
        ``__getattr__`` and so the version check inside it. Importing from
        ``.saver`` instead would reach the saver directly, and an
        installation with too old a zarr would write a complete store rather
        than raising.
        """
        mocker.patch.object(
            zarr_package.importlib.metadata, "version", _report_old_zarr
        )
        cube = Cube(np.arange(4, dtype=np.float32), var_name="a")

        with pytest.raises(ImportError) as excinfo:
            iris.save(cube, tmp_path / "out.zarr")
        assert str(excinfo.value) == _MISSING_MESSAGE

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the public surface of :mod:`iris.fileformats.cf`.

The package is a re-export layer over a private file layout, so what needs
testing is the layer itself: that it exposes everything it used to, exposes
nothing it should not, and imports cleanly on its own.
"""

import subprocess
import sys
from types import ModuleType

import pytest

import iris.fileformats.cf as cf
from iris.fileformats.cf import _group, _reader, _variables

PRIVATE_MODULES = (_variables, _group, _reader)


def _origin(value, fallback):
    """Return the module path a name came from, for values and modules alike."""
    if isinstance(value, ModuleType):
        return value.__name__
    return getattr(value, "__module__", fallback)


def _defined_public_names(module):
    """Return the public names a module defines itself, ignoring imports."""
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, ModuleType)
        and _origin(value, module.__name__) == module.__name__
    }


@pytest.mark.parametrize("module", PRIVATE_MODULES, ids=lambda m: m.__name__)
def test_public_names_are_re_exported(module):
    # Review Focus 1. A name defined in a private module and left out of
    # __all__ vanishes from the API docs without any warning, because autodoc
    # rejects it on __module__ - see the plan, section 3.2.
    assert _defined_public_names(module) <= set(cf.__all__)


@pytest.mark.parametrize("name", cf.__all__)
def test_all_entries_resolve(name):
    # Review Focus 1, the other direction: a name in __all__ that the package
    # cannot supply breaks "from iris.fileformats.cf import *" and the docs.
    assert hasattr(cf, name)


@pytest.mark.parametrize("module", [_variables, _group], ids=lambda m: m.__name__)
def test_only_the_reader_is_coupled_to_netcdf(module):
    # Review Focus 3. Confining the netCDF import to _reader is the whole
    # point of the split; F401 is disabled here, so a stale import left
    # behind by the cut would otherwise go unremarked.
    coupled = {
        name
        for name, value in vars(module).items()
        if _origin(value, "").startswith("iris.fileformats.netcdf")
    }
    assert coupled == set()


@pytest.mark.parametrize(
    "first_import",
    ["iris.fileformats.cf", "iris.fileformats.netcdf"],
)
def test_imports_in_a_fresh_interpreter(first_import):
    # Review Focus 4. cf and netcdf import each other, and the split adds a
    # hop to that cycle. pytest has already imported iris by the time this
    # module is collected, so a new process is the only way to see it.
    code = f"import {first_import}\nimport iris.fileformats.cf"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "name",
    ["_CFFormulaTermsVariable", "_is_str_dtype", "_getncattr", "_CF_PARSE"],
)
def test_private_names_are_not_re_exported(name):
    # Review Focus 5. These were reachable on the flat module and are
    # deliberately not reachable on the package - spec section 4.1.
    assert not hasattr(cf, name)

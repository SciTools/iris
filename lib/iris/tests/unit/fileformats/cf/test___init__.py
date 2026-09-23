# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the public surface of :mod:`iris.fileformats.cf`.

The package is a re-export layer over a private file layout, so what needs
testing is the layer itself: that it exposes everything it used to, exposes
nothing it should not, and imports cleanly on its own.
"""

import ast
import inspect
from pathlib import Path
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
    """Return the public names a module's own source binds at the top level.

    Read from the source rather than from ``vars()``, because ``__module__``
    does not exist at all on data members such as ``reference_terms``. A
    ``vars()`` walk would therefore have to skip exactly the kind of name that
    has already gone astray once - see
    ``test_data_members_are_assigned_in_the_package_source``.
    """
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


@pytest.mark.parametrize("module", PRIVATE_MODULES, ids=lambda m: m.__name__)
def test_public_names_are_re_exported(module):
    # Review Focus 1. A name defined in a private module and left out of
    # __all__ vanishes from the API docs without any warning, because autodoc
    # rejects it on __module__ - see the plan, section 3.2.
    assert _defined_public_names(module) <= set(cf.__all__)


@pytest.mark.parametrize("name", cf.__all__)
def test_re_exported_classes_keep_their_source(name):
    # Rewriting __module__ to the package - a tempting way to hide the split
    # from repr() - makes inspect look for the class body in __init__.py,
    # where it is not. That breaks getsource, IPython's "??", debuggers, and
    # silently drops the "[source]" link from every class on the API page,
    # with no warning in the docs build to say so. See the plan, section 8.4.
    value = getattr(cf, name)
    if isinstance(value, type):
        assert inspect.getsource(value).lstrip().startswith("class ")


def test_data_members_are_assigned_in_the_package_source():
    # Sphinx autodoc documents module data only where the module's own source
    # assigns it, and drops an imported name silently. Anything in __all__
    # that is not a class therefore has to be re-stated in cf/__init__.py, or
    # it disappears from the API reference with no warning at all.
    assigned = _defined_public_names(cf)
    data_names = {
        name for name in cf.__all__ if not isinstance(getattr(cf, name), type)
    }
    assert data_names <= assigned


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
        # An import deadlock is one of the failures this is looking for, and
        # without a timeout it would hang the suite instead of failing it.
        timeout=120,
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

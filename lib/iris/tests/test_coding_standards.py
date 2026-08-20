# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import ast
from datetime import datetime
from fnmatch import fnmatch
from glob import glob
import os
from pathlib import Path
import subprocess
from typing import Iterator, List, Tuple, cast

from packaging.version import Version
import pytest

import iris
from iris.tests import system_test
from iris.tests.unit.fileformats.netcdf import test_bytecoding_datasets

LICENSE_TEMPLATE = """# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details."""

# Guess iris repo directory of Iris - realpath is used to mitigate against
# Python finding the iris package via a symlink.
IRIS_DIR = Path(iris.__file__).parent.resolve()
IRIS_INSTALL_DIR = Path(IRIS_DIR).parent.parent
DOCS_DIR = Path(IRIS_INSTALL_DIR) / "docs" / "iris"
DOCS_DIR = iris.config.get_option("Resources", "doc_dir", default=str(DOCS_DIR))
exclusion = ["Makefile", "build"]
DOCS_DIRS = glob(str(Path(DOCS_DIR) / "*"))
DOCS_DIRS = [DOC_DIR for DOC_DIR in DOCS_DIRS if Path(DOC_DIR).name not in exclusion]

# Get a dirpath to the git repository : allow setting with an environment
# variable, so Travis can test for headers in the repo, not the installation.
IRIS_REPO_DIRPATH = os.environ.get("IRIS_REPO_DIR", IRIS_INSTALL_DIR)


def test_netcdf4_import():
    """Use of netCDF4 must be via iris.fileformats.netcdf._thread_safe_nc ."""
    # Please avoid including these phrases in any comments/strings throughout
    #  Iris (e.g. use "from the netCDF4 library" instead) - this allows the
    #  below search to remain quick and simple.
    from iris.fileformats.netcdf import _thread_safe_nc
    from iris.tests.unit.fileformats.netcdf._thread_safe_nc import test_NetCDFWriteProxy

    import_strings = ("import netCDF4", "from netCDF4")

    files_including_import = []
    for file_path in Path(IRIS_DIR).rglob("*.py"):
        file_text = file_path.read_text()

        if any([i in file_text for i in import_strings]):
            files_including_import.append(file_path)

    expected = [
        Path(_thread_safe_nc.__file__),
        Path(test_NetCDFWriteProxy.__file__),
        Path(system_test.__file__),
        Path(__file__),
        Path(test_bytecoding_datasets.__file__),
    ]
    assert set(files_including_import) == set(expected)


def test_python_versions() -> None:
    """Test Python Versions.

    Test is designed to fail whenever Iris' supported Python versions are
    updated, insisting that versions are updated EVERYWHERE in-sync.
    """
    all_supported = ["3.12", "3.13", "3.14"]
    _parsed = [Version(v) for v in all_supported]
    latest_supported = str(max(_parsed))

    root_dir = Path(__file__).parents[3]
    workflows_dir = root_dir / ".github" / "workflows"
    benchmarks_dir = root_dir / "benchmarks"

    # Places that are checked:
    pyproject_toml_file = root_dir / "pyproject.toml"
    requirements_dir = root_dir / "requirements"
    nox_file = root_dir / "noxfile.py"
    ci_wheels_file = workflows_dir / "ci-wheels.yml"
    ci_tests_file = workflows_dir / "ci-tests.yml"
    benchmark_runner_file = benchmarks_dir / "bm_runner.py"

    text_searches: List[Tuple[Path, str]] = [
        (
            pyproject_toml_file,
            "\n    ".join(
                [f'"Programming Language :: Python :: {ver}",' for ver in all_supported]
            ),
        ),
        (
            nox_file,
            "_PY_VERSIONS_ALL = [" + ", ".join([f'"{ver}"' for ver in all_supported]),
        ),
        (
            ci_wheels_file,
            "python-version: [" + ", ".join([f'"{ver}"' for ver in all_supported]),
        ),
        (
            ci_tests_file,
            (
                f'python-version: ["{latest_supported}"]\n'
                f'{" " * 8}session: ["doctest", "gallery"]'
            ),
        ),
        (benchmark_runner_file, f'python_version = "{latest_supported}"'),
    ]

    for ver in all_supported:
        req_yaml = requirements_dir / f"py{ver.replace('.', '')}.yml"
        text_searches.append((req_yaml, f"- python ={ver}"))

        text_searches.append(
            (
                ci_tests_file,
                f'python-version: "{ver}"\n{" " * 12}session: "tests"',
            )
        )

    for path, search in text_searches:
        assert search in path.read_text()


def test_categorised_warnings() -> None:
    r"""To ensure that all UserWarnings raised by Iris are categorised, for ease of use.

    No obvious category? Use the parent:
    :class:`iris.warnings.IrisUserWarning`.

    Warning matches multiple categories? Create a one-off combo class. For
    example:

    .. code-block:: python

        class _WarnComboCfDefaulting(IrisCfWarning, IrisDefaultingWarning):
            \"\"\"
            One-off combination of warning classes - enhances user filtering.
            \"\"\"
            pass

    """
    warns_without_category = []
    warns_with_user_warning = []
    tmp_list = []

    for file_path in Path(IRIS_DIR).rglob("*.py"):
        file_text = file_path.read_text()
        parsed = ast.parse(source=file_text)
        calls: Iterator[ast.Call] = cast(
            "Iterator[ast.Call]",
            filter(lambda node: hasattr(node, "func"), ast.walk(parsed)),
        )
        warn_calls: Iterator[ast.Call] = filter(
            lambda c: getattr(c.func, "attr", None) == "warn", calls
        )

        warn_call: ast.Call
        for warn_call in warn_calls:
            warn_ref = f"{file_path}:{warn_call.lineno}"
            tmp_list.append(warn_ref)

            category_kwargs = filter(lambda k: k.arg == "category", warn_call.keywords)
            category_kwarg: ast.keyword | None = next(category_kwargs, None)

            if category_kwarg is None:
                warns_without_category.append(warn_ref)
            # Work with Attribute or Name instances.
            elif (
                getattr(category_kwarg.value, "attr", None)
                or getattr(category_kwarg.value, "id", None)
            ) == "UserWarning":
                warns_with_user_warning.append(warn_ref)

    # This avoids UserWarnings being raised by unwritten default behaviour.
    assert warns_without_category == [], (
        "All warnings raised by Iris must be raised with the category kwarg."
    )

    assert warns_with_user_warning == [], (
        "No warnings raised by Iris can be the base UserWarning class."
    )

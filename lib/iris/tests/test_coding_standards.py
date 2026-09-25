# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import importlib.util
import os
from pathlib import Path

import pytest

import iris

# Guess iris repo directory of Iris - realpath is used to mitigate against
# Python finding the iris package via a symlink.
IRIS_DIR = os.path.realpath(os.path.dirname(iris.__file__))
IRIS_INSTALL_DIR = os.path.dirname(os.path.dirname(IRIS_DIR))
# Get a dirpath to the git repository : allow setting with an environment
# variable, so Travis can test for headers in the repo, not the installation.
IRIS_REPO_DIRPATH = os.environ.get("IRIS_REPO_DIR", IRIS_INSTALL_DIR)


def _load_hook(hook_name: str):
    """Load a hook module from .hooks/.

    Args:
        hook_name: Name of the hook module (without .py extension).

    Returns
    -------
        The loaded module.

    Raises
    ------
        RuntimeError: If the hook module cannot be loaded.
    """
    hook_path = Path(__file__).parents[3] / ".hooks" / f"{hook_name}.py"
    spec = importlib.util.spec_from_file_location(hook_name, hook_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hook: {hook_name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_netcdf4_import():
    """Use of netCDF4 must be via iris.fileformats.netcdf._thread_safe_nc ."""
    # Logic lives in .hooks/check_netcdf4_imports.py (also used as a pre-commit hook).
    check_netcdf4_imports = _load_hook("check_netcdf4_imports")
    check_file = check_netcdf4_imports.check_file

    all_violations = []
    for file_path in Path(IRIS_DIR).rglob("*.py"):
        all_violations.extend(check_file(file_path))

    message = (
        "The following files import netCDF4 directly, which is not allowed:\n"
        + "\n".join(all_violations)
        + "\nAll netCDF4 imports must be via iris.fileformats.netcdf._thread_safe_nc."
    )
    assert not all_violations, message


def test_python_versions():
    """Test Python Versions.

    Test is designed to fail whenever Iris' supported Python versions are
    updated, insisting that versions are updated EVERYWHERE in-sync.

    Logic lives in .hooks/check_python_versions.py (also used as a pre-commit hook).
    """
    check_python_versions = _load_hook("check_python_versions")
    repo_root = Path(IRIS_REPO_DIRPATH)

    violations = check_python_versions.check_consistency(repo_root)

    message = (
        "Python version consistency check failed.\n"
        "Python versions must be updated consistently across all config files.\n"
        + "\n".join(violations)
    )
    assert not violations, message


def test_categorised_warnings():
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

    Logic lives in .hooks/check_categorised_warnings.py (also used as a pre-commit hook).
    """
    check_categorised_warnings = _load_hook("check_categorised_warnings")
    check_file = check_categorised_warnings.check_file

    all_violations = []
    for file_path in Path(IRIS_DIR).rglob("*.py"):
        all_violations.extend(check_file(file_path))

    message = "The following files have warning categorisation issues:\n" + "\n".join(
        all_violations
    )
    assert not all_violations, message


def test_license_headers():
    """Check that all Python files have the required license header.

    Logic lives in .hooks/check_license_headers.py (also used as a pre-commit hook).
    """
    check_license_headers = _load_hook("check_license_headers")
    repo_root = Path(IRIS_REPO_DIRPATH)

    all_violations: list[str] = []
    for file_path in check_license_headers._get_all_tracked_files(repo_root):
        all_violations.extend(check_license_headers.check_file(file_path, repo_root))

    message = (
        "The following files are missing or have incorrect license headers:\n"
        + "\n".join(all_violations)
    )
    assert not all_violations, message


_DOCS_PAGE = """\
.. z_reference:: Phrasebook
   :tags: topic_interoperability

   Information on terminology differences between Iris and similar packages.

.. _phrasebook:

Package Phrasebook
==================

.. readingtime::

Body text.
"""
"""A documentation page carrying every piece of metadata Iris requires."""


def test_docs_page_metadata():
    """Check that all documentation pages carry their required metadata.

    Logic lives in .hooks/check_docs_page_metadata.py (also used as a
    pre-commit hook, and by the docs build via the page_metadata_validator
    Sphinx extension).
    """
    check_docs_page_metadata = _load_hook("check_docs_page_metadata")
    docs_src = Path(IRIS_REPO_DIRPATH) / "docs" / "src"

    all_violations = check_docs_page_metadata.check_tree(docs_src)

    message = "The following documentation pages are missing metadata:\n" + "\n".join(
        all_violations
    )
    assert not all_violations, message


def test_docs_page_metadata_clean():
    """A fully populated page must be accepted, or the checks below prove nothing."""
    check_docs_page_metadata = _load_hook("check_docs_page_metadata")

    problems = check_docs_page_metadata.check_page(
        "user_manual/reference/phrasebook", _DOCS_PAGE
    )

    assert problems == []


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        pytest.param(
            _DOCS_PAGE.replace(".. readingtime::\n", ""),
            "Missing '.. readingtime::' directive",
            id="missing_readingtime",
        ),
        pytest.param(
            _DOCS_PAGE.replace(".. z_reference:: Phrasebook", "Phrasebook"),
            "found 0.",
            id="missing_item",
        ),
        pytest.param(
            _DOCS_PAGE + _DOCS_PAGE,
            "found 2.",
            id="duplicate_item",
        ),
        pytest.param(
            _DOCS_PAGE.replace(".. z_reference::", ".. tutorial::"),
            "found type 'tutorial'",
            id="wrong_type",
        ),
        pytest.param(
            "\n" * 30 + _DOCS_PAGE,
            "within first 25 lines",
            id="item_too_far_down",
        ),
        pytest.param(
            _DOCS_PAGE.replace(
                "   Information on terminology differences between Iris and"
                " similar packages.\n",
                "",
            ),
            "non-empty content section",
            id="empty_content",
        ),
        pytest.param(
            _DOCS_PAGE.replace(":tags: topic_interoperability", ":tags: data-model"),
            "'topic_xxx' tag",
            id="missing_topic_tag",
        ),
    ],
)
def test_docs_page_metadata_detects(page, expected):
    """Each way of breaking a page's metadata must be detected.

    These are the faults that would otherwise only surface minutes into a
    documentation build.
    """
    check_docs_page_metadata = _load_hook("check_docs_page_metadata")

    problems = check_docs_page_metadata.check_page(
        "user_manual/reference/phrasebook", page
    )

    assert any(expected in problem for problem in problems), (
        f"Expected a problem containing {expected!r}; got: {problems}"
    )

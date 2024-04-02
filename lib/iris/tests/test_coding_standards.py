# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

# import iris.tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import ast
from datetime import datetime
from fnmatch import fnmatch
from glob import glob
import os
from pathlib import Path
import subprocess
from typing import List, Tuple

import iris
from iris.fileformats.netcdf import _thread_safe_nc
from iris.tests import system_test

LICENSE_TEMPLATE = """# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details."""

# Guess iris repo directory of Iris - realpath is used to mitigate against
# Python finding the iris package via a symlink.
IRIS_DIR = os.path.realpath(os.path.dirname(iris.__file__))
IRIS_INSTALL_DIR = os.path.dirname(os.path.dirname(IRIS_DIR))
DOCS_DIR = os.path.join(IRIS_INSTALL_DIR, "docs", "iris")
DOCS_DIR = iris.config.get_option("Resources", "doc_dir", default=DOCS_DIR)
exclusion = ["Makefile", "build"]
DOCS_DIRS = glob(os.path.join(DOCS_DIR, "*"))
DOCS_DIRS = [
    DOC_DIR for DOC_DIR in DOCS_DIRS if os.path.basename(DOC_DIR) not in exclusion
]
# Get a dirpath to the git repository : allow setting with an environment
# variable, so Travis can test for headers in the repo, not the installation.
IRIS_REPO_DIRPATH = os.environ.get("IRIS_REPO_DIR", IRIS_INSTALL_DIR)


def test_netcdf4_import():
    """Use of netCDF4 must be via iris.fileformats.netcdf._thread_safe_nc ."""
    # Please avoid including these phrases in any comments/strings throughout
    #  Iris (e.g. use "from the netCDF4 library" instead) - this allows the
    #  below search to remain quick and simple.
    import_strings = ("import netCDF4", "from netCDF4")

    files_including_import = []
    for file_path in Path(IRIS_DIR).rglob("*.py"):
        file_text = file_path.read_text()

        if any([i in file_text for i in import_strings]):
            files_including_import.append(file_path)

    expected = [
        Path(_thread_safe_nc.__file__),
        Path(system_test.__file__),
        Path(__file__),
    ]
    assert set(files_including_import) == set(expected)


def test_python_versions():
    """Test Python Versions.

    Test is designed to fail whenever Iris' supported Python versions are
    updated, insisting that versions are updated EVERYWHERE in-sync.
    """
    latest_supported = "3.12"
    all_supported = ["3.10", "3.11", latest_supported]

    root_dir = Path(__file__).parents[3]
    workflows_dir = root_dir / ".github" / "workflows"
    benchmarks_dir = root_dir / "benchmarks"

    # Places that are checked:
    pyproject_toml_file = root_dir / "pyproject.toml"
    requirements_dir = root_dir / "requirements"
    nox_file = root_dir / "noxfile.py"
    ci_wheels_file = workflows_dir / "ci-wheels.yml"
    ci_tests_file = workflows_dir / "ci-tests.yml"
    asv_config_file = benchmarks_dir / "asv.conf.json"
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
                f'{" " * 8}session: ["doctest", "gallery", "linkcheck"]'
            ),
        ),
        (asv_config_file, f"PY_VER={latest_supported}"),
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

    """
    warns_without_category = []
    warns_with_user_warning = []
    tmp_list = []

    for file_path in Path(IRIS_DIR).rglob("*.py"):
        file_text = file_path.read_text()
        parsed = ast.parse(source=file_text)
        calls = filter(lambda node: hasattr(node, "func"), ast.walk(parsed))
        warn_calls = filter(lambda c: getattr(c.func, "attr", None) == "warn", calls)

        warn_call: ast.Call
        for warn_call in warn_calls:
            warn_ref = f"{file_path}:{warn_call.lineno}"
            tmp_list.append(warn_ref)

            category_kwargs = filter(lambda k: k.arg == "category", warn_call.keywords)
            category_kwarg: ast.keyword = next(category_kwargs, None)

            if category_kwarg is None:
                warns_without_category.append(warn_ref)
            # Work with Attribute or Name instances.
            elif (
                getattr(category_kwarg.value, "attr", None)
                or getattr(category_kwarg.value, "id", None)
            ) == "UserWarning":
                warns_with_user_warning.append(warn_ref)

    # This avoids UserWarnings being raised by unwritten default behaviour.
    assert (
        warns_without_category == []
    ), "All warnings raised by Iris must be raised with the category kwarg."

    assert (
        warns_with_user_warning == []
    ), "No warnings raised by Iris can be the base UserWarning class."


class TestLicenseHeaders(tests.IrisTest):
    @staticmethod
    def whatchanged_parse(whatchanged_output):
        r"""Returns a generator of tuples of data parsed from
        "git whatchanged --pretty='TIME:%at". The tuples are of the form
        ``(filename, last_commit_datetime)``.

        Sample input::

            ['TIME:1366884020', '',
             ':000000 100644 0000000... 5862ced... A\tlib/iris/cube.py']

        """
        dt = None
        for line in whatchanged_output:
            if not line.strip():
                continue
            elif line.startswith("TIME:"):
                dt = datetime.fromtimestamp(int(line[5:]))
            else:
                # Non blank, non date, line -> must be the lines
                # containing the file info.
                fname = " ".join(line.split("\t")[1:])
                yield fname, dt

    @staticmethod
    def last_change_by_fname():
        """Return a dictionary of all the files under git which maps to
        the datetime of their last modification in the git history.

        .. note::

            This function raises a ValueError if the repo root does
            not have a ".git" folder. If git is not installed on the system,
            or cannot be found by subprocess, an IOError may also be raised.

        """
        # Check the ".git" folder exists at the repo dir.
        if not os.path.isdir(os.path.join(IRIS_REPO_DIRPATH, ".git")):
            msg = "{} is not a git repository."
            raise ValueError(msg.format(IRIS_REPO_DIRPATH))

        # Call "git whatchanged" to get the details of all the files and when
        # they were last changed.
        output = subprocess.check_output(
            ["git", "whatchanged", "--pretty=TIME:%ct"], cwd=IRIS_REPO_DIRPATH
        )

        output = output.decode().split("\n")
        res = {}
        for fname, dt in TestLicenseHeaders.whatchanged_parse(output):
            if fname not in res or dt > res[fname]:
                res[fname] = dt

        return res

    def test_license_headers(self):
        exclude_patterns = (
            "setup.py",
            "noxfile.py",
            "build/*",
            "dist/*",
            "docs/gallery_code/*/*.py",
            "docs/src/developers_guide/documenting/*.py",
            "docs/src/userguide/plotting_examples/*.py",
            "docs/src/userguide/regridding_plots/*.py",
            "docs/src/_build/*",
            "lib/iris/analysis/_scipy_interpolate.py",
        )

        try:
            last_change_by_fname = self.last_change_by_fname()
        except ValueError as err:
            # Caught the case where this is not a git repo.
            msg = "Iris installation did not look like a git repo?\nERR = {}\n\n"
            return self.skipTest(msg.format(str(err)))

        failed = False
        for fname, last_change in sorted(last_change_by_fname.items()):
            full_fname = os.path.join(IRIS_REPO_DIRPATH, fname)
            if (
                full_fname.endswith(".py")
                and os.path.isfile(full_fname)
                and not any(fnmatch(fname, pat) for pat in exclude_patterns)
            ):
                with open(full_fname) as fh:
                    content = fh.read()
                    if content.startswith("#!"):
                        # account for files with leading shebang directives
                        # i.e., first strip out the shebang line before
                        # then performing license header compliance checking
                        content = "\n".join(content.split("\n")[1:])
                    if not content.startswith(LICENSE_TEMPLATE):
                        print(
                            "The file {} does not start with the required "
                            "license header.".format(fname)
                        )
                        failed = True

        if failed:
            raise ValueError("There were license header failures. See stdout.")


if __name__ == "__main__":
    tests.main()

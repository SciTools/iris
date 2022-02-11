"""
Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import nox
from nox.logger import logger

#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: Python versions we can run sessions under
_PY_VERSIONS_ALL = ["3.8"]
_PY_VERSION_LATEST = _PY_VERSIONS_ALL[-1]

#: One specific python version for docs builds
_PY_VERSION_DOCSBUILD = _PY_VERSION_LATEST

#: Cirrus-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", _PY_VERSIONS_ALL)

#: Default cartopy cache directory.
CARTOPY_CACHE_DIR = os.environ.get("HOME") / Path(".local/share/cartopy")

# https://github.com/numpy/numpy/pull/19478
# https://github.com/matplotlib/matplotlib/pull/22099
#: Common session environment variables.
ENV = dict(
    NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512VL,AVX512BW,AVX512DQ,AVX512_SKX"
)


def session_lockfile(session: nox.sessions.Session) -> Path:
    """Return the path of the session lockfile."""
    return Path(
        f"requirements/ci/nox.lock/py{session.python.replace('.', '')}-linux-64.lock"
    )


def session_cachefile(session: nox.sessions.Session) -> Path:
    """Returns the path of the session lockfile cache."""
    lockfile = session_lockfile(session)
    tmp_dir = Path(session.create_tmp())
    cache = tmp_dir / lockfile.name
    return cache


def venv_populated(session: nox.sessions.Session) -> bool:
    """Returns True if the conda venv has been created
    and the list of packages in the lockfile installed."""
    return session_cachefile(session).is_file()


def venv_changed(session: nox.sessions.Session) -> bool:
    """Returns True if the installed session is different to that specified
    in the lockfile."""
    changed = False
    cache = session_cachefile(session)
    lockfile = session_lockfile(session)
    if cache.is_file():
        with open(lockfile, "rb") as fi:
            expected = hashlib.sha256(fi.read()).hexdigest()
        with open(cache, "r") as fi:
            actual = fi.read()
        changed = actual != expected
    return changed


def cache_venv(session: nox.sessions.Session) -> None:
    """
    Cache the nox session environment.

    This consists of saving a hexdigest (sha256) of the associated
    conda lock file.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    lockfile = session_lockfile(session)
    cache = session_cachefile(session)
    with open(lockfile, "rb") as fi:
        hexdigest = hashlib.sha256(fi.read()).hexdigest()
    with open(cache, "w") as fo:
        fo.write(hexdigest)


def cache_cartopy(session: nox.sessions.Session) -> None:
    """
    Determine whether to cache the cartopy natural earth shapefiles.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    if not CARTOPY_CACHE_DIR.is_dir():
        session.run_always(
            "python",
            "-c",
            "import cartopy; cartopy.io.shapereader.natural_earth()",
        )


def prepare_venv(session: nox.sessions.Session) -> None:
    """
    Create and cache the nox session conda environment, and additionally
    provide conda environment package details and info.

    Note that, iris is installed into the environment using pip.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    Notes
    -----
    See
      - https://github.com/theacodes/nox/issues/346
      - https://github.com/theacodes/nox/issues/260

    """
    lockfile = session_lockfile(session)
    venv_dir = session.virtualenv.location_name

    if not venv_populated(session):
        # environment has been created but packages not yet installed
        # populate the environment from the lockfile
        logger.debug(f"Populating conda env at {venv_dir}")
        session.conda_install("--file", str(lockfile))
        cache_venv(session)

    elif venv_changed(session):
        # destroy the environment and rebuild it
        logger.debug(f"Lockfile changed. Re-creating conda env at {venv_dir}")
        _re_orig = session.virtualenv.reuse_existing
        session.virtualenv.reuse_existing = False
        session.virtualenv.create()
        session.conda_install("--file", str(lockfile))
        session.virtualenv.reuse_existing = _re_orig
        cache_venv(session)

    logger.debug(f"Environment {venv_dir} is up to date")

    cache_cartopy(session)

    # Determine whether verbose diagnostics have been requested
    # from the command line.
    verbose = "-v" in session.posargs or "--verbose" in session.posargs

    if verbose:
        session.run_always("conda", "info")
        session.run_always("conda", "list", f"--prefix={venv_dir}")
        session.run_always(
            "conda",
            "list",
            f"--prefix={venv_dir}",
            "--explicit",
        )


@nox.session
def precommit(session: nox.sessions.Session):
    """
    Perform pre-commit hooks of iris codebase.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    import yaml

    # Pip install the session requirements.
    session.install("pre-commit")

    # Load the pre-commit configuration YAML file.
    with open(".pre-commit-config.yaml", "r") as fi:
        config = yaml.load(fi, Loader=yaml.FullLoader)

    # List of pre-commit hook ids that we don't want to run.
    excluded = ["no-commit-to-branch"]

    # Enumerate the ids of pre-commit hooks we do want to run.
    ids = [
        hook["id"]
        for entry in config["repos"]
        for hook in entry["hooks"]
        if hook["id"] not in excluded
    ]

    # Execute the pre-commit hooks.
    [session.run("pre-commit", "run", "--all-files", id) for id in ids]


@nox.session(python=PY_VER, venv_backend="conda")
def tests(session: nox.sessions.Session):
    """
    Perform iris system, integration and unit tests.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.install("--no-deps", "--editable", ".")
    session.env.update(ENV)
    session.run(
        "python",
        "-m",
        "iris.tests.runner",
        "--default-tests",
        "--system-tests",
    )


@nox.session(python=_PY_VERSION_DOCSBUILD, venv_backend="conda")
def doctest(session: nox.sessions.Session):
    """
    Perform iris doctests and gallery.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.install("--no-deps", "--editable", ".")
    session.env.update(ENV)
    session.cd("docs")
    session.run(
        "make",
        "clean",
        "html",
        external=True,
    )
    session.run(
        "make",
        "doctest",
        external=True,
    )
    session.cd("..")
    session.run(
        "python",
        "-m",
        "iris.tests.runner",
        "--gallery-tests",
    )


@nox.session(python=_PY_VERSION_DOCSBUILD, venv_backend="conda")
def linkcheck(session: nox.sessions.Session):
    """
    Perform iris doc link check.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.install("--no-deps", "--editable", ".")
    session.cd("docs")
    session.run(
        "make",
        "clean",
        "html",
        external=True,
    )
    session.run(
        "make",
        "linkcheck",
        external=True,
    )


@nox.session(python=PY_VER[-1], venv_backend="conda")
@nox.parametrize(
    "run_type",
    ["overnight", "branch", "custom"],
    ids=["overnight", "branch", "custom"],
)
def benchmarks(
    session: nox.sessions.Session,
    run_type: Literal["overnight", "branch", "custom"],
):
    """
    Perform Iris performance benchmarks (using Airspeed Velocity).

    All run types require a single Nox positional argument (e.g.
    ``nox --session="foo" -- my_pos_arg``) - detailed in the parameters
    section - and can optionally accept a series of further arguments that will
    be added to session's ASV command.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.
    run_type: {"overnight", "branch", "custom"}
        * ``overnight``: benchmarks all commits between the input **first
          commit** to ``HEAD``, comparing each to its parent for performance
          shifts. If a commit causes shifts, the output is saved to a file:
          ``performance-shifts/<commit-sha>``. Designed for checking the
          previous 24 hours' commits, typically in a scheduled script.
        * ``branch``: Performs the same operations as ``overnight``, but always
          on two commits only - ``HEAD``, and ``HEAD``'s merge-base with the
          input **base branch**. Output from this run is never saved to a file.
          Designed for testing if the active branch's changes cause performance
          shifts - anticipating what would be caught by ``overnight`` once
          merged.
        * ``custom``: run ASV with the input **ASV command type**, without any
          preset arguments - must all be supplied by the user. So just like
          running ASV manually, with the convenience of re-using the session's
          scripted setup steps.

    Examples
    --------
    * ``nox --session="benchmarks(overnight)" -- a1b23d4``
    * ``nox --session="benchmarks(branch)" -- upstream/main``
    * ``nox --session="benchmarks(branch)" -- upstream/mesh-data-model``
    * ``nox --session="benchmarks(branch)" -- upstream/main --bench=regridding``
    * ``nox --session="benchmarks(custom)" -- continuous a1b23d4 HEAD --quick``

    """
    # The threshold beyond which shifts are 'notable'. See `asv compare`` docs
    #  for more.
    COMPARE_FACTOR = 1.2

    session.install("asv", "nox")
    session.cd("benchmarks")
    # Skip over setup questions for a new machine.
    session.run("asv", "machine", "--yes")

    # All run types require one Nox posarg.
    run_type_arg = {
        "overnight": "first commit",
        "branch": "base branch",
        "custom": "ASV command type",
    }
    if run_type not in run_type_arg.keys():
        message = f"Unsupported run-type: {run_type}"
        raise NotImplementedError(message)
    first_arg = session.posargs[0]
    if not first_arg:
        message = (
            f"Missing mandatory first Nox session posarg: "
            f"{run_type_arg[run_type]}"
        )
        raise ValueError(message)
    # Optional extra arguments to be passed down to ASV.
    asv_args = session.posargs[1:]

    def asv_compare(*commits):
        """Run through a list of commits comparing each one to the next."""
        commits = [commit[:8] for commit in commits]
        shifts_dir = Path(".asv") / "performance-shifts"
        for i in range(len(commits) - 1):
            before = commits[i]
            after = commits[i + 1]
            asv_command_ = f"asv compare {before} {after} --factor={COMPARE_FACTOR} --split"
            session.run(*asv_command_.split(" "))

            if run_type == "overnight":
                # Record performance shifts.
                # Run the command again but limited to only showing performance
                #  shifts.
                shifts = session.run(
                    *asv_command_.split(" "), "--only-changed", silent=True
                )
                if shifts:
                    # Write the shifts report to a file.
                    shifts_dir.mkdir(exist_ok=True, parents=True)
                    shifts_path = shifts_dir / after
                    with shifts_path.open("w") as shifts_file:
                        shifts_file.write(shifts)

    # Common ASV arguments used for both `overnight` and `bench` run_types.
    asv_harness = (
        "asv run {posargs} --attribute rounds=4 --interleave-rounds --strict"
    )

    if run_type == "overnight":
        first_commit = first_arg
        commit_range = f"{first_commit}^^.."
        asv_command = asv_harness.format(posargs=commit_range)
        session.run(*asv_command.split(" "), *asv_args)

        # git rev-list --first-parent is the command ASV uses.
        git_command = f"git rev-list --first-parent {commit_range}"
        commit_string = session.run(
            *git_command.split(" "), silent=True, external=True
        )
        commit_list = commit_string.rstrip().split("\n")
        asv_compare(*reversed(commit_list))

    elif run_type == "branch":
        base_branch = first_arg
        git_command = f"git merge-base HEAD {base_branch}"
        merge_base = session.run(
            *git_command.split(" "), silent=True, external=True
        )[:8]

        with NamedTemporaryFile("w") as hashfile:
            hashfile.writelines([merge_base, "\n", "HEAD"])
            hashfile.flush()
            commit_range = f"HASHFILE:{hashfile.name}"
            asv_command = asv_harness.format(posargs=commit_range)
            session.run(*asv_command.split(" "), *asv_args)

        asv_compare(merge_base, "HEAD")

    else:
        asv_command_type = first_arg
        session.run("asv", asv_command_type, *asv_args)

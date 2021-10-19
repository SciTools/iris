# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
ASV plug-in providing an alternative ``Environment`` subclass, which uses Nox
for environment management.

"""
from importlib.util import find_spec
from pathlib import Path
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

from asv.config import Config
from asv.console import log
from asv.environment import get_env_name
from asv.plugins.conda import Conda, _find_conda
from asv.repo import get_repo, Repo
from asv import util as asv_util


class NoxConda(Conda):
    """
    Manage a Conda environment using Nox, updating environment at each commit.

    Defers environment management to the project's noxfile, which must be able
    to create/update the benchmarking environment using ``nox --install-only``,
    with the ``--session`` specified in ``asv.conf.json.nox_session_name``.

    Notes
    -----
    If not all benchmarked commits support this use of Nox: the plugin will
    need to be modified to prep the environment in other ways.

    """

    tool_name = "nox-conda"

    @classmethod
    def matches(cls, python: str) -> bool:
        """Used by ASV to work out if this type of environment can be used."""
        result = find_spec("nox") is not None
        if result:
            result = super().matches(python)

        if result:
            message = (
                f"NOTE: ASV env match check incomplete. Not possible to know "
                f"if selected Nox session (asv.conf.json.nox_session_name) is "
                f"compatible with ``--python={python}`` until project is "
                f"checked out."
            )
            log.warning(message)

        return result

    def __init__(self, conf: Config, python: str, requirements: dict) -> None:
        """
        Parameters
        ----------
        conf: Config instance

        python : str
            Version of Python. Must be of the form "MAJOR.MINOR".

        requirements : dict
            Dictionary mapping a PyPI package name to a version
            identifier string.

        """
        from nox.sessions import _normalize_path

        # Need to checkout the project BEFORE the benchmark run - to access a noxfile.
        self.project_temp_checkout = TemporaryDirectory(
            prefix="nox_asv_checkout_"
        )
        repo = get_repo(conf)
        repo.checkout(self.project_temp_checkout.name, conf.nox_setup_commit)
        self.noxfile_rel_path = conf.noxfile_rel_path
        self.setup_noxfile = (
            Path(self.project_temp_checkout.name) / self.noxfile_rel_path
        )
        self.nox_session_name = conf.nox_session_name

        # Some duplication of parent code - need these attributes BEFORE
        #  running inherited code.
        self._python = python
        self._requirements = requirements
        self._env_dir = conf.env_dir

        # Prepare the actual environment path, to override self._path.
        nox_envdir = str(Path(self._env_dir).absolute() / self.hashname)
        nox_friendly_name = self._get_nox_session_name(python)
        self._nox_path = Path(_normalize_path(nox_envdir, nox_friendly_name))

        # For storing any extra conda requirements from asv.conf.json.
        self._extra_reqs_path = self._nox_path / "asv-extra-reqs.yaml"

        super().__init__(conf, python, requirements)

    @property
    def _path(self) -> str:
        """
        Using a property to override getting and setting in parent classes -
        unable to modify parent classes as this is a plugin.

        """
        return str(self._nox_path)

    @_path.setter
    def _path(self, value) -> None:
        """Enforce overriding of this variable by disabling modification."""
        pass

    @property
    def name(self) -> str:
        """Overridden to prevent inclusion of user input requirements."""
        return get_env_name(self.tool_name, self._python, {})

    def _get_nox_session_name(self, python: str) -> str:
        nox_cmd_substring = (
            f"--noxfile={self.setup_noxfile} "
            f"--session={self.nox_session_name} "
            f"--python={python}"
        )

        list_output = asv_util.check_output(
            ["nox", "--list", *nox_cmd_substring.split(" ")],
            display_error=False,
            dots=False,
        )
        list_output = list_output.split("\n")
        list_matches = list(filter(lambda s: s.startswith("*"), list_output))
        matches_count = len(list_matches)

        if matches_count == 0:
            message = f"No Nox sessions found for: {nox_cmd_substring} ."
            log.error(message)
            raise RuntimeError(message)
        elif matches_count > 1:
            message = (
                f"Ambiguous - >1 Nox session found for: {nox_cmd_substring} ."
            )
            log.error(message)
            raise RuntimeError(message)
        else:
            line = list_matches[0]
            session_name = line.split(" ")[1]
            assert isinstance(session_name, str)
            return session_name

    def _nox_prep_env(self, setup: bool = False) -> None:
        message = f"Running Nox environment update for: {self.name}"
        log.info(message)

        build_root_path = Path(self._build_root)
        env_path = Path(self._path)

        def copy_asv_files(src_parent: Path, dst_parent: Path) -> None:
            """For copying between self._path and a temporary cache."""
            asv_files = list(src_parent.glob("asv*"))
            # build_root_path.name usually == "project" .
            asv_files += [src_parent / build_root_path.name]
            for src_path in asv_files:
                dst_path = dst_parent / src_path.name
                if not dst_path.exists():
                    # Only cache-ing in case Nox has rebuilt the env @
                    #  self._path. If the dst_path already exists: rebuilding
                    #  hasn't happened. Also a non-issue when copying in the
                    #  reverse direction because the cache dir is temporary.
                    if src_path.is_dir():
                        func = copytree
                    else:
                        func = copy2
                    func(src_path, dst_path)

        with TemporaryDirectory(prefix="nox_asv_cache_") as asv_cache:
            asv_cache_path = Path(asv_cache)
            if setup:
                noxfile = self.setup_noxfile
            else:
                # Cache all of ASV's files as Nox may remove and re-build the environment.
                copy_asv_files(env_path, asv_cache_path)
                # Get location of noxfile in cache.
                noxfile_original = (
                    build_root_path / self._repo_subdir / self.noxfile_rel_path
                )
                noxfile_subpath = noxfile_original.relative_to(
                    build_root_path.parent
                )
                noxfile = asv_cache_path / noxfile_subpath

            nox_cmd = [
                "nox",
                f"--noxfile={noxfile}",
                # Place the env in the ASV env directory, instead of the default.
                f"--envdir={env_path.parent}",
                f"--session={self.nox_session_name}",
                f"--python={self._python}",
                "--install-only",
                "--no-error-on-external-run",
                "--verbose",
            ]

            _ = asv_util.check_output(nox_cmd)
            if not env_path.is_dir():
                message = f"Expected Nox environment not found: {env_path}"
                log.error(message)
                raise RuntimeError(message)

            if not setup:
                # Restore ASV's files from the cache (if necessary).
                copy_asv_files(asv_cache_path, env_path)

    def _setup(self) -> None:
        """Used for initial environment creation - mimics parent method where possible."""
        try:
            self.conda = _find_conda()
        except IOError as e:
            raise asv_util.UserError(str(e))
        if find_spec("nox") is None:
            raise asv_util.UserError("Module not found: nox")

        message = f"Creating Nox-Conda environment for {self.name} ."
        log.info(message)

        try:
            self._nox_prep_env(setup=True)
        finally:
            # No longer need the setup checkout now that the environment has been built.
            self.project_temp_checkout.cleanup()

        conda_args, pip_args = self._get_requirements(self.conda)
        if conda_args or pip_args:
            message = (
                "Ignoring user input package requirements. Benchmark "
                "environment management is exclusively performed by Nox."
            )
            log.warning(message)

    def checkout_project(self, repo: Repo, commit_hash: str) -> None:
        """Check out the working tree of the project at given commit hash."""
        super().checkout_project(repo, commit_hash)
        self._nox_prep_env()
        log.info(
            f"Environment {self.name} updated to spec at {commit_hash[:8]}"
        )

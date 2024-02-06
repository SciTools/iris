# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""ASV plug-in providing an alternative :class:`asv.plugins.conda.Conda` subclass.

Manages the Conda environment via custom user scripts.

"""

from os import environ
from os.path import getmtime
from pathlib import Path
from shutil import copy2, copytree, rmtree
from tempfile import TemporaryDirectory

from asv import util as asv_util
from asv.config import Config
from asv.console import log
from asv.plugins.conda import Conda
from asv.repo import Repo


class CondaDelegated(Conda):
    """Manage a Conda environment using custom user scripts, run at each commit.

    Ignores user input variations - ``matrix`` / ``pythons`` /
    ``conda_environment_file``, since environment is being managed outside ASV.

    Original environment creation behaviour is inherited, but upon checking out
    a commit the custom script(s) are run and the original environment is
    replaced with a symlink to the custom environment. This arrangement is then
    re-used in subsequent runs.

    """

    tool_name = "conda-delegated"

    def __init__(
        self,
        conf: Config,
        python: str,
        requirements: dict,
        tagged_env_vars: dict,
    ) -> None:
        """__init__.

        Parameters
        ----------
        conf : Config instance

        python : str
            Version of Python.  Must be of the form "MAJOR.MINOR".

        requirements : dict
            Dictionary mapping a PyPI package name to a version
            identifier string.

        tagged_env_vars : dict
            Environment variables, tagged for build vs. non-build

        """
        ignored = ["`python`"]
        if requirements:
            ignored.append("`requirements`")
        if tagged_env_vars:
            ignored.append("`tagged_env_vars`")
        if conf.conda_channels:
            ignored.append("conda_channels")
        if conf.conda_environment_file:
            ignored.append("`conda_environment_file`")
        message = (
            f"Ignoring ASV setting(s): {', '.join(ignored)}. Benchmark "
            "environment management is delegated to third party script(s)."
        )
        log.warning(message)
        requirements = {}
        tagged_env_vars = {}
        # All that is required to create ASV's bare-bones environment.
        conf.conda_channels = ["defaults"]
        conf.conda_environment_file = None

        super().__init__(conf, python, requirements, tagged_env_vars)
        self._update_info()

        self._env_commands = self._interpolate_commands(conf.delegated_env_commands)
        # Again using _interpolate_commands to get env parent path - allows use
        #  of the same ASV env variables.
        env_parent_interpolated = self._interpolate_commands(conf.delegated_env_parent)
        # Returns list of tuples, we just want the first.
        env_parent_first = env_parent_interpolated[0]
        # The 'command' is the first item in the returned tuple.
        env_parent_string = " ".join(env_parent_first[0])
        self._delegated_env_parent = Path(env_parent_string).resolve()

    @property
    def name(self):
        """Get a name to uniquely identify this environment."""
        return asv_util.sanitize_filename(self.tool_name)

    def _update_info(self) -> None:
        """Make sure class properties reflect the actual environment being used."""
        # Follow symlink if it has been created.
        actual_path = Path(self._path).resolve()
        self._path = str(actual_path)

        # Get custom environment's Python version if it exists yet.
        try:
            get_version = (
                "from sys import version_info; "
                "print(f'{version_info.major}.{version_info.minor}')"
            )
            actual_python = self.run(["-c", get_version])
            self._python = actual_python
        except OSError:
            pass

    def _prep_env(self) -> None:
        """Run the custom environment script(s) and switch to using that environment."""
        message = f"Running delegated environment management for: {self.name}"
        log.info(message)
        env_path = Path(self._path)

        def copy_asv_files(src_parent: Path, dst_parent: Path) -> None:
            """For copying between self._path and a temporary cache."""
            asv_files = list(src_parent.glob("asv*"))
            # build_root_path.name usually == "project" .
            asv_files += [src_parent / Path(self._build_root).name]
            for src_path in asv_files:
                dst_path = dst_parent / src_path.name
                if not dst_path.exists():
                    # Only caching in case the environment has been rebuilt.
                    #  If the dst_path already exists: rebuilding hasn't
                    #  happened. Also a non-issue when copying in the reverse
                    #  direction because the cache dir is temporary.
                    if src_path.is_dir():
                        func = copytree
                    else:
                        func = copy2
                    func(src_path, dst_path)

        with TemporaryDirectory(prefix="delegated_asv_cache_") as asv_cache:
            asv_cache_path = Path(asv_cache)
            # Cache all of ASV's files as delegated command may remove and
            #  re-build the environment.
            copy_asv_files(env_path.resolve(), asv_cache_path)

            # Adapt the build_dir to the cache location.
            build_root_path = Path(self._build_root)
            build_dir_original = build_root_path / self._repo_subdir
            build_dir_subpath = build_dir_original.relative_to(build_root_path.parent)
            build_dir = asv_cache_path / build_dir_subpath

            # Run the script(s) for delegated environment creation/updating.
            # (An adaptation of self._interpolate_and_run_commands).
            for command, env, return_codes, cwd in self._env_commands:
                local_envs = dict(environ)
                local_envs.update(env)
                if cwd is None:
                    cwd = str(build_dir)
                _ = asv_util.check_output(
                    command,
                    timeout=self._install_timeout,
                    cwd=cwd,
                    env=local_envs,
                    valid_return_codes=return_codes,
                )

            # Replace the env that ASV created with a symlink to the env
            #  created/updated by the custom script.
            delegated_env_path = sorted(
                self._delegated_env_parent.glob("*"),
                key=getmtime,
                reverse=True,
            )[0]
            if env_path.resolve() != delegated_env_path:
                try:
                    env_path.unlink(missing_ok=True)
                except IsADirectoryError:
                    rmtree(env_path)
                env_path.symlink_to(delegated_env_path, target_is_directory=True)

            # Check that environment exists.
            try:
                env_path.resolve(strict=True)
            except FileNotFoundError:
                message = f"Path does not resolve to environment: {env_path}"
                log.error(message)
                raise RuntimeError(message)

            # Restore ASV's files from the cache (if necessary).
            copy_asv_files(asv_cache_path, env_path.resolve())

            # Record new environment information in properties.
            self._update_info()

    def checkout_project(self, repo: Repo, commit_hash: str) -> None:
        """Check out the working tree of the project at given commit hash."""
        super().checkout_project(repo, commit_hash)
        self._prep_env()
        log.info(f"Environment {self.name} updated to spec at {commit_hash[:8]}")

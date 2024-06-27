# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""ASV plug-in providing an alternative :class:`asv.environments.Environment` subclass.

Preps an environment via custom user scripts, then uses that as the
benchmarking environment.

"""

from contextlib import contextmanager, suppress
from os import environ
from os.path import getmtime
from pathlib import Path
import sys

from asv import util as asv_util
from asv.console import log
from asv.environment import Environment, EnvironmentUnavailable
from asv.repo import Repo
from asv.util import ProcessError


class EnvPrepCommands:
    """A container for the environment preparation commands for a given commit.

    Designed to read a value from the `delegated_env_commands` in the ASV
    config, and validate that the command(s) are structured correctly.
    """

    ENV_PARENT_VAR = "ENV_PARENT"
    env_parent: Path
    commands: list[str]

    def __init__(self, environment: Environment, raw_commands: tuple[str]):
        env_var = self.ENV_PARENT_VAR
        raw_commands_list = list(raw_commands)

        (first_command,) = environment._interpolate_commands(raw_commands_list[0])
        env: dict
        command, env, return_codes, cwd = first_command

        valid = command == []
        valid = valid and return_codes == {0}
        valid = valid and cwd is None
        valid = valid and list(env.keys()) == [env_var]
        if not valid:
            message = (
                "First command MUST ONLY "
                f"define the {env_var} env var, with no command e.g: "
                f"`{env_var}=foo/`. Got: \n {raw_commands_list[0]}"
            )
            raise ValueError(message)

        self.env_parent = Path(env[env_var]).resolve()
        self.commands = raw_commands_list[1:]


class CommitFinder(dict[str, EnvPrepCommands]):
    """A specialised dict for finding the appropriate env prep script for a commit."""

    def __call__(self, repo: Repo, commit_hash: str):
        """Return the latest env prep script that is earlier than the given commit."""

        def validate_commit(commit: str, is_lookup: bool) -> None:
            try:
                _ = repo.get_date(commit)
            except ProcessError:
                if is_lookup:
                    message_start = "Lookup commit"
                else:
                    message_start = "Requested commit"
                repo_path = getattr(repo, "_path", "unknown")
                message = f"{message_start}: {commit} not found in repo: {repo_path}"
                raise KeyError(message)

        for lookup in self.keys():
            validate_commit(lookup, is_lookup=True)
        validate_commit(commit_hash, is_lookup=False)

        def parent_distance(parent_hash: str) -> int:
            range_spec = repo.get_range_spec(parent_hash, commit_hash)
            parents = repo.get_hashes_from_range(range_spec)

            if parent_hash[:8] == commit_hash[:8]:
                distance = 0
            elif len(parents) == 0:
                distance = -1
            else:
                distance = len(parents)
            return distance

        parentage = {commit: parent_distance(commit) for commit in self.keys()}
        parentage = {k: v for k, v in parentage.items() if v >= 0}
        if len(parentage) == 0:
            message = f"No env prep script available for commit: {commit_hash} ."
            raise KeyError(message)
        else:
            parentage = dict(sorted(parentage.items(), key=lambda item: item[1]))
            commit = next(iter(parentage))
            content = self[commit]
            return content


class Delegated(Environment):
    """Manage a benchmark environment using custom user scripts, run at each commit.

    Ignores user input variations - ``matrix`` / ``pythons`` /
    ``exclude``, since environment is being managed outside ASV.

    A vanilla :class:`asv.environment.Environment` is created for containing
    the expected ASV configuration files and checked-out project. The actual
    'functional' environment is created/updated using the command(s) specified
    in the config ``delegated_env_commands``, then the location is recorded via
    a symlink within the ASV environment. The symlink is used as the
    environment path used for any executable calls (e.g.
    ``python my_script.py``).

    """

    tool_name = "delegated"
    """Required by ASV as a unique identifier of the environment type."""

    DELEGATED_LINK_NAME = "delegated_env"
    """The name of the symlink to the delegated environment."""

    COMMIT_ENVS_VAR = "ASV_COMMIT_ENVS"
    """Env var that instructs a dedicated environment be created per commit."""

    def __init__(self, conf, python, requirements, tagged_env_vars):
        """Get a 'delegated' environment based on the given ASV config object.

        Parameters
        ----------
        conf : dict
            ASV configuration object.

        python : str
            Ignored - environment management is delegated. The value is always
            ``DELEGATED``.

        requirements : dict (str -> str)
            Ignored - environment management is delegated. The value is always
            an empty dict.

        tagged_env_vars : dict (tag, key) -> value
            Ignored - environment management is delegated. The value is always
            an empty dict.

        Raises
        ------
        EnvironmentUnavailable
            The original environment or delegated environment cannot be created.

        """
        ignored = []
        if python:
            ignored.append(f"{python=}")
        if requirements:
            ignored.append(f"{requirements=}")
        if tagged_env_vars:
            ignored.append(f"{tagged_env_vars=}")
        message = (
            f"Ignoring ASV setting(s): {', '.join(ignored)}. Benchmark "
            "environment management is delegated to third party script(s)."
        )
        log.warning(message)
        self._python = "DELEGATED"
        self._requirements = {}
        self._tagged_env_vars = {}
        super().__init__(
            conf,
            self._python,
            self._requirements,
            self._tagged_env_vars,
        )

        self._path_undelegated = Path(self._path)
        """Preserves the 'true' path of the environment so that self._path can
        be safely modified and restored."""

        env_commands = getattr(conf, "delegated_env_commands")
        try:
            env_prep_commands = {
                commit: EnvPrepCommands(self, commands)
                for commit, commands in env_commands.items()
            }
        except ValueError as err:
            message = f"Problem handling `delegated_env_commands`:\n{err}"
            log.error(message)
            raise EnvironmentUnavailable(message)
        self._env_prep_lookup = CommitFinder(**env_prep_commands)
        """An object that can be called downstream to get the appropriate
        env prep script for a given repo and commit."""

    @property
    def _path_delegated(self) -> Path:
        """The path of the symlink to the delegated environment."""
        return self._path_undelegated / self.DELEGATED_LINK_NAME

    @property
    def _delegated_found(self) -> bool:
        """Whether self._path_delegated successfully resolves to a directory."""
        resolved = None
        with suppress(FileNotFoundError):
            resolved = self._path_delegated.resolve(strict=True)
        result = resolved is not None and resolved.is_dir()
        return result

    def _symlink_to_delegated(self, delegated_env_path: Path) -> None:
        """Create the symlink to the delegated environment."""
        self._path_delegated.unlink(missing_ok=True)
        self._path_delegated.parent.mkdir(parents=True, exist_ok=True)
        self._path_delegated.symlink_to(delegated_env_path, target_is_directory=True)
        assert self._delegated_found

    def _setup(self):
        """Temporarily try to set the user's active env as the delegated env.

        Environment prep will be run anyway once ASV starts checking out
        commits, but this step tries to provide a usable environment (with
        python, etc.) at the moment that ASV expects it.

        """
        current_env = Path(sys.executable).parents[1]
        message = (
            "Temporarily using user's active environment as benchmarking "
            f"environment: {current_env} . "
        )
        try:
            self._symlink_to_delegated(current_env)
            _ = self.find_executable("python")
        except Exception:
            message = (
                f"Delegated environment {self.name} not yet set up (unable to "
                "determine current environment)."
            )
            self._path_delegated.unlink(missing_ok=True)

        message += "Correct environment will be set up at the first commit checkout."
        log.warning(message)

    def _prep_env(self, repo: Repo, commit_hash: str) -> None:
        """Prepare the delegated environment for the given commit hash."""
        message = (
            f"Running delegated environment management for: {self.name} "
            f"at commit: {commit_hash[:8]}"
        )
        log.info(message)

        env_prep: EnvPrepCommands
        try:
            env_prep = self._env_prep_lookup(repo, commit_hash)
        except KeyError as err:
            message = f"Problem finding env prep commands: {err}"
            log.error(message)
            raise EnvironmentUnavailable(message)

        new_env_per_commit = self.COMMIT_ENVS_VAR in environ
        if new_env_per_commit:
            env_parent = env_prep.env_parent / commit_hash[:8]
        else:
            env_parent = env_prep.env_parent

        # See :meth:`Environment._interpolate_commands`.
        #  All ASV-namespaced env vars are available in the below format when
        #  interpolating commands:
        #   ASV_FOO_BAR = {foo_bar}
        # We want the env parent path to be one of those available.
        global_key = f"ASV_{EnvPrepCommands.ENV_PARENT_VAR}"
        self._global_env_vars[global_key] = str(env_parent)

        # The project checkout.
        build_dir = Path(self._build_root) / self._repo_subdir

        # Run the script(s) for delegated environment creation/updating.
        # (An adaptation of :meth:`Environment._interpolate_and_run_commands`).
        for command, env, return_codes, cwd in self._interpolate_commands(
            env_prep.commands
        ):
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

        # Find the environment created/updated by running env_prep.commands.
        #  The most recently updated directory in env_parent.
        delegated_env_path = sorted(
            env_parent.glob("*"),
            key=getmtime,
            reverse=True,
        )[0]
        # Record the environment's path via a symlink within this environment.
        self._symlink_to_delegated(delegated_env_path)

        message = f"Environment {self.name} updated to spec at {commit_hash[:8]}"
        log.info(message)

    def checkout_project(self, repo: Repo, commit_hash: str) -> None:
        """Check out the working tree of the project at given commit hash."""
        super().checkout_project(repo, commit_hash)
        self._prep_env(repo, commit_hash)

    @contextmanager
    def _delegate_path(self):
        """Context manager to use the delegated env path as this env's path."""
        if not self._delegated_found:
            message = f"Delegated environment not found at: {self._path_delegated}"
            log.error(message)
            raise EnvironmentUnavailable(message)

        try:
            self._path = str(self._path_delegated)
            yield
        finally:
            self._path = str(self._path_undelegated)

    def find_executable(self, executable):
        """Find an executable (e.g. python, pip) in the DELEGATED environment.

        Raises
        ------
        OSError
            If the executable is not found in the environment.
        """
        if not self._delegated_found:
            # Required during environment setup. OSError expected if executable
            #  not found.
            raise OSError

        with self._delegate_path():
            return super().find_executable(executable)

    def run_executable(self, executable, args, **kwargs):
        """Run a given executable (e.g. python, pip) in the DELEGATED environment."""
        with self._delegate_path():
            return super().run_executable(executable, args, **kwargs)

    def run(self, args, **kwargs):
        # This is not a specialisation - just implementing the abstract method.
        log.debug(f"Running '{' '.join(args)}' in {self.name}")
        return self.run_executable("python", args, **kwargs)

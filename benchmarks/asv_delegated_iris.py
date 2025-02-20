"""Proper class and docstring pending."""

from os import environ
from os.path import getmtime
from pathlib import Path

from asv import util as asv_util

from asv_delegated import Delegated, EnvPrepCommands


class Refactor(Delegated):
    """Proper class and docstring pending."""

    tool_name = "delegated-iris"

    def _prep_env_script(self, env_parent_dir: Path, **kwargs) -> Path:
        # See :meth:`Environment._interpolate_commands`.
        #  All ASV-namespaced env vars are available in the below format when
        #  interpolating commands:
        #   ASV_FOO_BAR = {foo_bar}
        # We want the env parent path to be one of those available.
        global_key = f"ASV_{EnvPrepCommands.ENV_PARENT_VAR}"
        self._global_env_vars[global_key] = str(env_parent_dir)

        # The project checkout.
        build_dir = Path(self._build_root) / self._repo_subdir

        commands = kwargs["commands"]
        # Run the script(s) for delegated environment creation/updating.
        # (An adaptation of :meth:`Environment._interpolate_and_run_commands`).
        for command, env, return_codes, cwd in self._interpolate_commands(commands):
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
            env_parent_dir.glob("*"),
            key=getmtime,
            reverse=True,
        )[0]
        return delegated_env_path

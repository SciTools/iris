# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
ASV plug-in providing an alternative ``Environment`` subclass, which uses Nox
for environment management.

"""
from asv.console import log
from asv.plugins.conda import Conda, _find_conda


class CondaLock(Conda):
    """
    Create the environment based on a **version-controlled** lockfile.

    Creating the environment instance is deferred until ``build_project`` time,
    when the commit hash etc is known and we can access the lock file.
    The environment is then overwritten by the specification provided at the
    ``config.conda_lockfile`` path.  ``conda.conda_lockfile`` must point to
    an @EXPLICIT conda manifest, e.g. the output of either the ``conda-lock`` tool,
    or ``conda list --explicit``.
    """

    tool_name = "conda-lock"

    def __init__(self, conf, python, requirements):
        self._lockfile_path = conf.conda_lockfile
        super().__init__(conf, python, requirements)

    def _uninstall_project(self):
        if self._get_installed_commit_hash():
            # we can only run the uninstall command if an environment has already
            # been made before, otherwise there is no python to use to uninstall
            super()._uninstall_project()
            # TODO: we probably want to conda uninstall all the packages too
            #       something like:
            #       conda list --no-pip | sed /^#/d | cut -f 1 -d " " | xargs conda uninstall

    def _setup(self):
        # create the shell of a conda environment, that includes no packages
        log.info("Creating conda environment for {0}".format(self.name))
        self.run_executable(
            _find_conda(), ["create", "-y", "-p", self._path, "--force"]
        )

    def _build_project(self, repo, commit_hash, build_dir):
        # at "build" time, we build the environment from the provided lockfile
        self.run_executable(
            _find_conda(),
            [
                "install",
                "-y",
                "-p",
                self._path,
                "--file",
                f"{build_dir}/{self._lockfile_path}",
            ],
        )
        log.info(
            f"Environment {self.name} updated to spec at {commit_hash[:8]}"
        )
        log.debug(
            self.run_executable(_find_conda(), ["list", "-p", self._path])
        )
        return super()._build_project(repo, commit_hash, build_dir)

"""Iris setup."""

import os
import sys

from setuptools import Command, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


class BaseCommand(Command):
    """A minimal no-op setuptools command."""

    description: str = "A no-op command."
    user_options: list = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


def custom_command(cmd, help=""):
    """Create custom command with factory function.

    Custom command will add additional behaviour to build the CF
    standard names module.

    """

    class CustomCommand(cmd):
        description = help or cmd.description

        def _build_std_names(self, directory):
            # Call out to tools/generate_std_names.py to build std_names module.

            script_path = os.path.join("tools", "generate_std_names.py")
            xml_path = os.path.join("etc", "cf-standard-name-table.xml")
            module_path = os.path.join(directory, "iris", "std_names.py")
            args = [sys.executable, script_path, xml_path, module_path]
            self.spawn(args)

        def finalize_options(self):
            # Execute the parent "cmd" class method.
            cmd.finalize_options(self)

            if not hasattr(self, "editable_mode") or self.editable_mode is None:
                # Default to editable i.e., applicable to "std_names" and
                # and "develop" commands.
                self.editable_mode = True

        def run(self):
            # Execute the parent "cmd" class method.
            cmd.run(self)

            # Determine the target root directory
            if self.editable_mode:
                # Pick the source dir instead (currently in the sub-dir "lib").
                target = "lib"
                msg = "in-place"
            else:
                # Not editable - must be building.
                target = self.build_lib
                msg = "as-build"

            print(f"\n[Running {msg}]")

            # Build the CF standard names.
            self._build_std_names(target)

    return CustomCommand


custom_commands = {
    "develop": custom_command(develop),
    "build_py": custom_command(build_py),
    "std_names": custom_command(BaseCommand, help="generate CF standard names"),
}


setup(
    cmdclass=custom_commands,
)

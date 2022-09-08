import os
import sys

from setuptools import Command, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


class BaseCommand(Command):
    """A valid no-op command for setuptools & distutils."""

    description: str = "A no-op command."
    editable_mode: bool = True
    user_options: list = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


class DevelopCommand(develop):
    editable_mode: bool = True


def build_std_names(cmd, directory):
    # Call out to tools/generate_std_names.py to build std_names module.

    script_path = os.path.join("tools", "generate_std_names.py")
    xml_path = os.path.join("etc", "cf-standard-name-table.xml")
    module_path = os.path.join(directory, "iris", "std_names.py")
    args = (sys.executable, script_path, xml_path, module_path)
    cmd.spawn(args)


def custom_cmd(klass, functions, help=""):
    """
    Allows command specialisation to include calls to the given functions.

    """

    class CustomCommand(klass):
        description = help or klass.description

        def run(self):
            # Run the original command to perform associated behaviour.
            klass.run(self)

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

            # Run the custom command functions.
            for func in functions:
                func(self, target)

    return CustomCommand


custom_commands = {
    "develop": custom_cmd(DevelopCommand, [build_std_names]),
    "build_py": custom_cmd(build_py, [build_std_names]),
    "std_names": custom_cmd(
        BaseCommand,
        [build_std_names],
        help="generate CF standard names module",
    ),
}


setup(
    cmdclass=custom_commands,
)

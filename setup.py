from contextlib import contextmanager
import os
from shutil import copyfile
import sys

from setuptools import Command, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop as develop_cmd


@contextmanager
def temporary_path(directory):
    """
    Context manager that adds and subsequently removes the given directory
    to sys.path

    """
    sys.path.insert(0, directory)
    try:
        yield
    finally:
        del sys.path[0]


# Add full path so Python doesn't load any __init__.py in the intervening
# directories, thereby saving setup.py from additional dependencies.
with temporary_path("lib/iris/tests/runner"):
    from _runner import TestRunner  # noqa:


class SetupTestRunner(TestRunner, Command):
    pass


class BaseCommand(Command):
    """A valid no-op command for setuptools & distutils."""

    description = "A no-op command."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


class CleanSource(BaseCommand):
    description = "clean orphaned pyc/pyo files from the source directory"

    def run(self):
        for root_path, dir_names, file_names in os.walk("lib"):
            for file_name in file_names:
                if file_name.endswith("pyc") or file_name.endswith("pyo"):
                    compiled_path = os.path.join(root_path, file_name)
                    source_path = compiled_path[:-1]
                    if not os.path.exists(source_path):
                        print("Cleaning", compiled_path)
                        os.remove(compiled_path)


def copy_copyright(cmd, directory):
    # Copy the COPYRIGHT information into the package root
    iris_build_dir = os.path.join(directory, "iris")
    for fname in ["COPYING", "COPYING.LESSER"]:
        copyfile(fname, os.path.join(iris_build_dir, fname))


def build_std_names(cmd, directory):
    # Call out to tools/generate_std_names.py to build std_names module.

    script_path = os.path.join("tools", "generate_std_names.py")
    xml_path = os.path.join("etc", "cf-standard-name-table.xml")
    module_path = os.path.join(directory, "iris", "std_names.py")
    args = (sys.executable, script_path, xml_path, module_path)
    cmd.spawn(args)


def custom_cmd(command_to_override, functions, help_doc=""):
    """
    Allows command specialisation to include calls to the given functions.

    """

    class ExtendedCommand(command_to_override):
        description = help_doc or command_to_override.description

        def run(self):
            # Run the original command first to make sure all the target
            # directories are in place.
            command_to_override.run(self)

            # build_lib is defined if we are building the package. Otherwise
            # we want to to the work in-place.
            dest = getattr(self, "build_lib", None)
            if dest is None:
                print(" [Running in-place]")
                # Pick the source dir instead (currently in the sub-dir "lib")
                dest = "lib"

            for func in functions:
                func(self, dest)

    return ExtendedCommand


custom_commands = {
    "test": SetupTestRunner,
    "develop": custom_cmd(develop_cmd, [build_std_names]),
    "build_py": custom_cmd(build_py, [build_std_names, copy_copyright]),
    "std_names": custom_cmd(
        BaseCommand,
        [build_std_names],
        help_doc="generate CF standard name module",
    ),
    "clean_source": CleanSource,
}


setup(
    cmdclass=custom_commands,
)

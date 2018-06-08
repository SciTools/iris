from __future__ import print_function

from contextlib import contextmanager
from distutils.util import convert_path
import os
from shutil import copyfile
import sys
import textwrap

from setuptools import setup, Command
from setuptools.command.develop import develop as develop_cmd
from setuptools.command.build_py import build_py


# Returns the package and all its sub-packages
def find_package_tree(root_path, root_package):
    root_path = root_path.replace('/', os.path.sep)
    packages = [root_package]
    root_count = len(root_path.split(os.path.sep))
    for (dir_path, dir_names, file_names) in os.walk(convert_path(root_path)):
        # Prune dir_names *in-place* to prevent unwanted directory recursion
        for dir_name in list(dir_names):
            contains_init_file = os.path.isfile(os.path.join(dir_path,
                                                             dir_name,
                                                             '__init__.py'))
            if not contains_init_file:
                dir_names.remove(dir_name)
            # Exclude compiled PyKE rules, but keep associated unit tests.
            if dir_name == 'compiled_krb' and 'tests' not in dir_path:
                dir_names.remove(dir_name)
        if dir_names:
            prefix = dir_path.split(os.path.sep)[root_count:]
            packages.extend(['.'.join([root_package] + prefix + [dir_name])
                             for dir_name in dir_names])
    return packages


def file_walk_relative(top, remove=''):
    """
    Returns a generator of files from the top of the tree, removing
    the given prefix from the root/file result.

    """
    top = top.replace('/', os.path.sep)
    remove = remove.replace('/', os.path.sep)
    for root, dirs, files in os.walk(top):
        for file in files:
            yield os.path.join(root, file).replace(remove, '')


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
with temporary_path('lib/iris/tests/runner'):
    from _runner import TestRunner  # noqa:

SETUP_DIR = os.path.dirname(__file__)

def pip_requirements(name):
    fname = os.path.join(SETUP_DIR, 'requirements', '{}.txt'.format(name))
    if not os.path.exists(fname):
        raise RuntimeError('Unable to find the {} requirements file at {}'
                           ''.format(name, fname))
    reqs = []
    with open(fname, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            reqs.append(line)
    return reqs


class SetupTestRunner(TestRunner, Command):
    pass


class BaseCommand(Command):
    """A valid no-op command for setuptools & distutils."""

    description = 'A no-op command.'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


class CleanSource(BaseCommand):
    description = 'clean orphaned pyc/pyo files from the source directory'

    def run(self):
        for root_path, dir_names, file_names in os.walk('lib'):
            for file_name in file_names:
                if file_name.endswith('pyc') or file_name.endswith('pyo'):
                    compiled_path = os.path.join(root_path, file_name)
                    source_path = compiled_path[:-1]
                    if not os.path.exists(source_path):
                        print('Cleaning', compiled_path)
                        os.remove(compiled_path)


def compile_pyke_rules(cmd, directory):
    # Call out to the python executable to pre-compile the Pyke rules.
    # Significant effort was put in to trying to get these to compile
    # within this build process but there was no obvious way of finding
    # a workaround to the issue presented in
    # https://github.com/SciTools/iris/issues/2481.

    shelled_code = textwrap.dedent("""\

    import os

    # Monkey patch the load method to avoid "ModuleNotFoundError: No module
    # named 'iris.fileformats._pyke_rules.compiled_krb'". In this instance
    # we simply don't want the knowledge engine, so we turn the load method
    # into a no-op.
    from pyke.target_pkg import target_pkg
    target_pkg.load = lambda *args, **kwargs: None

    # Compile the rules by hand, without importing iris. That way we can
    # avoid the need for all of iris' dependencies being installed.
    os.chdir(os.path.join('{bld_dir}', 'iris', 'fileformats', '_pyke_rules'))
    
    # Import pyke *after* changing directory. Without this we get the compiled
    # rules in the wrong place. Identified in
    # https://github.com/SciTools/iris/pull/2891#issuecomment-341404187
    from pyke import knowledge_engine
    knowledge_engine.engine('')

    """.format(bld_dir=directory)).split('\n')
    shelled_code = '; '.join(
        [line for line in shelled_code
         if not line.strip().startswith('#') and line.strip()])
    args = [sys.executable, '-c', shelled_code]
    cmd.spawn(args)


def copy_copyright(cmd, directory):
    # Copy the COPYRIGHT information into the package root
    iris_build_dir = os.path.join(directory, 'iris')
    for fname in ['COPYING', 'COPYING.LESSER']:
        copyfile(fname, os.path.join(iris_build_dir, fname))


def build_std_names(cmd, directory):
    # Call out to tools/generate_std_names.py to build std_names module.

    script_path = os.path.join('tools', 'generate_std_names.py')
    xml_path = os.path.join('etc', 'cf-standard-name-table.xml')
    module_path = os.path.join(directory, 'iris', 'std_names.py')
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
            dest = getattr(self, 'build_lib', None)
            if dest is None:
                print(' [Running in-place]')
                # Pick the source dir instead (currently in the sub-dir "lib")
                dest = 'lib'

            for func in functions:
                func(self, dest)

    return ExtendedCommand


def extract_version():
    version = None
    fnme = os.path.join(SETUP_DIR, 'lib', 'iris', '__init__.py')
    with open(fnme) as fd:
        for line in fd:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version


custom_commands = {
    'test': SetupTestRunner,
    'develop': custom_cmd(
        develop_cmd, [build_std_names, compile_pyke_rules]),
    'build_py': custom_cmd(
        build_py,
        [build_std_names, compile_pyke_rules, copy_copyright]),
    'std_names':
        custom_cmd(BaseCommand, [build_std_names],
                   help_doc="generate CF standard name module"),
    'pyke_rules':
        custom_cmd(BaseCommand, [compile_pyke_rules],
                   help_doc="compile CF-NetCDF loader rules"),
    'clean_source': CleanSource,
    }


pypi_name = 'scitools-iris'

with open(os.path.join(SETUP_DIR, 'README.md'), 'r') as fh:
    description = ''.join(fh.readlines())

setup(
    name=pypi_name,
    version=extract_version(),
    url='http://scitools.org.uk/iris/',
    author='UK Met Office',
    author_email='scitools-iris-dev@googlegroups.com',
    description="A powerful, format-agnostic, community-driven Python "
                "library for analysing and visualising Earth science data",
    long_description=description,
    long_description_content_type='text/markdown',
    packages=find_package_tree('lib/iris', 'iris'),
    package_dir={'': 'lib'},
    include_package_data=True,
    cmdclass=custom_commands,

    zip_safe=False,

    setup_requires=pip_requirements('setup'),
    install_requires=pip_requirements('setup') + pip_requirements('core'),
    tests_require=['{}[test]'.format(pypi_name)],
    extras_require = {
                      'test': pip_requirements('test'),
                      'all': pip_requirements('all'),
                      'extensions': pip_requirements('extensions'),
                      },
)

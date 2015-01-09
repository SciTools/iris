from __future__ import print_function

import contextlib
from distutils.command import build_ext, build_py
from distutils.core import setup, Command
from distutils.sysconfig import get_config_var
from distutils.util import convert_path
import fnmatch
import multiprocessing
import os
import sys

import numpy as np
import setuptools

# Add full path so Python doesn't load any __init__.py in the intervening
# directories, thereby saving setup.py from additional dependencies.
sys.path.append('lib/iris/tests/runner')
from _runner import TestRunner

exclude_dirs = ['compiled_krb']

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
            if dir_name in exclude_dirs or not contains_init_file:
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


def std_name_cmd(target_dir):
    script_path = os.path.join('tools', 'generate_std_names.py')
    xml_path = os.path.join('etc', 'cf-standard-name-table.xml')
    module_path = os.path.join(target_dir, 'iris', 'std_names.py')
    cmd = (sys.executable, script_path, xml_path, module_path)
    return cmd


class SetupTestRunner(TestRunner, setuptools.Command):
    pass


class CleanSource(Command):
    """
    Removes orphaned pyc/pyo files from the sources.

    """
    description = 'clean orphaned pyc/pyo files from sources'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for root_path, dir_names, file_names in os.walk('lib'):
            for file_name in file_names:
                if file_name.endswith('pyc') or file_name.endswith('pyo'):
                    compiled_path = os.path.join(root_path, file_name)
                    source_path = compiled_path[:-1]
                    if not os.path.exists(source_path):
                        print('Cleaning', compiled_path)
                        os.remove(compiled_path)


class MakeStdNames(Command):
    """
    Generates the CF standard name module containing mappings from
    CF standard name to associated metadata.

    """
    description = "generate CF standard name module"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd = std_name_cmd('lib')
        self.spawn(cmd)


class MakePykeRules(Command):
    """
    Compile the PyKE CF-NetCDF loader rule base.

    """
    description = "compile CF-NetCDF loader rule base"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def _pyke_rule_compile():
        """Compile the PyKE rule base."""
        from pyke import knowledge_engine
        import iris.fileformats._pyke_rules
        knowledge_engine.engine(iris.fileformats._pyke_rules)

    def run(self):
        # Compile the PyKE rules.
        MakePykeRules._pyke_rule_compile()


class MissingHeaderError(Exception):
    """
    Raised when one or more files do not have the required copyright
    and licence header.

    """
    pass


class BuildPyWithExtras(build_py.build_py):
    """
    Adds the creation of the CF standard names module and compilation
    of the PyKE rules to the standard "build_py" command.

    """
    @contextlib.contextmanager
    def temporary_path(self):
        """
        Context manager that adds and subsequently removes the build
        directory to the beginning of the module search path.

        """
        sys.path.insert(0, self.build_lib)
        try:
            yield
        finally:
            del sys.path[0]

    def run(self):
        # Run the main build command first to make sure all the target
        # directories are in place.
        build_py.build_py.run(self)

        # Now build the std_names module.
        cmd = std_name_cmd(self.build_lib)
        self.spawn(cmd)

        # Compile the PyKE rules.
        with self.temporary_path():
            MakePykeRules._pyke_rule_compile()


def extract_version():
    version = None
    fdir = os.path.dirname(__file__)
    fnme = os.path.join(fdir, 'lib', 'iris', '__init__.py')
    with open(fnme) as fd:
        for line in fd:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version


setup(
    name='Iris',
    version=extract_version(),
    url='http://scitools.org.uk/iris/',
    author='UK Met Office',

    packages=find_package_tree('lib/iris', 'iris'),
    package_dir={'': 'lib'},
    package_data={
        'iris': list(file_walk_relative('lib/iris/etc', remove='lib/iris/')) + \
                list(file_walk_relative('lib/iris/tests/results',
                                        remove='lib/iris/')) + \
                ['fileformats/_pyke_rules/*.k?b'] + \
                ['tests/stock*.npz']
        },
    data_files=[('iris', ['CHANGES', 'COPYING', 'COPYING.LESSER'])],
    tests_require=['nose'],
    features={
        'unpack': setuptools.Feature(
            "use of UKMO unpack library",
            standard=False,
            ext_modules=[
                setuptools.Extension(
                    'iris.fileformats.pp_packing',
                    ['src/iris/fileformats/pp_packing/pp_packing.c'],
                    libraries=['mo_unpack'],
                    include_dirs=[np.get_include()]
                )
            ]
        )
    },
    cmdclass={'test': SetupTestRunner, 'build_py': BuildPyWithExtras,
              'std_names': MakeStdNames, 'pyke_rules': MakePykeRules,
              'clean_source': CleanSource},
)

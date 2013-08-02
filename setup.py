import contextlib
from distutils.command import build_ext, build_py
from distutils.core import setup, Command
from distutils.sysconfig import get_config_var
from distutils.util import convert_path
import fnmatch
import multiprocessing
import os
import sys

import nose
import numpy as np
import setuptools


exclude_dirs = ['compiled_krb']

# Returns the package and all its sub-packages
def find_package_tree(root_path, root_package):
    packages = [root_package]
    root_count = len(root_path.split('/'))
    for (dir_path, dir_names, file_names) in os.walk(convert_path(root_path)):
        # Prune dir_names *in-place* to prevent unwanted directory recursion
        for dir_name in list(dir_names):
            contains_init_file = os.path.isfile(os.path.join(dir_path,
                                                             dir_name,
                                                             '__init__.py'))
            if dir_name in exclude_dirs or not contains_init_file:
                dir_names.remove(dir_name)
        if dir_names:
            prefix = dir_path.split('/')[root_count:]
            packages.extend(['.'.join([root_package] + prefix + [dir_name])
                                for dir_name in dir_names])
    return packages


def file_walk_relative(top, remove=''):
    """
    Returns a generator of files from the top of the tree, removing
    the given prefix from the root/file result.

    """
    for root, dirs, files in os.walk(top):
        for file in files:
            yield os.path.join(root, file).replace(remove, '')


def std_name_cmd(target_dir):
    script_path = os.path.join('tools', 'generate_std_names.py')
    xml_path = os.path.join('etc', 'cf-standard-name-table.xml')
    module_path = os.path.join(target_dir, 'iris', 'std_names.py')
    cmd = (sys.executable, script_path, xml_path, module_path)
    return cmd


class TestRunner(setuptools.Command):
    """Run the Iris tests under nose and multiprocessor for performance"""
    description = "run tests under nose and multiprocessor for performance"
    user_options = [('no-data', 'n', 'Override the paths to the data '
                                     'repositories so it appears to the '
                                     'tests that it does not exist.'),
                    ('system-tests', 's', 'Run only the limited subset of '
                                          'system tests.')
                   ]
    
    boolean_options = ['no-data', 'system-tests']
    
    def initialize_options(self):
        self.no_data = False
        self.system_tests = False
    
    def finalize_options(self):
        if self.no_data:
            print "Running tests in no-data mode..."

            # This enviroment variable will be propagated to all the processes that
            # nose.run creates allowing us to simluate the absence of test data
            os.environ["override_test_data_repository"] = "true"
        if self.system_tests:
            print "Running system tests..."

    def run(self):
        if self.distribution.tests_require:
            self.distribution.fetch_build_eggs(self.distribution.tests_require)

        script_path = sys.path[0]
        tests = []
        lib_dir = os.path.join(script_path, 'lib')
        for mod in os.listdir(lib_dir):
            path = os.path.join(lib_dir, mod)
            if mod != '.svn' and os.path.exists(os.path.join(path, 'tests')):
                tests.append('%s.tests' % mod)

        if not tests:
            raise CommandError('No tests found in %s/*/tests' % lib_dir)

        if self.system_tests:
            regexp_pat = r'--match=^[Ss]ystem'
        else:
            regexp_pat = r'--match=^([Tt]est(?![Mm]ixin)|[Ss]ystem)'

        n_processors = max(multiprocessing.cpu_count() - 1, 1)
        
        result = True
        for test in tests:
            print
            print 'Running test discovery on %s with %s processors.' % (test, n_processors)
            # run the tests at module level i.e. my_module.tests
            # - test must start with test/Test and must not contain the word Mixin.
            result &= nose.run(argv=['', test, '--processes=%s' % n_processors,
                               '--verbosity=2', regexp_pat,
                               '--process-timeout=250'])
        if result is False:
            exit(1)


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


setup(
    name='Iris',
    version='1.5.0-dev',
    url='http://scitools.github.com/iris',
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
    cmdclass={'test': TestRunner, 'build_py': BuildPyWithExtras,
              'std_names': MakeStdNames, 'pyke_rules': MakePykeRules},
)

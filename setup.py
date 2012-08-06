from distutils.command import build_ext, build_py
from distutils.core import setup, Command
from distutils.sysconfig import get_config_var
from distutils.util import convert_path
import multiprocessing
import os
import sys

import nose
import numpy as np
import setuptools


# Automated package discovery - extracted/modified from Distutils Cookbook:
#   http://wiki.python.org/moin/Distutils/Cookbook/AutoPackageDiscovery
def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
    )


exclude_dirs = ['compiled_krb']


# Returns the package and all its sub-packages
def find_package_tree(root_path, root_package):
    packages = [root_package]
    root_count = len(root_path.split('/'))
    for (dir_path, dir_names, file_names) in os.walk(convert_path(root_path)):
        # Prune dir_names *in-place* to prevent unwanted directory recursion
        for dir_name in list(dir_names):
            if not os.path.isfile(os.path.join(dir_path, dir_name, '__init__.py')):
                dir_names.remove(dir_name)
            if dir_name in exclude_dirs:
                dir_names.remove(dir_name)
        if dir_names:
            prefix = dir_path.split('/')[root_count:]
            packages.extend(['.'.join([root_package] + prefix + [dir_name]) for dir_name in dir_names])
    return packages


def file_walk_relative(top, remove=''):
    """Returns a generator of files from the top of the tree. Removing the given prefix from the root/file result."""
    for root, dirs, files in os.walk(top):
       if '.svn' in dirs:
           dirs.remove('.svn')
       for file in files:
           yield os.path.join(root, file).replace(remove, '')


class TestRunner(setuptools.Command):
    """Run the Iris tests under nose and multiprocessor for performance"""
    description = "run tests under nose and multiprocessor for performance"
    user_options = [
        ('no-data', 'n', 'Override the paths to the data repositorys so it appears to the tests that it does not exist.'),
        ('system-tests', 's', 'Run only the limited subset of system tests.')
    ]
    
    boolean_options = [
        'no-data',
        'system-tests'
    ]
    
    def initialize_options(self):
        self.no_data = False
        self.system_tests = False
    
    def finalize_options(self):
        if self.no_data:
            print "Running tests in no-data mode..."
            
            # This enviroment variable will be propagated to all the processes that
            # nose.run creates allowing us to simluate the absence of test data
            os.environ["override_data_repository"] = "true"
        if self.system_tests:
            print "Running only system tests..."

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
        
        for test in tests:
            print
            print 'Running test discovery on %s with %s processors.' % (test, n_processors)
            # run the tests at module level i.e. my_module.tests 
            # - test must start with test/Test and must not contain the word Mixin.
            nose.run(argv=['', test, '--processes=%s' % n_processors,
                               '--verbosity=2', regexp_pat,
                               '--process-timeout=250'])


def std_name_cmd(target_dir):
    script_path = os.path.join('tools', 'generate_std_names.py')
    xml_path = os.path.join('etc', 'cf-standard-name-table.xml')
    module_path = os.path.join(target_dir, 'iris', 'std_names.py')
    cmd = (sys.executable, script_path, xml_path, module_path)
    return cmd


class MakeStdNames(Command):

    description = "generate CF standard name module"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd = std_name_cmd('lib')
        self.spawn(cmd)


class BuildPyWithStdNames(build_py.build_py):
    """
    Adds the creation of the CF standard names module to the standard
    "build_py" command.

    """
    def run(self):
        # Run the main build command first to make sure all the target
        # directories are in place.
        build_py.build_py.run(self)

        # Now build the std_names module.
        cmd = std_name_cmd(self.build_lib)
        self.spawn(cmd)


class PostBuildExtRunner(build_ext.build_ext):
    """Runs after a standard "build_ext" to compile the PyKE rules."""
    def run(self):
        # Call parent
        build_ext.build_ext.run(self)

        # Add our new build dir to the start of the path to pick up the KE rules
        sys.path.insert(0, self.build_lib)
        
        # Compile the pyke rules
        from pyke import knowledge_engine
        import iris.fileformats._pyke_rules
        e = knowledge_engine.engine(iris.fileformats._pyke_rules)


setup(
    name='Iris',
    version='0.8-dev',
    url='https://github.com/SciTools/Iris',
    author='UK Met Office',

    packages=find_package_tree('lib/iris', 'iris'),
    package_dir={'': 'lib'},
    package_data={
        'iris': ['LICENCE', 'resources/logos/*.png'] + \
                list(file_walk_relative('lib/iris/etc', remove='lib/iris/')) + \
                ['fileformats/_pyke_rules/*.k?b'] + \
                list(file_walk_relative('lib/iris/tests/results', remove='lib/iris/'))
              ,
        },
    tests_require=['nose'],
    features={
        'unpack': setuptools.Feature(
            ("UKMO unpack library: \n"
            "    To append custom include paths and library dirs from the commandline,\n"
            "    python setup.py build_ext -I <custom include path> \n"
            "        -L <custom static libdir> -R <custom runtime libdir>"),
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
    cmdclass={'build_ext': PostBuildExtRunner, 'test': TestRunner,
              'build_py': BuildPyWithStdNames, 'std_names': MakeStdNames},
)

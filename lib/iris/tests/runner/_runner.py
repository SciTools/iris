# (C) British Crown Copyright 2010 - 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Provides testing capabilities for installed copies of Iris.

"""
from __future__ import print_function

# Because this file is imported by setup.py, there may be additional runtime
# imports later in the file.
import multiprocessing
import os
import sys


class TestRunner():
    """Run the Iris tests under nose and multiprocessor for performance"""

    description = 'run tests under nose and multiprocessor for performance'
    user_options = [
        ('no-data', 'n', 'Override the paths to the data repositories so it '
                         'appears to the tests that it does not exist.'),
        ('system-tests', 's', 'Run only the limited subset of system tests.'),
        ('stop', 'x', 'Stop running tests after the first error or failure.'),
        ('example-tests', 'e', 'Also run the example code tests.')
    ]
    boolean_options = ['no-data', 'system-tests', 'stop', 'example-tests']

    def initialize_options(self):
        self.no_data = False
        self.system_tests = False
        self.stop = False
        self.example_tests = False

    def finalize_options(self):
        if self.no_data:
            print('Running tests in no-data mode...')

            # This enviroment variable will be propagated to all the processes
            # that nose.run creates, allowing us to simulate the absence of
            # test data
            os.environ['override_test_data_repository'] = 'true'
        if self.system_tests:
            print('Running system tests...')
        if self.stop:
            print('Stopping tests after the first error or failure')

    def run(self):
        import nose

        if hasattr(self, 'distribution') and self.distribution.tests_require:
            self.distribution.fetch_build_eggs(self.distribution.tests_require)

        tests = ['iris.tests']
        if self.example_tests:
            import iris.config
            default_doc_path = os.path.join(sys.path[0], 'docs', 'iris')
            doc_path = iris.config.get_option('Resources', 'doc_dir',
                                              default=default_doc_path)
            example_path = os.path.join(doc_path, 'example_tests')
            if os.path.exists(example_path):
                tests.append(example_path)
            else:
                print('WARNING: Example path %s does not exist.' %
                      (example_path))

        if self.system_tests:
            regexp_pat = r'--match=^[Ss]ystem'
        else:
            regexp_pat = r'--match=^([Tt]est(?![Mm]ixin)|[Ss]ystem)'

        n_processors = max(multiprocessing.cpu_count() - 1, 1)

        args = ['', None, '--processes=%s' % n_processors,
                '--verbosity=2', regexp_pat,
                '--process-timeout=250']
        if self.stop:
            args.append('--stop')

        result = True
        for test in tests:
            args[1] = test
            print()
            print('Running test discovery on %s with %s processors.' %
                  (test, n_processors))
            # run the tests at module level i.e. my_module.tests
            # - test must start with test/Test and must not contain the
            #   word Mixin.
            result &= nose.run(argv=args)
        if result is False:
            exit(1)

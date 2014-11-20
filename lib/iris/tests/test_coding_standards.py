# (C) British Crown Copyright 2013 - 2016, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

from datetime import datetime
from fnmatch import fnmatch
from glob import glob
from itertools import chain
import os
import re
import subprocess
import unittest

import pep8

import iris


LICENSE_TEMPLATE = """
# (C) British Crown Copyright {YEARS}, Met Office
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
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.""".strip()


LICENSE_RE_PATTERN = re.escape(LICENSE_TEMPLATE).replace('\{YEARS\}', '(.*?)')
# Add shebang possibility to the LICENSE_RE_PATTERN
LICENSE_RE_PATTERN = r'(\#\!.*\n)?' + LICENSE_RE_PATTERN
LICENSE_RE = re.compile(LICENSE_RE_PATTERN, re.MULTILINE)


# Guess iris repo directory of Iris - realpath is used to mitigate against
# Python finding the iris package via a symlink.
IRIS_DIR = os.path.realpath(os.path.dirname(iris.__file__))
REPO_DIR = os.path.dirname(os.path.dirname(IRIS_DIR))
DOCS_DIR = os.path.join(REPO_DIR, 'docs', 'iris')
DOCS_DIR = iris.config.get_option('Resources', 'doc_dir', default=DOCS_DIR)
exclusion = ['Makefile', 'build']
DOCS_DIRS = glob(os.path.join(DOCS_DIR, '*'))
DOCS_DIRS = [DOC_DIR for DOC_DIR in DOCS_DIRS if os.path.basename(DOC_DIR) not
             in exclusion]


class StandardReportWithExclusions(pep8.StandardReport):
    expected_bad_files = [
        '*/iris/std_names.py',
        '*/iris/analysis/interpolate.py',
        '*/iris/analysis/trajectory.py',
        '*/iris/fileformats/cf.py',
        '*/iris/fileformats/dot.py',
        '*/iris/fileformats/grib/__init__.py',
        '*/iris/fileformats/grib/_grib_cf_map.py',
        '*/iris/fileformats/grib/load_rules.py',
        '*/iris/fileformats/pp.py',
        '*/iris/fileformats/pp_rules.py',
        '*/iris/fileformats/rules.py',
        '*/iris/fileformats/um_cf_map.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/compiled_pyke_files.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/fc_rules_cf_fc.py',
        '*/iris/io/__init__.py',
        '*/iris/io/format_picker.py',
        '*/iris/tests/__init__.py',
        '*/iris/tests/pp.py',
        '*/iris/tests/stock.py',
        '*/iris/tests/system_test.py',
        '*/iris/tests/test_analysis.py',
        '*/iris/tests/test_analysis_calculus.py',
        '*/iris/tests/test_basic_maths.py',
        '*/iris/tests/test_cartography.py',
        '*/iris/tests/test_cdm.py',
        '*/iris/tests/test_cell.py',
        '*/iris/tests/test_cf.py',
        '*/iris/tests/test_constraints.py',
        '*/iris/tests/test_coord_api.py',
        '*/iris/tests/test_coord_categorisation.py',
        '*/iris/tests/test_coordsystem.py',
        '*/iris/tests/test_cube_to_pp.py',
        '*/iris/tests/test_file_load.py',
        '*/iris/tests/test_file_save.py',
        '*/iris/tests/test_grib_save.py',
        '*/iris/tests/test_grib_save_rules.py',
        '*/iris/tests/test_hybrid.py',
        '*/iris/tests/test_interpolation.py',
        '*/iris/tests/test_intersect.py',
        '*/iris/tests/test_io_init.py',
        '*/iris/tests/test_iterate.py',
        '*/iris/tests/test_load.py',
        '*/iris/tests/test_merge.py',
        '*/iris/tests/test_pickling.py',
        '*/iris/tests/test_pp_cf.py',
        '*/iris/tests/test_pp_module.py',
        '*/iris/tests/test_pp_stash.py',
        '*/iris/tests/test_pp_to_cube.py',
        '*/iris/tests/test_quickplot.py',
        '*/iris/tests/test_regrid.py',
        '*/iris/tests/test_rules.py',
        '*/iris/tests/test_std_names.py',
        '*/iris/tests/test_trajectory.py',
        '*/iris/tests/test_unit.py',
        '*/iris/tests/test_uri_callback.py',
        '*/iris/tests/test_util.py']

    # Auto-generated by install process, though not always.
    optional_bad_files = ['*/iris/fileformats/pp_packing.py']
    expected_bad_files += optional_bad_files

    if DOCS_DIRS:
        expected_bad_docs_files = [
            '*/example_code/General/SOI_filtering.py',
            '*/example_code/General/cross_section.py',
            '*/example_code/General/custom_file_loading.py',
            '*/example_code/General/global_map.py',
            '*/example_code/Meteorology/COP_1d_plot.py',
            '*/example_code/Meteorology/COP_maps.py',
            '*/example_code/Meteorology/hovmoller.py',
            '*/example_code/Meteorology/lagged_ensemble.py',
            '*/src/conf.py',
            '*/src/developers_guide/gitwash_dumper.py',
            '*/src/userguide/plotting_examples/1d_with_legend.py']

        expected_bad_files += expected_bad_docs_files

    matched_exclusions = set()

    def get_file_results(self):
        # If the file had no errors, return self.file_errors (which will be 0)
        if not self._deferred_print:
            return self.file_errors

        # Iterate over all of the patterns, to find a possible exclusion. If we
        # the filename is to be excluded, go ahead and remove the counts that
        # self.error added.
        for pattern in self.expected_bad_files:
            if fnmatch(self.filename, pattern):
                self.matched_exclusions.add(pattern)
                # invert the error method's counters.
                for _, _, code, _, _ in self._deferred_print:
                    self.counters[code] -= 1
                    if self.counters[code] == 0:
                        self.counters.pop(code)
                        self.messages.pop(code)
                    self.file_errors -= 1
                    self.total_errors -= 1
                return self.file_errors

        # Otherwise call the superclass' method to print the bad results.
        return super(StandardReportWithExclusions,
                     self).get_file_results()


class TestCodeFormat(unittest.TestCase):
    def test_pep8_conformance(self):
        #
        #    Tests the iris codebase against the "pep8" tool.
        #
        #    Users can add their own excluded files (should files exist in the
        #    local directory which is not in the repository) by adding a
        #    ".pep8_test_exclude.txt" file in the same directory as this test.
        #    The file should be a line separated list of filenames/directories
        #    as can be passed to the "pep8" tool's exclude list.

        # To get a list of bad files, rather than the specific errors, add
        # "reporter=pep8.FileReport" to the StyleGuide constructor.
        pep8style = pep8.StyleGuide(quiet=False,
                                    reporter=StandardReportWithExclusions)

        # Allow users to add their own exclude list.
        extra_exclude_file = os.path.join(os.path.dirname(__file__),
                                          '.pep8_test_exclude.txt')
        if os.path.exists(extra_exclude_file):
            with open(extra_exclude_file, 'r') as fh:
                extra_exclude = [line.strip() for line in fh if line.strip()]
            pep8style.options.exclude.extend(extra_exclude)

        check_paths = [os.path.dirname(iris.__file__)]
        if DOCS_DIRS:
            check_paths.extend(DOCS_DIRS)

        result = pep8style.check_files(check_paths)
        self.assertEqual(result.total_errors, 0, "Found code syntax "
                                                 "errors (and warnings).")

        reporter = pep8style.options.reporter
        # If we've been using the exclusions reporter, check that we didn't
        # exclude files unnecessarily.
        if reporter is StandardReportWithExclusions:
            unexpectedly_good = sorted(set(reporter.expected_bad_files) -
                                       set(reporter.optional_bad_files) -
                                       reporter.matched_exclusions)

            if unexpectedly_good:
                self.fail('Some exclude patterns were unnecessary as the '
                          'files they pointed to either passed the PEP8 tests '
                          'or do not point to a file:\n  '
                          '{}'.format('\n  '.join(unexpectedly_good)))


class TestLicenseHeaders(unittest.TestCase):
    @staticmethod
    def years_of_license_in_file(fh):
        """
        Using :data:`LICENSE_RE` look for the years defined in the license
        header of the given file handle.

        If the license cannot be found in the given fh, None will be returned,
        else a tuple of (start_year, end_year) will be returned.

        """
        license_matches = LICENSE_RE.match(fh.read())
        if not license_matches:
            # no license found in file.
            return None

        years = license_matches.groups()[-1]
        if len(years) == 4:
            start_year = end_year = int(years)
        elif len(years) == 11:
            start_year, end_year = int(years[:4]), int(years[7:])
        else:
            fname = getattr(fh, 'name', 'unknown filename')
            raise ValueError("Unexpected year(s) string in {}'s copyright "
                             "notice: {!r}".format(fname, years))
        return (start_year, end_year)

    @staticmethod
    def whatchanged_parse(whatchanged_output):
        """
        Returns a generator of tuples of data parsed from
        "git whatchanged --pretty='TIME:%at". The tuples are of the form
        ``(filename, last_commit_datetime)``

        Sample input::

            ['TIME:1366884020', '',
             ':000000 100644 0000000... 5862ced... A\tlib/iris/cube.py']

        """
        dt = None
        for line in whatchanged_output:
            if not line.strip():
                continue
            elif line.startswith('TIME:'):
                dt = datetime.fromtimestamp(int(line[5:]))
            else:
                # Non blank, non date, line -> must be the lines
                # containing the file info.
                fname = ' '.join(line.split('\t')[1:])
                yield fname, dt

    @staticmethod
    def last_change_by_fname():
        """
        Return a dictionary of all the files under git which maps to
        the datetime of their last modification in the git history.

        .. note::

            This function raises a ValueError if the repo root does
            not have a ".git" folder. If git is not installed on the system,
            or cannot be found by subprocess, an IOError may also be raised.

        """
        # Check the ".git" folder exists at the repo dir.
        if not os.path.isdir(os.path.join(REPO_DIR, '.git')):
            raise ValueError('{} is not a git repository.'.format(REPO_DIR))

        # Call "git whatchanged" to get the details of all the files and when
        # they were last changed.
        output = subprocess.check_output(['git', 'whatchanged',
                                          "--pretty=TIME:%ct"],
                                         cwd=REPO_DIR)
        output = output.decode().split('\n')
        res = {}
        for fname, dt in TestLicenseHeaders.whatchanged_parse(output):
            if fname not in res or dt > res[fname]:
                res[fname] = dt

        return res

    def test_license_headers(self):
        exclude_patterns = ('setup.py',
                            'build/*',
                            'dist/*',
                            'docs/iris/example_code/*/*.py',
                            'docs/iris/src/developers_guide/documenting/*.py',
                            'docs/iris/src/sphinxext/gen_gallery.py',
                            'docs/iris/src/userguide/plotting_examples/*.py',
                            'docs/iris/src/userguide/regridding_plots/*.py',
                            'docs/iris/src/developers_guide/gitwash_dumper.py',
                            'docs/iris/build/*',
                            'lib/iris/analysis/_scipy_interpolate.py',
                            'lib/iris/fileformats/_pyke_rules/*',
                            'lib/iris/fileformats/grib/_grib_cf_map.py')

        try:
            last_change_by_fname = self.last_change_by_fname()
        except ValueError:
            # Caught the case where this is not a git repo.
            return self.skipTest('Iris installation did not look like a '
                                 'git repo.')

        failed = False
        for fname, last_change in sorted(last_change_by_fname.items()):
            full_fname = os.path.join(REPO_DIR, fname)
            if full_fname.endswith('.py') and os.path.isfile(full_fname) and \
                    not any(fnmatch(fname, pat) for pat in exclude_patterns):
                with open(full_fname) as fh:
                    years = TestLicenseHeaders.years_of_license_in_file(fh)
                    if years is None:
                        print('The file {} has no valid header license and '
                              'has not been excluded from the license header '
                              'test.'.format(fname))
                        failed = True
                    elif last_change.year > years[1]:
                        print('The file header at {} is out of date. The last'
                              ' commit was in {}, but the copyright states it'
                              ' was {}.'.format(fname, last_change.year,
                                                years[1]))
                        failed = True

        if failed:
            raise ValueError('There were license header failures. See stdout.')


class TestFutureImports(unittest.TestCase):
    excluded = (
        '*/iris/fileformats/pp_packing.py',
        '*/iris/fileformats/_pyke_rules/__init__.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/__init__.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/compiled_pyke_files.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/fc_rules_cf_fc.py',
        '*/docs/iris/example_code/*/*.py',
        '*/docs/iris/src/developers_guide/documenting/*.py',
    )

    future_imports_pattern = re.compile(
        r"^from __future__ import \(absolute_import,\s*division,\s*"
        r"print_function(,\s*unicode_literals)?\)$",
        flags=re.MULTILINE)

    six_import_pattern = re.compile(
        r"^from six.moves import \(filter, input, map, range, zip\)  # noqa$",
        flags=re.MULTILINE)

    def test_future_imports(self):
        # Tests that every single Python file includes the appropriate
        # __future__ import to enforce consistent behaviour.
        check_paths = [os.path.dirname(iris.__file__)]
        if DOCS_DIRS:
            check_paths.extend(DOCS_DIRS)

        failed = False
        for dirpath, _, files in chain.from_iterable(os.walk(path)
                                                     for path in check_paths):
            for fname in files:
                full_fname = os.path.join(dirpath, fname)
                if not full_fname.endswith('.py'):
                    continue
                if not os.path.isfile(full_fname):
                    continue
                if any(fnmatch(full_fname, pat) for pat in self.excluded):
                    continue

                with open(full_fname, "r") as fh:
                    content = fh.read()

                    if re.search(self.future_imports_pattern, content) is None:
                        print('The file {} has no valid __future__ imports '
                              'and has not been excluded from the imports '
                              'test.'.format(full_fname))
                        failed = True

                    if re.search(self.six_import_pattern, content) is None:
                        print('The file {} has no valid six import '
                              'and has not been excluded from the imports '
                              'test.'.format(full_fname))
                        failed = True

        if failed:
            raise AssertionError('There were Python 3 compatibility import '
                                 'check failures. See stdout.')


if __name__ == '__main__':
    unittest.main()

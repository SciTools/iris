# (C) British Crown Copyright 2013, Met Office
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


from fnmatch import fnmatch
import os
import unittest

import pep8

import iris


class StandardReportWithExclusions(pep8.StandardReport):
    expected_bad_files = [
        '*/iris/_cube_coord_common.py',
        '*/iris/_merge.py',
        '*/iris/aux_factory.py',
        '*/iris/config.py',
        '*/iris/coord_categorisation.py',
        '*/iris/coord_systems.py',
        '*/iris/cube.py',
        '*/iris/exceptions.py',
        '*/iris/iterate.py',
        '*/iris/palette.py',
        '*/iris/pandas.py',
        '*/iris/proxy.py',
        '*/iris/quickplot.py',
        '*/iris/std_names.py',
        '*/iris/symbols.py',
        '*/iris/unit.py',
        '*/iris/util.py',
        '*/iris/analysis/__init__.py',
        '*/iris/analysis/calculus.py',
        '*/iris/analysis/cartography.py',
        '*/iris/analysis/geometry.py',
        '*/iris/analysis/interpolate.py',
        '*/iris/analysis/maths.py',
        '*/iris/analysis/trajectory.py',
        '*/iris/fileformats/__init__.py',
        '*/iris/fileformats/cf.py',
        '*/iris/fileformats/dot.py',
        '*/iris/fileformats/ff.py',
        '*/iris/fileformats/grib.py',
        '*/iris/fileformats/grib_save_rules.py',
        '*/iris/fileformats/manager.py',
        '*/iris/fileformats/mosig_cf_map.py',
        '*/iris/fileformats/netcdf.py',
        '*/iris/fileformats/pp.py',
        '*/iris/fileformats/rules.py',
        '*/iris/fileformats/um_cf_map.py',
        '*/iris/fileformats/_pyke_rules/__init__.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/compiled_pyke_files.py',
        '*/iris/fileformats/_pyke_rules/compiled_krb/fc_rules_cf_fc.py',
        '*/iris/io/__init__.py',
        '*/iris/io/format_picker.py',
        '*/iris/tests/__init__.py',
        '*/iris/tests/idiff.py',
        '*/iris/tests/pp.py',
        '*/iris/tests/stock.py',
        '*/iris/tests/system_test.py',
        '*/iris/tests/test_aggregate_by.py',
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
        '*/iris/tests/test_grib_load.py',
        '*/iris/tests/test_grib_save.py',
        '*/iris/tests/test_grib_save_rules.py',
        '*/iris/tests/test_hybrid.py',
        '*/iris/tests/test_interpolation.py',
        '*/iris/tests/test_intersect.py',
        '*/iris/tests/test_io_init.py',
        '*/iris/tests/test_iterate.py',
        '*/iris/tests/test_load.py',
        '*/iris/tests/test_mapping.py',
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
        '*/iris/tests/test_util.py',
        '*/iris/tests/test_verbose_logging.py']

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
#        Tests the iris codebase against the "pep8" tool.
#
#        Users can add their own excluded files (should files exist in the
#        local directory which is not in the repository) by adding a
#        ".pep8_test_exclude.txt" file in the same directory as this test.
#        The file should be a line separated list of filenames/directories
#        as can be passed to the "pep8" tool's exclude list.

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

        result = pep8style.check_files([os.path.dirname(iris.__file__)])
        self.assertEqual(result.total_errors, 0, "Found code syntax "
                                                 "errors (and warnings).")

        reporter = pep8style.options.reporter
        # If we've been using the exclusions reporter, check that we didn't
        # exclude files unnecessarily.
        if reporter is StandardReportWithExclusions:
            unexpectedly_good = sorted(set(reporter.expected_bad_files) -
                                       reporter.matched_exclusions)

            if unexpectedly_good:
                self.fail('Some exclude patterns were unnecessary as the '
                          'files they pointed to either passed the PEP8 tests '
                          'or do not point to a file:\n  '
                          '{}'.format('\n  '.join(unexpectedly_good)))


if __name__ == '__main__':
    unittest.main()

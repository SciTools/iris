# (C) British Crown Copyright 2014 - 2017, Met Office
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
Tests for function
:func:`iris.fileformats.grib._load_convert.generating_process`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris.tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.fileformats.grib._load_convert import generating_process


class TestGeneratingProcess(tests.IrisTest):
    def setUp(self):
        self.warn_patch = self.patch('warnings.warn')

    def test_nowarn(self):
        generating_process(None)
        self.assertEqual(self.warn_patch.call_count, 0)

    def _check_warnings(self, with_forecast=True):
        module = 'iris_grib._load_convert'
        self.patch(module + '.options.warn_on_unsupported', True)
        call_args = [None]
        call_kwargs = {}
        expected_fragments = [
            'Unable to translate type of generating process',
            'Unable to translate background generating process']
        if with_forecast:
            expected_fragments.append(
                'Unable to translate forecast generating process')
        else:
            call_kwargs['include_forecast_process'] = False
        generating_process(*call_args, **call_kwargs)
        got_msgs = [call[0][0] for call in self.warn_patch.call_args_list]
        for got_msg, expected_fragment in zip(sorted(got_msgs),
                                              sorted(expected_fragments)):
            self.assertIn(expected_fragment, got_msg)

    def test_warn_full(self):
        self._check_warnings()

    def test_warn_no_forecast(self):
        self._check_warnings(with_forecast=False)


if __name__ == '__main__':
    tests.main()

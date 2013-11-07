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
"""Unit tests for the `iris.fileformats.pp._field_gen` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import contextlib

import mock
import numpy as np

import iris.fileformats.pp as pp


class Test__field_gen(tests.IrisTest):
    @contextlib.contextmanager
    def mock_for_field_gen(self, fields):
        side_effect_fields = list(fields)[:]

        def make_pp_field_override(*args):
            # Iterates over the fields passed to this context manager,
            # until there are no more, upon which the np.fromfile
            # returns an empty list and the while loop in load() is
            # broken.
            result = side_effect_fields.pop(0)
            if not side_effect_fields:
                np.fromfile.return_value = []
            return result

        with mock.patch('numpy.fromfile', return_value=[0]), \
                mock.patch('__builtin__.open'), \
                mock.patch('struct.unpack_from', return_value=[4]), \
                mock.patch('iris.fileformats.pp.make_pp_field',
                           side_effect=make_pp_field_override):
            yield

    def gen_fields(self, fields):
        with self.mock_for_field_gen(fields):
            return list(pp._field_gen('mocked', 'mocked'))

    def test_lblrec_invalid(self):
        pp_field = mock.Mock(lblrec=2,
                             lbext=0)
        with self.assertRaises(ValueError) as err:
            self.gen_fields([pp_field])
        self.assertEqual(str(err.exception),
                         ('LBLREC has a different value to the integer '
                          'recorded after the header in the file (8 '
                          'and 4).'))

    def test_read_headers_call(self):
        # Checks that the two calls to np.fromfile are called in the
        # expected way.
        pp_field = mock.Mock(lblrec=1,
                             lbext=0,
                             lbuser=[0])
        with self.mock_for_field_gen([pp_field]):
            open_fh = mock.Mock()
            open.return_value = open_fh
            next(pp._field_gen('mocked', read_data_bytes=False))
            calls = [mock.call(open_fh, count=45, dtype='>i4'),
                     mock.call(open_fh, count=19, dtype='>f4')]
            np.fromfile.assert_has_calls(calls)
        expected_deferred_bytes = pp.DeferredArrayBytes('mocked',
                                                        open_fh.tell(),
                                                        4, np.dtype('>f4'))
        self.assertEqual(pp_field._data, expected_deferred_bytes)

    def test_read_data_call(self):
        # Checks that data is read if read_data is True.
        pp_field = mock.Mock(lblrec=1,
                             lbext=0,
                             lbuser=[0])
        with self.mock_for_field_gen([pp_field]):
            open_fh = mock.Mock()
            open.return_value = open_fh
            next(pp._field_gen('mocked', read_data_bytes=True))
        expected_loaded_bytes = pp.LoadedArrayBytes(open_fh.read(),
                                                    np.dtype('>f4'))
        self.assertEqual(pp_field._data, expected_loaded_bytes)

if __name__ == "__main__":
    tests.main()

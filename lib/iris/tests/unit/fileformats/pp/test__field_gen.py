# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp._field_gen` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import contextlib
import io
from unittest import mock
import warnings

import numpy as np

import iris.fileformats.pp as pp


class Test(tests.IrisTest):
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

        open_func = "builtins.open"
        with mock.patch("numpy.fromfile", return_value=[0]), mock.patch(
            open_func
        ), mock.patch("struct.unpack_from", return_value=[4]), mock.patch(
            "iris.fileformats.pp.make_pp_field",
            side_effect=make_pp_field_override,
        ):
            yield

    def gen_fields(self, fields):
        with self.mock_for_field_gen(fields):
            return list(pp._field_gen("mocked", "mocked"))

    def test_lblrec_invalid(self):
        pp_field = mock.Mock(lblrec=2, lbext=0)
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            self.gen_fields([pp_field])
        self.assertEqual(len(warn), 1)
        wmsg = (
            "LBLREC has a different value to the .* the header in the "
            r"file \(8 and 4\)\. Skipping .*"
        )
        self.assertRegex(str(warn[0].message), wmsg)

    def test_read_headers_call(self):
        # Checks that the two calls to np.fromfile are called in the
        # expected way.
        pp_field = mock.Mock(lblrec=1, lbext=0, lbuser=[0])
        with self.mock_for_field_gen([pp_field]):
            open_fh = mock.MagicMock(spec=io.RawIOBase)
            open.return_value = open_fh
            next(pp._field_gen("mocked", read_data_bytes=False))
            with open_fh as open_fh_ctx:
                calls = [
                    mock.call(open_fh_ctx, count=45, dtype=">i4"),
                    mock.call(open_fh_ctx, count=19, dtype=">f4"),
                ]
            np.fromfile.assert_has_calls(calls)
        with open_fh as open_fh_ctx:
            expected_deferred_bytes = (
                "mocked",
                open_fh_ctx.tell(),
                4,
                np.dtype(">f4"),
            )
        self.assertEqual(pp_field.data, expected_deferred_bytes)

    def test_read_data_call(self):
        # Checks that data is read if read_data is True.
        pp_field = mock.Mock(lblrec=1, lbext=0, lbuser=[0])
        with self.mock_for_field_gen([pp_field]):
            open_fh = mock.MagicMock(spec=io.RawIOBase)
            open.return_value = open_fh
            next(pp._field_gen("mocked", read_data_bytes=True))
        with open_fh as open_fh_ctx:
            expected_loaded_bytes = pp.LoadedArrayBytes(
                open_fh_ctx.read(), np.dtype(">f4")
            )
        self.assertEqual(pp_field.data, expected_loaded_bytes)

    def test_invalid_header_release(self):
        # Check that an unknown LBREL value just results in a warning
        # and the end of the file iteration instead of raising an error.
        with self.temp_filename() as temp_path:
            np.zeros(65, dtype="i4").tofile(temp_path)
            generator = pp._field_gen(temp_path, False)
            with mock.patch("warnings.warn") as warn:
                with self.assertRaises(StopIteration):
                    next(generator)
            self.assertEqual(warn.call_count, 1)
            self.assertIn("header release number", warn.call_args[0][0])


if __name__ == "__main__":
    tests.main()

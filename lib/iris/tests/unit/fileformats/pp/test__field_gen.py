# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp._field_gen` function."""

import contextlib
import io

import numpy as np
import pytest

import iris.fileformats.pp as pp


class Test:
    @pytest.fixture()
    def mock_for_field_gen(self, mocker):
        @contextlib.contextmanager
        def _mock_for_field_gen(fields):
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
            mocker.patch("numpy.fromfile", return_value=[0])
            mocker.patch(open_func)
            mocker.patch("struct.unpack_from", return_value=[4])
            mocker.patch(
                "iris.fileformats.pp.make_pp_field",
                side_effect=make_pp_field_override,
            )
            yield

        return _mock_for_field_gen

    def gen_fields(self, fields, mock_for_field_gen):
        with mock_for_field_gen(fields):
            return list(pp._field_gen("mocked", "mocked"))

    def test_lblrec_invalid(self, mocker, mock_for_field_gen):
        pp_field = mocker.Mock(lblrec=2, lbext=0)
        wmsg = (
            "LBLREC has a different value to the .* the header in the "
            r"file \(8 and 4\)\. Skipping .*"
        )
        with pytest.warns(UserWarning, match=wmsg) as warn:
            self.gen_fields([pp_field], mock_for_field_gen)
        assert len(warn) == 1

    def test_read_headers_call(self, mocker, mock_for_field_gen):
        # Checks that the two calls to np.fromfile are called in the
        # expected way.
        pp_field = mocker.Mock(lblrec=1, lbext=0, lbuser=[0])
        with mock_for_field_gen([pp_field]):
            open_fh = mocker.MagicMock(spec=io.RawIOBase)
            open.return_value = open_fh
            next(pp._field_gen("mocked", read_data_bytes=False))
            with open_fh as open_fh_ctx:
                calls = [
                    mocker.call(open_fh_ctx, count=45, dtype=">i4"),
                    mocker.call(open_fh_ctx, count=19, dtype=">f4"),
                ]
            np.fromfile.assert_has_calls(calls)
        with open_fh as open_fh_ctx:
            expected_deferred_bytes = (
                "mocked",
                open_fh_ctx.tell(),
                4,
                np.dtype(">f4"),
            )
        assert pp_field.data == expected_deferred_bytes

    def test_read_data_call(self, mocker, mock_for_field_gen):
        # Checks that data is read if read_data is True.
        pp_field = mocker.Mock(lblrec=1, lbext=0, lbuser=[0])
        with mock_for_field_gen([pp_field]):
            open_fh = mocker.MagicMock(spec=io.RawIOBase)
            open.return_value = open_fh
            next(pp._field_gen("mocked", read_data_bytes=True))
        with open_fh as open_fh_ctx:
            expected_loaded_bytes = pp.LoadedArrayBytes(
                open_fh_ctx.read(), np.dtype(">f4")
            )
        assert pp_field.data == expected_loaded_bytes

    def test_invalid_header_release(self, tmp_path):
        # Check that an unknown LBREL value just results in a warning
        # and the end of the file iteration instead of raising an error.
        temp_path = tmp_path / "temp"
        np.zeros(65, dtype="i4").tofile(temp_path)
        generator = pp._field_gen(temp_path, False)
        with pytest.warns(
            pp._WarnComboIgnoringLoad, match="header release number"
        ) as warn:
            with pytest.raises(StopIteration):
                next(generator)
            assert len(warn) == 1

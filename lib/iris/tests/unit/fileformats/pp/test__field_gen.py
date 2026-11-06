# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp._field_gen` function."""

import contextlib
import io
import struct

import numpy as np
import pytest

import iris.fileformats.pp as pp

# Byte size of the full header record that _field_gen reads in one go:
#   leading length word + long headers + float headers + trailing length word
_HEADER_BYTES = pp.PP_WORD_DEPTH * (1 + pp.NUM_LONG_HEADERS + pp.NUM_FLOAT_HEADERS + 1)
# A valid data-length word: 4 bytes encoding the value 4 (big-endian uint32),
# matching lblrec=1 * PP_WORD_DEPTH=4 so LBLREC validation passes.
_DATA_LEN_WORD = struct.pack(">L", 4)


class TestFieldGen:
    @pytest.fixture
    def mock_for_field_gen(self, mocker):
        @contextlib.contextmanager
        def _mock_for_field_gen(fields):
            side_effect_fields = list(fields)[:]

            def make_pp_field_override(*args):
                return side_effect_fields.pop(0)

            # Build the sequence of bytes that pp_file.read() will return:
            #   For each field: a _HEADER_BYTES-sized buffer (all zeros is fine
            #   for our purposes — make_pp_field is fully mocked), followed by
            #   a 4-byte data-length word.
            # After all fields: b"" to signal EOF on the next header read.
            read_side_effects = []
            for _ in fields:
                read_side_effects.append(bytes(_HEADER_BYTES))  # header read
                read_side_effects.append(_DATA_LEN_WORD)  # data-len word
            read_side_effects.append(b"")  # EOF

            mock_file = mocker.MagicMock(spec=io.RawIOBase)
            mock_file.__enter__ = mocker.Mock(return_value=mock_file)
            mock_file.__exit__ = mocker.Mock(return_value=False)
            mock_file.read.side_effect = read_side_effects

            mocker.patch("builtins.open", return_value=mock_file)
            mocker.patch("struct.unpack_from", return_value=[4])
            mocker.patch(
                "iris.fileformats.pp.make_pp_field",
                side_effect=make_pp_field_override,
            )
            yield mock_file

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
        # Checks that the file is read in a single call of _HEADER_BYTES and
        # that np.frombuffer is used to parse longs and floats from that buffer.
        pp_field = mocker.Mock(lblrec=1, lbext=0, lbuser=[0])
        mock_frombuffer = mocker.patch("numpy.frombuffer", wraps=np.frombuffer)
        with mock_for_field_gen([pp_field]) as mock_file:
            next(pp._field_gen("mocked", read_data_bytes=False))
        # The first read() call should request exactly _HEADER_BYTES bytes.
        first_read_call = mock_file.read.call_args_list[0]
        assert first_read_call == mocker.call(_HEADER_BYTES)

        # frombuffer should have been called twice: once for longs, once for floats.
        assert mock_frombuffer.call_count == 2
        calls = mock_frombuffer.call_args_list
        assert calls[0].kwargs["count"] == pp.NUM_LONG_HEADERS
        assert calls[0].kwargs["dtype"] == np.dtype(">i4")
        assert calls[1].kwargs["count"] == pp.NUM_FLOAT_HEADERS
        assert calls[1].kwargs["dtype"] == np.dtype(">f4")

    def test_read_data_call(self, mocker, mock_for_field_gen):
        # Checks that data is read if read_data is True.
        pp_field = mocker.Mock(lblrec=1, lbext=0, lbuser=[0])
        with mock_for_field_gen([pp_field]) as mock_file:
            next(pp._field_gen("mocked", read_data_bytes=True))
        # The third read() call (index 2) should be the data payload read
        # with data_len = lblrec*PP_WORD_DEPTH - lbext*PP_WORD_DEPTH = 4 bytes.
        data_read_call = mock_file.read.call_args_list[2]
        assert data_read_call == mocker.call(4)
        assert isinstance(pp_field.data, pp.LoadedArrayBytes)

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

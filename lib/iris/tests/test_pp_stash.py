# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import pytest

import iris
import iris.fileformats.pp
import iris.io
from iris.tests import _shared_utils
import iris.tests.stock
import iris.util


class TestPPStash:
    @_shared_utils.skip_data
    def test_cube_attributes(self):
        cube = iris.tests.stock.simple_pp()
        assert "m01s16i203" == cube.attributes["STASH"]
        assert "m01s16i999" != cube.attributes["STASH"]
        # Also exercise iris.fileformats.pp.STASH eq and ne methods.
        assert cube.attributes["STASH"] == "m01s16i203"
        assert cube.attributes["STASH"] != "m01s16i999"

    @_shared_utils.skip_data
    def test_ppfield(self):
        data_path = _shared_utils.get_data_path(("PP", "simple_pp", "global.pp"))
        pps = iris.fileformats.pp.load(data_path)
        for pp in pps:
            assert "m01s16i203" == pp.stash
            assert "m01s16i999" != pp.stash
            # Also exercise iris.fileformats.pp.STASH eq and ne methods.
            assert pp.stash == "m01s16i203"
            assert pp.stash != "m01s16i999"

    def test_stash_against_stash(self):
        assert iris.fileformats.pp.STASH(1, 2, 3) == iris.fileformats.pp.STASH(1, 2, 3)
        assert iris.fileformats.pp.STASH(1, 2, 3) != iris.fileformats.pp.STASH(2, 3, 4)

    def test_stash_against_str(self):
        # Also exercise iris.fileformats.pp.STASH eq and ne methods.
        assert iris.fileformats.pp.STASH(1, 2, 3) == "m01s02i003"
        assert "m01s02i003" == iris.fileformats.pp.STASH(1, 2, 3)
        assert iris.fileformats.pp.STASH(1, 2, 3) != "m02s03i004"
        assert "m02s03i004" != iris.fileformats.pp.STASH(1, 2, 3)

    def test_irregular_stash_str(self):
        # Also exercise iris.fileformats.pp.STASH eq and ne methods.
        assert iris.fileformats.pp.STASH(1, 2, 3) == "m01s02i0000000003"
        assert iris.fileformats.pp.STASH(1, 2, 3) == "m01s02i3"
        assert iris.fileformats.pp.STASH(1, 2, 3) == "m01s2i3"
        assert iris.fileformats.pp.STASH(1, 2, 3) == "m1s2i3"

        assert "m01s02i0000000003" == iris.fileformats.pp.STASH(1, 2, 3)
        assert "m01s02i3" == iris.fileformats.pp.STASH(1, 2, 3)
        assert "m01s2i3" == iris.fileformats.pp.STASH(1, 2, 3)
        assert "m1s2i3" == iris.fileformats.pp.STASH(1, 2, 3)

        assert iris.fileformats.pp.STASH(2, 3, 4) != "m01s02i0000000003"
        assert iris.fileformats.pp.STASH(2, 3, 4) != "m01s02i3"
        assert iris.fileformats.pp.STASH(2, 3, 4) != "m01s2i3"
        assert iris.fileformats.pp.STASH(2, 3, 4) != "m1s2i3"

        assert "m01s02i0000000003" != iris.fileformats.pp.STASH(2, 3, 4)
        assert "m01s02i3" != iris.fileformats.pp.STASH(2, 3, 4)
        assert "m01s2i3" != iris.fileformats.pp.STASH(2, 3, 4)
        assert "m1s2i3" != iris.fileformats.pp.STASH(2, 3, 4)

        assert iris.fileformats.pp.STASH.from_msi("M01s02i003") == "m01s02i003"
        assert "m01s02i003" == iris.fileformats.pp.STASH.from_msi("M01s02i003")

    def test_illegal_stash_str_range(self):
        # Also exercise iris.fileformats.pp.STASH eq and ne methods.
        assert iris.fileformats.pp.STASH(0, 2, 3) == "m??s02i003"
        assert iris.fileformats.pp.STASH(0, 2, 3) != "m01s02i003"

        assert "m??s02i003" == iris.fileformats.pp.STASH(0, 2, 3)
        assert "m01s02i003" != iris.fileformats.pp.STASH(0, 2, 3)

        assert iris.fileformats.pp.STASH(0, 2, 3) == "m??s02i003"
        assert iris.fileformats.pp.STASH(0, 2, 3) == "m00s02i003"
        assert "m??s02i003" == iris.fileformats.pp.STASH(0, 2, 3)
        assert "m00s02i003" == iris.fileformats.pp.STASH(0, 2, 3)

        assert iris.fileformats.pp.STASH(100, 2, 3) == "m??s02i003"
        assert iris.fileformats.pp.STASH(100, 2, 3) == "m100s02i003"
        assert "m??s02i003" == iris.fileformats.pp.STASH(100, 2, 3)
        assert "m100s02i003" == iris.fileformats.pp.STASH(100, 2, 3)

    def test_illegal_stash_stash_range(self):
        assert iris.fileformats.pp.STASH(0, 2, 3) == iris.fileformats.pp.STASH(0, 2, 3)
        assert iris.fileformats.pp.STASH(100, 2, 3) == iris.fileformats.pp.STASH(
            100, 2, 3
        )
        assert iris.fileformats.pp.STASH(100, 2, 3) == iris.fileformats.pp.STASH(
            999, 2, 3
        )

    def test_illegal_stash_format(self):
        test_values = (
            ("abc", (1, 2, 3)),
            ("mlotstmin", (1, 2, 3)),
            ("m01s02003", (1, 2, 3)),
        )

        for test_value, reference in test_values:
            msg = "Expected STASH code .* {!r}".format(test_value)
            with pytest.raises(ValueError, match=msg):
                test_value == iris.fileformats.pp.STASH(*reference)
            with pytest.raises(ValueError, match=msg):
                iris.fileformats.pp.STASH(*reference) == test_value

    def test_illegal_stash_type(self):
        test_values = (
            (102003, "m01s02i003"),
            (["m01s02i003"], "m01s02i003"),
        )

        for test_value, reference in test_values:
            msg = "Expected STASH code .* {!r}".format(test_value)
            with pytest.raises(TypeError, match=msg):
                iris.fileformats.pp.STASH.from_msi(test_value) == reference
            with pytest.raises(TypeError, match=msg):
                reference == iris.fileformats.pp.STASH.from_msi(test_value)

    def test_stash_lbuser(self):
        stash = iris.fileformats.pp.STASH(2, 32, 456)
        assert stash.lbuser6() == 2
        assert stash.lbuser3() == 32456

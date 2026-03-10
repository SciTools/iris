# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.experimental.stratify.relevel` function."""

import sys
import types

import numpy as np
import pytest

from iris.coords import AuxCoord, DimCoord
import iris.tests.stock as stock


@pytest.fixture(autouse=True, scope="module")
def fake_stratify():
    try:
        import stratify
    except:
        fake = types.ModuleType("stratify")
        fake.interpolate = lambda *a, **k: None
        sys.modules["stratify"] = fake
        yield
    finally:
        # Remove fake after tests in this module complete
        sys.modules.pop("stratify", None)


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        cube = stock.simple_3d()[:, :1, :1]
        #: The data from which to get the levels.
        self.src_levels = cube.copy()
        #: The data to interpolate.
        self.cube = cube.copy()
        self.cube.rename("foobar")
        self.cube *= 10
        self.coord = self.src_levels.coord("wibble")
        self.axes = (self.coord, self.coord.name(), None, 0)
        self.patch_interpolate = mocker.patch(
            "iris.experimental.stratify.stratify.interpolate"
        )
        self.patch_interpolate.return_value = np.ones((3, 1, 1))

    def test_broadcast_fail_src_levels(self):
        from iris.experimental.stratify import relevel

        emsg = "Cannot broadcast the cube and src_levels"
        data = np.arange(60).reshape(3, 4, 5)
        with pytest.raises(ValueError, match=emsg):
            relevel(self.cube, AuxCoord(data), [1, 2, 3])

    def test_broadcast_fail_tgt_levels(self):
        from iris.experimental.stratify import relevel

        emsg = "Cannot broadcast the cube and tgt_levels"
        data = np.arange(60).reshape(3, 4, 5)
        with pytest.raises(ValueError, match=emsg):
            relevel(self.cube, self.coord, data)

    def test_standard_input(self):
        from iris.experimental.stratify import relevel

        for axis in self.axes:
            result = relevel(self.cube, self.src_levels, [-1, 0, 5.5], axis=axis)
            expected = DimCoord([-1, 0, 5.5], units=1, long_name="thingness")
            assert expected == result.coord("thingness")
            self.patch_interpolate.assert_called()

    def test_coord_input(self):
        from iris.experimental.stratify import relevel

        source = AuxCoord(self.src_levels.data)
        metadata = self.src_levels.metadata._asdict()
        metadata["coord_system"] = None
        metadata["climatological"] = None
        source.metadata = metadata

        for axis in self.axes:
            result = relevel(self.cube, source, [0, 12, 13], axis=axis)
            assert result.shape == (3, 1, 1)
        self.patch_interpolate.assert_called()

    def test_custom_interpolator(self, mocker):
        from iris.experimental.stratify import relevel

        mock_interpolate = mocker.Mock()
        mock_interpolate.return_value = np.ones((3, 1, 1))

        interpolator = mock_interpolate

        for axis in self.axes:
            _ = relevel(
                self.cube,
                self.src_levels,
                [-1, 0, 6.5],
                axis=axis,
                interpolator=interpolator,
            )
            mock_interpolate.assert_called()

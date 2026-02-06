# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp._interpret_field` function."""

from copy import deepcopy

import numpy as np
import pytest

import iris.fileformats.pp as pp
from iris.warnings import IrisLoadWarning


class Test__interpret_fields__land_packed_fields:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        return_value = ("dummy", 0, 0, np.dtype("f4"))
        core_data = mocker.MagicMock(return_value=return_value)
        # A field packed using a land/sea mask.
        self.pp_field = mocker.Mock(
            lblrec=1,
            lbext=0,
            lbuser=[0] * 7,
            lbrow=0,
            lbnpt=0,
            raw_lbpack=21,
            lbpack=mocker.Mock(n1=0, n2=2, n3=1),
            core_data=core_data,
        )
        # The field specifying the land/seamask.
        lbuser = [None, None, None, 30, None, None, 1]  # m01s00i030
        self.land_mask_field = mocker.Mock(
            lblrec=1,
            lbext=0,
            lbuser=lbuser,
            lbrow=3,
            lbnpt=4,
            raw_lbpack=0,
            core_data=core_data,
        )

    def test_non_deferred_fix_lbrow_lbnpt(self):
        # Checks the fix_lbrow_lbnpt is applied to fields which are not
        # deferred.
        f1, mask = self.pp_field, self.land_mask_field
        assert f1.lbrow == 0
        assert f1.lbnpt == 0
        list(pp._interpret_fields([mask, f1]))
        assert f1.lbrow == 3
        assert f1.lbnpt == 4
        # Check the data's shape has been updated too.
        assert f1.data.shape == (3, 4)

    def test_fix_lbrow_lbnpt_no_mask_available(self):
        # Check a warning is issued when loading a land masked field
        # without a land mask.
        with pytest.warns(
            IrisLoadWarning,
            match="Landmask compressed fields existed without a landmask",
        ) as warn:
            list(pp._interpret_fields([self.pp_field]))
        assert len(warn) == 1

    def test_deferred_mask_field(self):
        # Check that the order of the load is yielded last if the mask
        # hasn't yet been seen.
        result = list(pp._interpret_fields([self.pp_field, self.land_mask_field]))
        assert result == [self.land_mask_field, self.pp_field]

    def test_not_deferred_mask_field(self):
        # Check that the order of the load is unchanged if a land mask
        # has already been seen.
        f1, mask = self.pp_field, self.land_mask_field
        mask2 = deepcopy(mask)
        result = list(pp._interpret_fields([mask, f1, mask2]))
        assert result == [mask, f1, mask2]

    def test_deferred_fix_lbrow_lbnpt(self):
        # Check the fix is also applied to fields which are deferred.
        f1, mask = self.pp_field, self.land_mask_field
        list(pp._interpret_fields([f1, mask]))
        assert f1.lbrow == 3
        assert f1.lbnpt == 4

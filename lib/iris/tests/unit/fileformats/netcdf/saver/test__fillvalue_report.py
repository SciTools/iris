# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.fileformats.netcdf.saver._fillvalue_report`.
"""
import warnings

import numpy as np
import pytest

from iris.fileformats.netcdf._thread_safe_nc import default_fillvals
from iris.fileformats.netcdf.saver import (
    SaverFillValueWarning,
    _fillvalue_report,
    _FillvalueCheckInfo,
)


class Test__fillvaluereport:
    @pytest.mark.parametrize(
        "is_bytes", [True, False], ids=["ByteData", "NonbyteData"]
    )
    @pytest.mark.parametrize(
        "is_masked", [True, False], ids=["MaskedData", "NonmaskedData"]
    )
    @pytest.mark.parametrize(
        "contains_fv", [True, False], ids=["FillInData", "NofillInData"]
    )
    @pytest.mark.parametrize(
        "given_user_fv", [True, False], ids=["WithUserfill", "NoUserfill"]
    )
    def test_fillvalue_checking(
        self, is_bytes, is_masked, contains_fv, given_user_fv
    ):
        dtype_code = "u1" if is_bytes else "f4"
        dtype = np.dtype(dtype_code)
        if given_user_fv:
            user_fill = 123 if is_bytes else 1.234
            check_value = user_fill
        else:
            user_fill = None
            check_value = default_fillvals[dtype_code]

        fill_info = _FillvalueCheckInfo(
            user_value=user_fill,
            check_value=check_value,
            dtype=dtype,
            varname="<testvar>",
        )

        # Work out expected action, according to intended logic.
        if is_bytes and is_masked and not given_user_fv:
            msg_fragment = "'<testvar>' contains byte data with masked points"
        elif contains_fv:
            msg_fragment = "'<testvar>' contains unmasked data points equal to the fill-value"
        else:
            msg_fragment = None

        # Trial the action
        result = _fillvalue_report(
            fill_info,
            is_masked=is_masked,
            contains_fill_value=contains_fv,
            warn=False,
        )

        # Check the result
        if msg_fragment is None:
            assert result is None
        else:
            assert isinstance(result, Warning)
            assert msg_fragment in result.args[0]

    @pytest.mark.parametrize(
        "has_collision",
        [True, False],
        ids=["WithFvCollision", "NoFvCollision"],
    )
    def test_warn(self, has_collision):
        fill_info = _FillvalueCheckInfo(
            user_value=1.23,
            check_value=1.23,
            dtype=np.float32,
            varname="<testvar>",
        )

        # Check results
        if has_collision:
            # Check that we get the expected warning
            expected_msg = "'<testvar>' contains unmasked data points equal to the fill-value"
            # Enter a warnings context that checks for the error.
            warning_context = pytest.warns(
                SaverFillValueWarning, match=expected_msg
            )
            warning_context.__enter__()
        else:
            # Check that we get NO warning of the expected type.
            warnings.filterwarnings("error", category=SaverFillValueWarning)

        # Do call: it should raise AND return a warning, ONLY IF there was a collision.
        result = _fillvalue_report(
            fill_info,
            is_masked=True,
            contains_fill_value=has_collision,
            warn=True,
        )

        # Check result
        if has_collision:
            # Fail if no warning was raised ..
            warning_context.__exit__(None, None, None)
            # .. or result does not have the expected message content
            assert expected_msg in result.args[0]
        else:
            # Fail if any warning result was produced.
            assert result is None

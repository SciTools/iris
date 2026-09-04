# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFLabelVariable`."""

import numpy as np
import pytest

from iris.fileformats.cf import CFDataVariable, CFLabelVariable

from .identify_mixins import (
    IdentifyByAttributeMixin,
    SpansMixin,
    _NetCDFVar,
    _NetCDFVarWithDimensions,
)


class TestIdentify(IdentifyByAttributeMixin):
    __test__ = True

    CF_CLASS = CFLabelVariable
    CF_IDENTITIES = ["coordinates"]
    MISSING_WARN_REGEX = r"Missing CF-netCDF label variable {subject!r}.*"
    SUBJECT_DTYPE_DEFAULT = np.bytes_

    def test_non_string_ref_ignored(self):
        # Label identify should reject non-string referenced variables.
        subject_name = "ref_subject"
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            subject_name: _NetCDFVar(subject_name, dtype=int),
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert result == {}


class TestCfLabelDimensions:
    def _make_label_var(self, label_dims):
        stub = _NetCDFVarWithDimensions("label_var", label_dims, dtype=np.bytes_)
        return CFLabelVariable("label_var", stub)

    def _make_data_var(self, data_dims):
        stub = _NetCDFVarWithDimensions("data_var", data_dims)
        return CFDataVariable("data_var", stub)

    def test_raises_for_non_cfdata_var(self):
        label_var = self._make_label_var(("x",))
        message = "cf_data_var argument should be of type CFDataVariable"
        with pytest.raises(TypeError, match=message):
            label_var.cf_label_dimensions(object())

    def test_returns_overlap_dimensions(self):
        label_var = self._make_label_var(("x", "strlen"))
        data_var = self._make_data_var(("x", "y"))
        result = label_var.cf_label_dimensions(data_var)
        assert result == ("x",)

    def test_no_overlap_returns_empty(self):
        label_var = self._make_label_var(("a", "b"))
        data_var = self._make_data_var(("x", "y"))
        result = label_var.cf_label_dimensions(data_var)
        assert result == ()


class TestSpans(SpansMixin):
    """Tests for CFLabelVariable.spans()."""

    __test__ = True
    CF_CLASS = CFLabelVariable

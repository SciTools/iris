# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFLabelVariable`."""

import numpy as np
import pytest

from iris.fileformats.cf import CFDataVariable, CFLabelVariable

from .identify_catalogue import IdentifyByAttributeCatalog


class _NetCDFVarWithDimensions:
    """Stub with a dimensions attribute for spans() and cf_label_dimensions() tests."""

    def __init__(self, name, dimensions, dtype=int):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.dimensions = dimensions

    def ncattrs(self):
        return [
            attr
            for attr in self.__dict__
            if not attr.startswith("_") and attr not in ["name", "dtype", "dimensions"]
        ]


class TestIdentify(IdentifyByAttributeCatalog):
    __test__ = True

    CF_CLASS = CFLabelVariable
    CF_IDENTITY = "coordinates"
    MISSING_WARN_REGEX = r"Missing CF-netCDF label variable {subject!r}.*"
    SUBJECT_DTYPE_DEFAULT = np.bytes_

    def test_two_refs(self, named_variable):
        # Label coordinates may be listed space-delimited on a single source.
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITY, " ".join(subject_names))
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        expected = {
            name: self._expected_var(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_non_string_ref_ignored(self, named_variable):
        # Label identify should reject non-string referenced variables.
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        self._set_ref(ref_source, subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=int),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert {} == result


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


class TestSpans:
    """Tests for CFLabelVariable.spans()."""

    def _make_cf_var(self, name, dimensions):
        stub = _NetCDFVarWithDimensions(name, dimensions, dtype=np.bytes_)
        return CFLabelVariable(name, stub)

    def test_empty_dimensions_spans(self):
        """Scalar label variable always spans the target."""
        cf_label = self._make_cf_var("label_var", ())
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_label.spans(cf_target)

    def test_source_trailing_subset_spans(self):
        """source[:-1] (drop string length dim) is subset of target => spans."""
        cf_label = self._make_cf_var("label_var", ("x", "strlen"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_label.spans(cf_target)

    def test_source_leading_subset_spans(self):
        """source[1:] is a subset of target dimensions => spans."""
        cf_label = self._make_cf_var("label_var", ("strlen", "x"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_label.spans(cf_target)

    def test_non_spanning(self):
        """Dimensions that don't fit either slice => does not span."""
        cf_label = self._make_cf_var("label_var", ("x", "b", "c"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert not cf_label.spans(cf_target)

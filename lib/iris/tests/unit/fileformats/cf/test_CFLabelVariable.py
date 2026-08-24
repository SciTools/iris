# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFLabelVariable`."""

import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFDataVariable, CFLabelVariable
import iris.warnings

CF_IDENTITY = "coordinates"


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


class TestIdentify:
    def test_string_ref_identified(self, named_variable):
        """String-dtype variables referenced via 'coordinates' are accepted as labels."""
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name, dtype=np.bytes_)
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: CFLabelVariable(subject_name, ref_subject)}
        result = CFLabelVariable.identify(vars_all)
        assert expected == result

    def test_non_string_ref_ignored(self, named_variable):
        """Non-string dtype variables referenced via 'coordinates' are not labels."""
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=int),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFLabelVariable.identify(vars_all)
        assert {} == result

    def test_two_refs(self, named_variable):
        """One source with two string-dtype coordinate refs yields two label vars."""
        subject_names = ("ref_label_1", "ref_label_2")
        ref_subject_vars = {
            name: named_variable(name, dtype=np.bytes_) for name in subject_names
        }
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, " ".join(subject_names))
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        expected = {
            name: CFLabelVariable(name, var) for name, var in ref_subject_vars.items()
        }
        result = CFLabelVariable.identify(vars_all)
        assert expected == result

    def test_duplicate_refs(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name, dtype=np.bytes_)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            setattr(var, CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_source_vars,
        }

        expected = {subject_name: CFLabelVariable(subject_name, ref_subject)}
        result = CFLabelVariable.identify(vars_all)
        assert expected == result

    def test_ignore(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name, dtype=np.bytes_) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: CFLabelVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFLabelVariable.identify(vars_all, ignore=subject_names[1])
        assert expected == result

    def test_target(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name, dtype=np.bytes_) for name in subject_names
        }

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: CFLabelVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFLabelVariable.identify(vars_all, target=source_names[0])
        assert expected == result

    def test_target_unknown_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFLabelVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFLabelVariable.identify(vars_all, target=object())

    def test_warn(self, named_variable, assert_warning_gated):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            CFLabelVariable.identify(vars_all, warn=warn)

        warn_regex = rf"Missing CF-netCDF label variable {subject_name!r}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )


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

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFBoundaryVariable`."""

import warnings

import pytest

from iris.fileformats.cf import CFBoundaryVariable
import iris.warnings

CF_IDENTITY = "bounds"


class _NetCDFVarWithDimensions:
    """Stub with a dimensions attribute for spans() tests."""

    def __init__(self, name, dimensions, dtype=int):
        import numpy as np

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
    def test_one_ref(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: CFBoundaryVariable(subject_name, ref_subject)}
        result = CFBoundaryVariable.identify(vars_all)
        assert expected == result

    def test_whitespace_padded_ref(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source = named_variable("ref_source")
        # bounds attribute uses .strip() - leading/trailing whitespace OK
        setattr(ref_source, CF_IDENTITY, f"  {subject_name}  ")
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: CFBoundaryVariable(subject_name, ref_subject)}
        result = CFBoundaryVariable.identify(vars_all)
        assert expected == result

    def test_two_refs(self, named_variable):
        """Two source variables each with their own bounds ref yields two boundary vars."""
        subject_names = ("ref_bounds_1", "ref_bounds_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

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

        expected = {
            name: CFBoundaryVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFBoundaryVariable.identify(vars_all)
        assert expected == result

    def test_duplicate_refs(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
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

        expected = {subject_name: CFBoundaryVariable(subject_name, ref_subject)}
        result = CFBoundaryVariable.identify(vars_all)
        assert expected == result

    def test_ignore(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

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
            expected_name: CFBoundaryVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFBoundaryVariable.identify(vars_all, ignore=subject_names[1])
        assert expected == result

    def test_target(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

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
            expected_name: CFBoundaryVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFBoundaryVariable.identify(vars_all, target=source_names[0])
        assert expected == result

    def test_target_unknown_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFBoundaryVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFBoundaryVariable.identify(vars_all, target=object())

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
            CFBoundaryVariable.identify(vars_all, warn=warn)

        warn_regex = rf"Missing CF-netCDF boundary variable {subject_name!r}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )


class TestSpans:
    """Tests for CFBoundaryVariable.spans()."""

    def _make_cf_var(self, name, dimensions):
        stub = _NetCDFVarWithDimensions(name, dimensions)
        return CFBoundaryVariable(name, stub)

    def test_empty_dimensions_spans(self):
        """Scalar boundary variable always spans the target."""
        cf_boundary = self._make_cf_var("bounds_var", ())
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_boundary.spans(cf_target)

    def test_source_trailing_subset_spans(self):
        """source[:-1] is a subset of target dimensions => spans."""
        # bounds_var has dims (x, y, bounds_extent); data_var has (x, y)
        cf_boundary = self._make_cf_var("bounds_var", ("x", "y", "bounds_extent"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_boundary.spans(cf_target)

    def test_source_leading_subset_spans(self):
        """source[1:] is a subset of target dimensions => spans."""
        # bounds_var has dims (bounds_extent, x, y); data_var has (x, y)
        cf_boundary = self._make_cf_var("bounds_var", ("bounds_extent", "x", "y"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_boundary.spans(cf_target)

    def test_non_spanning(self):
        """Dimensions that don't fit either slice => does not span."""
        cf_boundary = self._make_cf_var("bounds_var", ("x", "b", "c"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert not cf_boundary.spans(cf_target)

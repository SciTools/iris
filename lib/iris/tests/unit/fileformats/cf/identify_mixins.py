# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Shared test mixin classes for CF variable identify() behaviour.

This module provides reusable pytest test classes which concrete test modules
can subclass and configure via class attributes.
"""

from abc import ABC
import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFVariable
import iris.warnings


class _NetCDFVarWithDimensions:
    """Stub with dimensions for spans() tests."""

    def __init__(self, name, dimensions, dtype=np.bytes_):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.dimensions = dimensions

    def ncattrs(self):
        return [
            attr
            for attr in self.__dict__
            if not attr.startswith("_") and attr not in ["name", "dtype", "dimensions"]
        ]


class SpansMixin(ABC):
    """Shared spans() tests for CF variable wrappers."""

    __test__ = False

    CF_CLASS: type[CFVariable]

    def _make_cf_var(self, name, dimensions):
        stub = _NetCDFVarWithDimensions(name, dimensions, dtype=np.bytes_)
        return self.CF_CLASS(name, stub)

    def test_empty_dimensions_spans(self):
        """Scalar source variable always spans the target."""
        cf_source = self._make_cf_var("source_var", ())
        cf_target = self._make_cf_var("target_var", ("x", "y"))
        assert cf_source.spans(cf_target)

    def test_source_trailing_subset_spans(self):
        """source[:-1] is a subset of target dimensions => spans."""
        cf_source = self._make_cf_var("source_var", ("x", "y", "extra"))
        cf_target = self._make_cf_var("target_var", ("x", "y"))
        assert cf_source.spans(cf_target)

    def test_source_leading_subset_spans(self):
        """source[1:] is a subset of target dimensions => spans."""
        cf_source = self._make_cf_var("source_var", ("extra", "x", "y"))
        cf_target = self._make_cf_var("target_var", ("x", "y"))
        assert cf_source.spans(cf_target)

    def test_non_spanning(self):
        """Dimensions that don't fit either slice => does not span."""
        cf_source = self._make_cf_var("source_var", ("x", "b", "c"))
        cf_target = self._make_cf_var("target_var", ("x", "y"))
        assert not cf_source.spans(cf_target)


class IdentifyByAttributeMixin(ABC):
    """Parent class for CF variable identify() tests.

    Supports both single-attribute and multi-attribute identity patterns.
    Subclasses should define CF_IDENTITIES as a list of attribute names.
    Use CF_IDENTITIES[0] as the primary identity in tests.
    """

    __test__ = False

    CF_CLASS: type[CFVariable]
    CF_IDENTITIES: list[str]  # Always a list; use [0] as primary identity
    MISSING_WARN_REGEX: str
    SUBJECT_DTYPE_DEFAULT = None
    # Turn this off for identities that support only a single reference
    IDENTITY_SUPPORTS_MULTIPLE_REFS = True

    @classmethod
    def _make_subject(cls, named_variable, name):
        if cls.SUBJECT_DTYPE_DEFAULT is None:
            return named_variable(name)
        else:
            return named_variable(name, dtype=cls.SUBJECT_DTYPE_DEFAULT)

    # Common test methods (work for both single and multi-identity classes)

    def test_one_ref(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = self._make_subject(named_variable, subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: self.CF_CLASS(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert result == expected

    def test_two_refs(self, named_variable):
        # Check that a single "source" var may refer to multiple "subject" vars.
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        if self.IDENTITY_SUPPORTS_MULTIPLE_REFS:
            # make a single source reference multiple subjects.
            ref_source_name = "ref_source"
            ref_source_var = named_variable(ref_source_name)
            setattr(ref_source_var, self.CF_IDENTITIES[0], " ".join(subject_names))
            ref_source_vars = {ref_source_name: ref_source_var}
        else:
            # make two source vars, each referencing a different subject.
            ref_source_vars = {
                name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
            }
            for ix, var in enumerate(ref_source_vars.values()):
                setattr(var, self.CF_IDENTITIES[0], subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected = {
            name: self.CF_CLASS(name, var) for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert result == expected

    def test_duplicate_refs(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = self._make_subject(named_variable, subject_name)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            setattr(var, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_source_vars,
        }

        expected = {subject_name: self.CF_CLASS(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert result == expected

    def test_ignore(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.CF_IDENTITIES[0], subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: self.CF_CLASS(expected_name, ref_subject_vars[expected_name])
        }
        result = self.CF_CLASS.identify(vars_all, ignore=subject_names[1])
        assert result == expected

    def test_target(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.CF_IDENTITIES[0], subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: self.CF_CLASS(expected_name, ref_subject_vars[expected_name])
        }
        result = self.CF_CLASS.identify(vars_all, target=source_names[0])
        assert result == expected

    def test_target_unknown_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            self.CF_CLASS.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            self.CF_CLASS.identify(vars_all, target=object())

    def test_warn(self, named_variable, assert_warning_gated):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            result = self.CF_CLASS.identify(vars_all, warn=warn)
            # For multi-identity tests with string-typed vars, result may be empty
            # For single-identity tests, result may be non-empty

        assert_warning_gated(
            operation,
            iris.warnings.IrisCfMissingVarWarning,
            self.MISSING_WARN_REGEX.format(subject=subject_name),
        )


class IdentifyByAttributeListMixin(IdentifyByAttributeMixin):
    """Parent class for CF variables identified by multiple possible attributes.

    This class adds test methods specific to cases where CF_IDENTITIES contains
    multiple possible attribute names (typical for UGRID variables).
    Inherits all common tests from IdentifyByAttributeMixin.
    """

    __test__ = False

    def test_cf_identities(self, named_variable):
        """Test that all CF_IDENTITIES attributes are recognized."""
        assert self.CF_IDENTITIES

        for identity in self.CF_IDENTITIES:
            subject_name = "ref_subject"
            ref_subject = named_variable(subject_name)
            vars_common = {
                subject_name: ref_subject,
                "ref_not_subject": named_variable("ref_not_subject"),
            }
            expected = {subject_name: self.CF_CLASS(subject_name, ref_subject)}

            ref_source = named_variable("ref_source")
            setattr(ref_source, identity, subject_name)
            vars_all = dict({"ref_source": ref_source}, **vars_common)
            result = self.CF_CLASS.identify(vars_all)
            assert result == expected

    def test_two_identities(self, named_variable):
        """Test using multiple different CF_IDENTITIES."""
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.CF_IDENTITIES[ix], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        expected = {
            name: self.CF_CLASS(name, var) for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert result == expected

    def test_string_type_ignored(self, named_variable):
        """Test that string-typed referenced variables are ignored."""
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert result == {}

    def test_warn_string_type(self, named_variable, assert_warning_gated):
        """Test warning when string-typed var referenced by identity attribute."""
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            subject_name: named_variable(subject_name, dtype=np.bytes_),
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            result = self.CF_CLASS.identify(vars_all, warn=warn)
            assert result == {}

        warn_regex = r".*is a CF-netCDF label variable.*"
        assert_warning_gated(operation, iris.warnings.IrisCfLabelVarWarning, warn_regex)

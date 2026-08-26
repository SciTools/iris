# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Shared test catalog for CF variable identify() behaviour.

This module provides reusable pytest test classes which concrete test modules
can subclass and configure via class attributes.
"""

from abc import ABC
import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFVariable
import iris.warnings


class IdentifyByAttributeCatalog(ABC):
    """Catalog tests for CF variables identified via a single attribute."""

    __test__ = False

    CF_CLASS: type[CFVariable]
    CF_IDENTITY: str
    MISSING_WARN_REGEX: str

    @classmethod
    def _set_ref(cls, source_var, value):
        setattr(source_var, cls.CF_IDENTITY, value)

    @classmethod
    def _expected_var(cls, name, var):
        return cls.CF_CLASS(name, var)

    @classmethod
    def _make_subject(cls, named_variable, name):
        return named_variable(name)

    def test_one_ref(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = self._make_subject(named_variable, subject_name)
        ref_source = named_variable("ref_source")
        self._set_ref(ref_source, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: self._expected_var(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_two_refs(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            self._set_ref(var, subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected = {
            name: self._expected_var(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_duplicate_refs(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = self._make_subject(named_variable, subject_name)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            self._set_ref(var, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_source_vars,
        }

        expected = {subject_name: self._expected_var(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_ignore(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            self._set_ref(var, subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: self._expected_var(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = self.CF_CLASS.identify(vars_all, ignore=subject_names[1])
        assert expected == result

    def test_target(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: self._make_subject(named_variable, name) for name in subject_names
        }

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            self._set_ref(var, subject_names[ix])
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: self._expected_var(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = self.CF_CLASS.identify(vars_all, target=source_names[0])
        assert expected == result

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
        self._set_ref(ref_source, subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            self.CF_CLASS.identify(vars_all, warn=warn)

        assert_warning_gated(
            operation,
            iris.warnings.IrisCfMissingVarWarning,
            self.MISSING_WARN_REGEX.format(subject=subject_name),
        )


class IdentifyByAttributeListCatalog(ABC):
    """Catalog tests for UGRID variables identified by one of many attributes."""

    __test__ = False

    CF_CLASS: type[CFVariable]
    CF_IDENTITIES: list[str]
    MISSING_WARN_REGEX: str

    @classmethod
    def _set_ref(cls, source_var, identity, value):
        setattr(source_var, identity, value)

    @classmethod
    def _expected_var(cls, name, var):
        return cls.CF_CLASS(name, var)

    def test_cf_identities(self, named_variable):
        assert self.CF_IDENTITIES

        for identity in self.CF_IDENTITIES:
            subject_name = "ref_subject"
            ref_subject = named_variable(subject_name)
            vars_common = {
                subject_name: ref_subject,
                "ref_not_subject": named_variable("ref_not_subject"),
            }
            expected = {subject_name: self._expected_var(subject_name, ref_subject)}

            ref_source = named_variable("ref_source")
            self._set_ref(ref_source, identity, subject_name)
            vars_all = dict({"ref_source": ref_source}, **vars_common)
            result = self.CF_CLASS.identify(vars_all)
            assert expected == result

    def test_duplicate_refs(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            self._set_ref(var, self.CF_IDENTITIES[0], subject_name)
        vars_all = dict(
            {
                subject_name: ref_subject,
                "ref_not_subject": named_variable("ref_not_subject"),
            },
            **ref_source_vars,
        )

        expected = {subject_name: self._expected_var(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_two_identities(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            self._set_ref(var, self.CF_IDENTITIES[ix], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        expected = {
            name: self._expected_var(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_two_part_ref(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source = named_variable("ref_source")
        self._set_ref(ref_source, self.CF_IDENTITIES[0], " ".join(subject_names))
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert {} == result

    def test_string_type_ignored(self, named_variable):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        self._set_ref(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert {} == result

    def test_ignore(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            self._set_ref(var, self.CF_IDENTITIES[0], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        expected_name = subject_names[0]
        expected = {
            expected_name: self._expected_var(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = self.CF_CLASS.identify(vars_all, ignore=subject_names[1])
        assert expected == result

    def test_target(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            self._set_ref(var, self.CF_IDENTITIES[0], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        expected_name = subject_names[0]
        expected = {
            expected_name: self._expected_var(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = self.CF_CLASS.identify(vars_all, target=source_names[0])
        assert expected == result

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
        self._set_ref(ref_source, self.CF_IDENTITIES[0], subject_name)
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
            assert {} == result

        assert_warning_gated(
            operation,
            iris.warnings.IrisCfMissingVarWarning,
            self.MISSING_WARN_REGEX.format(subject=subject_name),
        )

    def test_warn_string_type(self, named_variable, assert_warning_gated):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        self._set_ref(ref_source, self.CF_IDENTITIES[0], subject_name)
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
            assert {} == result

        warn_regex = r".*is a CF-netCDF label variable.*"
        assert_warning_gated(operation, iris.warnings.IrisCfLabelVarWarning, warn_regex)

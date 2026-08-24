# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFAncillaryDataVariable`."""

import warnings

import pytest

from iris.fileformats.cf import CFAncillaryDataVariable
import iris.warnings

CF_IDENTITY = "ancillary_variables"


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

        expected = {subject_name: CFAncillaryDataVariable(subject_name, ref_subject)}
        result = CFAncillaryDataVariable.identify(vars_all)
        assert expected == result

    def test_two_refs(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, " ".join(subject_names))
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        expected = {
            name: CFAncillaryDataVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFAncillaryDataVariable.identify(vars_all)
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

        expected = {subject_name: CFAncillaryDataVariable(subject_name, ref_subject)}
        result = CFAncillaryDataVariable.identify(vars_all)
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
            expected_name: CFAncillaryDataVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFAncillaryDataVariable.identify(vars_all, ignore=subject_names[1])
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
            expected_name: CFAncillaryDataVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFAncillaryDataVariable.identify(vars_all, target=source_names[0])
        assert expected == result

    def test_target_unknown_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFAncillaryDataVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFAncillaryDataVariable.identify(vars_all, target=object())

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
            CFAncillaryDataVariable.identify(vars_all, warn=warn)

        warn_regex = rf"Missing CF-netCDF ancillary data variable {subject_name!r}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )

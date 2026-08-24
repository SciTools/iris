# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridConnectivityVariable`."""

import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFUGridConnectivityVariable
from iris.mesh import Connectivity
import iris.warnings


class TestIdentify:
    @pytest.mark.parametrize(
        "identity",
        Connectivity.UGRID_CF_ROLES,
        ids=Connectivity.UGRID_CF_ROLES,
    )
    def test_cf_identities(self, named_variable, identity):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        vars_common = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
        }
        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {
            subject_name: CFUGridConnectivityVariable(subject_name, ref_subject)
        }

        ref_source = named_variable("ref_source")
        setattr(ref_source, identity, subject_name)
        vars_all = dict({"ref_source": ref_source}, **vars_common)
        result = CFUGridConnectivityVariable.identify(vars_all)
        assert expected == result

    def test_duplicate_refs(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            setattr(var, Connectivity.UGRID_CF_ROLES[0], subject_name)
        vars_all = dict(
            {
                subject_name: ref_subject,
                "ref_not_subject": named_variable("ref_not_subject"),
            },
            **ref_source_vars,
        )

        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {
            subject_name: CFUGridConnectivityVariable(subject_name, ref_subject)
        }
        result = CFUGridConnectivityVariable.identify(vars_all)
        assert expected == result

    def test_two_cf_roles(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, Connectivity.UGRID_CF_ROLES[ix], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # Not expecting ref_not_subject.
        expected = {
            name: CFUGridConnectivityVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFUGridConnectivityVariable.identify(vars_all)
        assert expected == result

    def test_two_part_ref_ignored(self, named_variable):
        # Not expected to handle more than one variable for a connectivity
        # cf role - invalid UGRID.
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, Connectivity.UGRID_CF_ROLES[0], subject_name + " foo")
        vars_all = {
            subject_name: named_variable(subject_name),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridConnectivityVariable.identify(vars_all)
        assert {} == result

    def test_string_type_ignored(self, named_variable):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, Connectivity.UGRID_CF_ROLES[0], subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridConnectivityVariable.identify(vars_all)
        assert {} == result

    def test_ignore(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, Connectivity.UGRID_CF_ROLES[0], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # ONLY expect the subject variable that hasn't been ignored.
        expected_name = subject_names[0]
        expected = {
            expected_name: CFUGridConnectivityVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFUGridConnectivityVariable.identify(vars_all, ignore=subject_names[1])
        assert expected == result

    def test_target(self, named_variable):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, Connectivity.UGRID_CF_ROLES[0], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # ONLY expect the variable referenced by the named ref_source_var.
        expected_name = subject_names[0]
        expected = {
            expected_name: CFUGridConnectivityVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFUGridConnectivityVariable.identify(vars_all, target=source_names[0])
        assert expected == result

    def test_target_unknown_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFUGridConnectivityVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFUGridConnectivityVariable.identify(vars_all, target=object())

    def test_warn(self, named_variable, assert_warning_gated):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, Connectivity.UGRID_CF_ROLES[0], subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            result = CFUGridConnectivityVariable.identify(vars_all, warn=warn)
            assert {} == result

        # Missing warning.
        warn_regex = rf"Missing CF-UGRID connectivity variable {subject_name}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )

        # String variable warning.
        warn_regex = r".*is a CF-netCDF label variable.*"
        vars_all[subject_name] = named_variable(subject_name, dtype=np.bytes_)
        assert_warning_gated(operation, iris.warnings.IrisCfLabelVarWarning, warn_regex)

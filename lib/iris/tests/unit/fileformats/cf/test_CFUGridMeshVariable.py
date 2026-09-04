# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridMeshVariable`."""

import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFUGridMeshVariable
import iris.warnings

from .identify_mixins import _NetCDFVar, assert_warning_gated

CF_IDENTITY = "mesh"


class TestIdentify:
    def test_cf_role(self):
        # Test that mesh variables can be identified by having `cf_role="mesh_topology"`.
        match_name = "match"
        match = _NetCDFVar(match_name)
        setattr(match, "cf_role", "mesh_topology")

        not_match_name = f"not_{match_name}"
        not_match = _NetCDFVar(not_match_name)
        setattr(not_match, "cf_role", "foo")

        vars_all = {match_name: match, not_match_name: not_match}

        # ONLY expecting match, excluding not_match.
        expected = {match_name: CFUGridMeshVariable(match_name, match)}
        result = CFUGridMeshVariable.identify(vars_all)
        assert result == expected

    def test_cf_identity(self):
        # Test that mesh variables can be identified by being another variable's
        #  `mesh` attribute.
        subject_name = "ref_subject"
        ref_subject = _NetCDFVar(subject_name)
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {subject_name: CFUGridMeshVariable(subject_name, ref_subject)}
        result = CFUGridMeshVariable.identify(vars_all)
        assert result == expected

    def test_cf_role_and_identity(self):
        # Test that identification can successfully handle a combination of
        #  mesh variables having `cf_role="mesh_topology"` AND being referenced as
        #  another variable's `mesh` attribute.
        role_match_name = "match"
        role_match = _NetCDFVar(role_match_name)
        setattr(role_match, "cf_role", "mesh_topology")
        ref_source_1 = _NetCDFVar("ref_source_1")
        setattr(ref_source_1, CF_IDENTITY, role_match_name)

        subject_name = "ref_subject"
        ref_subject = _NetCDFVar(subject_name)
        ref_source_2 = _NetCDFVar("ref_source_2")
        setattr(ref_source_2, CF_IDENTITY, subject_name)

        vars_all = {
            role_match_name: role_match,
            subject_name: ref_subject,
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source_1": ref_source_1,
            "ref_source_2": ref_source_2,
        }

        # Expecting role_match and ref_subject but excluding other variables.
        expected = {
            role_match_name: CFUGridMeshVariable(role_match_name, role_match),
            subject_name: CFUGridMeshVariable(subject_name, ref_subject),
        }
        result = CFUGridMeshVariable.identify(vars_all)
        assert result == expected

    def test_duplicate_refs(self):
        subject_name = "ref_subject"
        ref_subject = _NetCDFVar(subject_name)
        ref_source_vars = {
            name: _NetCDFVar(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            setattr(var, CF_IDENTITY, subject_name)
        vars_all = dict(
            {
                subject_name: ref_subject,
                "ref_not_subject": _NetCDFVar("ref_not_subject"),
            },
            **ref_source_vars,
        )

        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {subject_name: CFUGridMeshVariable(subject_name, ref_subject)}
        result = CFUGridMeshVariable.identify(vars_all)
        assert result == expected

    def test_two_refs(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        ref_source_vars = {
            name: _NetCDFVar(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": _NetCDFVar("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # Not expecting ref_not_subject.
        expected = {
            name: CFUGridMeshVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFUGridMeshVariable.identify(vars_all)
        assert result == expected

    def test_two_part_ref_ignored(self):
        # Not expected to handle more than one variable for a mesh
        # cf role - invalid UGRID.
        subject_name = "ref_subject"
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name + " foo")
        vars_all = {
            subject_name: _NetCDFVar(subject_name),
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridMeshVariable.identify(vars_all)
        assert result == {}

    def test_string_type_ignored(self):
        subject_name = "ref_subject"
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: _NetCDFVar(subject_name, dtype=np.bytes_),
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridMeshVariable.identify(vars_all)
        assert result == {}

    def test_ignore(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        ref_source_vars = {
            name: _NetCDFVar(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": _NetCDFVar("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # ONLY expect the subject variable that hasn't been ignored.
        expected_name = subject_names[0]
        expected = {
            expected_name: CFUGridMeshVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFUGridMeshVariable.identify(vars_all, ignore=subject_names[1])
        assert result == expected

    def test_target(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: _NetCDFVar(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": _NetCDFVar("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # ONLY expect the variable referenced by the named ref_source_var.
        expected_name = subject_names[0]
        expected = {
            expected_name: CFUGridMeshVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFUGridMeshVariable.identify(vars_all, target=source_names[0])
        assert result == expected

    def test_target_unknown_raises(self):
        vars_all = {"ref_source": _NetCDFVar("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFUGridMeshVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self):
        vars_all = {"ref_source": _NetCDFVar("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFUGridMeshVariable.identify(vars_all, target=object())

    def test_warn(self):
        subject_name = "ref_subject"
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, subject_name)
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            result = CFUGridMeshVariable.identify(vars_all, warn=warn)
            assert result == {}

        # Missing warning.
        warn_regex = rf"Missing CF-UGRID mesh variable {subject_name}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )

        # String variable warning.
        warn_regex = r".*is a CF-netCDF label variable.*"
        vars_all[subject_name] = _NetCDFVar(subject_name, dtype=np.bytes_)
        assert_warning_gated(operation, iris.warnings.IrisCfLabelVarWarning, warn_regex)

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridMeshVariable`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import re
import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFUGridMeshVariable
from iris.tests.unit.fileformats.cf.test_CFReader import netcdf_variable
import iris.warnings


def named_variable(name):
    # Don't need to worry about dimensions or dtype for these tests.
    return netcdf_variable(name, "", int)


class TestIdentify(tests.IrisTest):
    def setUp(self):
        self.cf_identity = "mesh"

    def test_cf_role(self):
        # Test that mesh variables can be identified by having `cf_role="mesh_topology"`.
        match_name = "match"
        match = named_variable(match_name)
        setattr(match, "cf_role", "mesh_topology")

        not_match_name = f"not_{match_name}"
        not_match = named_variable(not_match_name)
        setattr(not_match, "cf_role", "foo")

        vars_all = {match_name: match, not_match_name: not_match}

        # ONLY expecting match, excluding not_match.
        expected = {match_name: CFUGridMeshVariable(match_name, match)}
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_cf_identity(self):
        # Test that mesh variables can be identified by being another variable's
        #  `mesh` attribute.
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identity, subject_name)
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {subject_name: CFUGridMeshVariable(subject_name, ref_subject)}
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_cf_role_and_identity(self):
        # Test that identification can successfully handle a combination of
        #  mesh variables having `cf_role="mesh_topology"` AND being referenced as
        #  another variable's `mesh` attribute.
        role_match_name = "match"
        role_match = named_variable(role_match_name)
        setattr(role_match, "cf_role", "mesh_topology")
        ref_source_1 = named_variable("ref_source_1")
        setattr(ref_source_1, self.cf_identity, role_match_name)

        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_2 = named_variable("ref_source_2")
        setattr(ref_source_2, self.cf_identity, subject_name)

        vars_all = {
            role_match_name: role_match,
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source_1": ref_source_1,
            "ref_source_2": ref_source_2,
        }

        # Expecting role_match and ref_subject but excluding other variables.
        expected = {
            role_match_name: CFUGridMeshVariable(role_match_name, role_match),
            subject_name: CFUGridMeshVariable(subject_name, ref_subject),
        }
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_duplicate_refs(self):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            setattr(var, self.cf_identity, subject_name)
        vars_all = dict(
            {
                subject_name: ref_subject,
                "ref_not_subject": named_variable("ref_not_subject"),
            },
            **ref_source_vars,
        )

        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {subject_name: CFUGridMeshVariable(subject_name, ref_subject)}
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_two_refs(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.cf_identity, subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # Not expecting ref_not_subject.
        expected = {
            name: CFUGridMeshVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_two_part_ref_ignored(self):
        # Not expected to handle more than one variable for a mesh
        # cf role - invalid UGRID.
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identity, subject_name + " foo")
        vars_all = {
            subject_name: named_variable(subject_name),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual({}, result)

    def test_string_type_ignored(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identity, subject_name)
        vars_all = {
            subject_name: netcdf_variable(subject_name, "", np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual({}, result)

    def test_ignore(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.cf_identity, subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
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
        self.assertDictEqual(expected, result)

    def test_target(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.cf_identity, subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
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
        self.assertDictEqual(expected, result)

    def test_warn(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identity, subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            result = CFUGridMeshVariable.identify(vars_all, warn=warn)
            self.assertDictEqual({}, result)

        # Missing warning.
        warn_regex = rf"Missing CF-UGRID mesh variable {subject_name}.*"
        with pytest.warns(iris.warnings.IrisCfMissingVarWarning, match=warn_regex):
            operation(warn=True)
        with pytest.warns() as record:
            operation(warn=False)
        warn_list = [str(w.message) for w in record]
        assert list(filter(re.compile(warn_regex).match, warn_list)) == []

        # String variable warning.
        warn_regex = r".*is a CF-netCDF label variable.*"
        vars_all[subject_name] = netcdf_variable(subject_name, "", np.bytes_)
        with pytest.warns(iris.warnings.IrisCfLabelVarWarning, match=warn_regex):
            operation(warn=True)
        with pytest.warns() as record:
            operation(warn=False)
        warn_list = [str(w.message) for w in record]
        assert list(filter(re.compile(warn_regex).match, warn_list)) == []

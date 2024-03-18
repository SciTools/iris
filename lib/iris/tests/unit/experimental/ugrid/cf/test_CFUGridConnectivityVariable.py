# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.experimental.ugrid.cf.CFUGridConnectivityVariable` class.

todo: fold these tests into cf tests when experimental.ugrid is folded into
 standard behaviour.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import re
import warnings

import numpy as np
import pytest

from iris.experimental.ugrid.cf import CFUGridConnectivityVariable
from iris.experimental.ugrid.mesh import Connectivity
from iris.tests.unit.experimental.ugrid.cf.test_CFUGridReader import (
    netcdf_ugrid_variable,
)
import iris.warnings


def named_variable(name):
    # Don't need to worry about dimensions or dtype for these tests.
    return netcdf_ugrid_variable(name, "", int)


class TestIdentify(tests.IrisTest):
    def test_cf_identities(self):
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

        for identity in Connectivity.UGRID_CF_ROLES:
            ref_source = named_variable("ref_source")
            setattr(ref_source, identity, subject_name)
            vars_all = dict({"ref_source": ref_source}, **vars_common)
            result = CFUGridConnectivityVariable.identify(vars_all)
            self.assertDictEqual(expected, result)

    def test_duplicate_refs(self):
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
        self.assertDictEqual(expected, result)

    def test_two_cf_roles(self):
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
        self.assertDictEqual(expected, result)

    def test_two_part_ref_ignored(self):
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
        self.assertDictEqual({}, result)

    def test_string_type_ignored(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, Connectivity.UGRID_CF_ROLES[0], subject_name)
        vars_all = {
            subject_name: netcdf_ugrid_variable(subject_name, "", np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridConnectivityVariable.identify(vars_all)
        self.assertDictEqual({}, result)

    def test_ignore(self):
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
        self.assertDictEqual(expected, result)

    def test_target(self):
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
        self.assertDictEqual(expected, result)

    def test_warn(self):
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
            self.assertDictEqual({}, result)

        # Missing warning.
        warn_regex = rf"Missing CF-UGRID connectivity variable {subject_name}.*"
        with pytest.warns(iris.warnings.IrisCfMissingVarWarning, match=warn_regex):
            operation(warn=True)
        with pytest.warns() as record:
            operation(warn=False)
        warn_list = [str(w.message) for w in record]
        assert list(filter(re.compile(warn_regex).match, warn_list)) == []

        # String variable warning.
        warn_regex = r".*is a CF-netCDF label variable.*"
        vars_all[subject_name] = netcdf_ugrid_variable(subject_name, "", np.bytes_)
        with pytest.warns(iris.warnings.IrisCfLabelVarWarning, match=warn_regex):
            operation(warn=True)
        with pytest.warns() as record:
            operation(warn=False)
        warn_list = [str(w.message) for w in record]
        assert list(filter(re.compile(warn_regex).match, warn_list)) == []

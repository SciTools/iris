# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridAuxiliaryCoordinateVariable`."""

import re
import warnings

import numpy as np
import pytest

from iris.fileformats.cf import CFUGridAuxiliaryCoordinateVariable
from iris.tests.unit.fileformats.cf.test_CFReader import netcdf_variable
import iris.warnings


def named_variable(name):
    # Don't need to worry about dimensions or dtype for these tests.
    return netcdf_variable(name, "", int)


class TestIdentify:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cf_identities = [
            "node_coordinates",
            "edge_coordinates",
            "face_coordinates",
            "volume_coordinates",
        ]

    def test_cf_identities(self):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        vars_common = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
        }
        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {
            subject_name: CFUGridAuxiliaryCoordinateVariable(subject_name, ref_subject)
        }

        for identity in self.cf_identities:
            ref_source = named_variable("ref_source")
            setattr(ref_source, identity, subject_name)
            vars_all = dict({"ref_source": ref_source}, **vars_common)
            result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
            assert expected == result

    def test_duplicate_refs(self):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for var in ref_source_vars.values():
            setattr(var, self.cf_identities[0], subject_name)
        vars_all = dict(
            {
                subject_name: ref_subject,
                "ref_not_subject": named_variable("ref_not_subject"),
            },
            **ref_source_vars,
        )

        # ONLY expecting ref_subject, excluding ref_not_subject.
        expected = {
            subject_name: CFUGridAuxiliaryCoordinateVariable(subject_name, ref_subject)
        }
        result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
        assert expected == result

    def test_two_coords(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.cf_identities[ix], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # Not expecting ref_not_subject.
        expected = {
            name: CFUGridAuxiliaryCoordinateVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
        assert expected == result

    def test_two_part_ref(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identities[0], " ".join(subject_names))
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        expected = {
            name: CFUGridAuxiliaryCoordinateVariable(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
        assert expected == result

    def test_string_type_ignored(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identities[0], subject_name)
        vars_all = {
            subject_name: netcdf_variable(subject_name, "", np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
        assert {} == result

    def test_ignore(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.cf_identities[0], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # ONLY expect the subject variable that hasn't been ignored.
        expected_name = subject_names[0]
        expected = {
            expected_name: CFUGridAuxiliaryCoordinateVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFUGridAuxiliaryCoordinateVariable.identify(
            vars_all, ignore=subject_names[1]
        )
        assert expected == result

    def test_target(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, self.cf_identities[0], subject_names[ix])
        vars_all = dict(
            {"ref_not_subject": named_variable("ref_not_subject")},
            **ref_subject_vars,
            **ref_source_vars,
        )

        # ONLY expect the variable referenced by the named ref_source_var.
        expected_name = subject_names[0]
        expected = {
            expected_name: CFUGridAuxiliaryCoordinateVariable(
                expected_name, ref_subject_vars[expected_name]
            )
        }
        result = CFUGridAuxiliaryCoordinateVariable.identify(
            vars_all, target=source_names[0]
        )
        assert expected == result

    def test_warn(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identities[0], subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all, warn=warn)
            assert {} == result

        # Missing warning.
        warn_regex = (
            rf"Missing CF-netCDF auxiliary coordinate variable {subject_name}.*"
        )
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

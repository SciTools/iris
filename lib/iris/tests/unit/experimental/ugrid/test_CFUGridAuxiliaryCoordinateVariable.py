# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.CFUGridAuxiliaryCoordinateVariable` class.

todo: fold these tests into cf tests when experimental.ugrid is folded into
 standard behaviour.

"""
import numpy as np

from iris.experimental.ugrid import CFUGridAuxiliaryCoordinateVariable, logger

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from iris.tests.unit.experimental.ugrid.test_CFUGridReader import (
    netcdf_ugrid_variable,
)


def named_variable(name):
    # Don't need to worry about dimensions or dtype for these tests.
    return netcdf_ugrid_variable(name, "", np.int)


class TestIdentify(tests.IrisTest):
    def setUp(self):
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
            subject_name: CFUGridAuxiliaryCoordinateVariable(
                subject_name, ref_subject
            )
        }

        for identity in self.cf_identities:
            ref_source = named_variable("ref_source")
            setattr(ref_source, identity, subject_name)
            vars_all = dict({"ref_source": ref_source}, **vars_common)
            result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
            self.assertDictEqual(expected, result)

    def test_duplicate_refs(self):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_vars = {
            name: named_variable(name)
            for name in ("ref_source_1", "ref_source_2")
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
            subject_name: CFUGridAuxiliaryCoordinateVariable(
                subject_name, ref_subject
            )
        }
        result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_two_coords(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name)
            for name in ("ref_source_1", "ref_source_2")
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
        self.assertDictEqual(expected, result)

    def test_two_part_ref(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name) for name in subject_names
        }

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
        self.assertDictEqual(expected, result)

    def test_string_type_ignored(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identities[0], subject_name)
        vars_all = {
            subject_name: netcdf_ugrid_variable(subject_name, "", np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridAuxiliaryCoordinateVariable.identify(vars_all)
        self.assertDictEqual({}, result)

    def test_ignore(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name)
            for name in ("ref_source_1", "ref_source_2")
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
        self.assertDictEqual(expected, result)

    def test_target(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name) for name in subject_names
        }

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
        self.assertDictEqual(expected, result)

    def test_warn(self):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.cf_identities[0], subject_name)
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        # The warn kwarg and expected corresponding log level.
        warn_and_level = {True: "WARNING", False: "DEBUG"}

        # Missing warning.
        log_regex = rf"Missing CF-netCDF auxiliary coordinate variable {subject_name}.*"
        for warn, level in warn_and_level.items():
            with self.assertLogs(logger, level=level, msg_regex=log_regex):
                result = CFUGridAuxiliaryCoordinateVariable.identify(
                    vars_all, warn=warn
                )
                self.assertDictEqual({}, result)

        # String variable warning.
        log_regex = r".*is a CF-netCDF label variable.*"
        for warn, level in warn_and_level.items():
            with self.assertLogs(logger, level=level, msg_regex=log_regex):
                vars_all[subject_name] = netcdf_ugrid_variable(
                    subject_name, "", np.bytes_
                )
                result = CFUGridAuxiliaryCoordinateVariable.identify(
                    vars_all, warn=warn
                )
                self.assertDictEqual({}, result)

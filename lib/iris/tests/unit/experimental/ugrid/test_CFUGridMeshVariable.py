# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.CFUGridMeshVariable` class.

todo: fold these tests into cf tests when experimental.ugrid is folded into
 standard behaviour.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.experimental.ugrid import (
    CFUGridMeshVariable,
    logger,
)
from iris.tests.unit.experimental.ugrid.test_CFUGridReader import (
    netcdf_ugrid_variable,
)


def named_variable(name):
    # Don't need to worry about dimensions or dtype for these tests.
    return netcdf_ugrid_variable(name, "", np.int)


class TestIdentify(tests.IrisTest):
    def setUp(self):
        self.cf_identity = "mesh"

    def test_cf_identity(self):
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
        expected = {
            subject_name: CFUGridMeshVariable(subject_name, ref_subject)
        }
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_duplicate_refs(self):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source_vars = {
            name: named_variable(name)
            for name in ("ref_source_1", "ref_source_2")
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
        expected = {
            subject_name: CFUGridMeshVariable(subject_name, ref_subject)
        }
        result = CFUGridMeshVariable.identify(vars_all)
        self.assertDictEqual(expected, result)

    def test_two_refs(self):
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {
            name: named_variable(name) for name in subject_names
        }

        ref_source_vars = {
            name: named_variable(name)
            for name in ("ref_source_1", "ref_source_2")
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
            subject_name: netcdf_ugrid_variable(subject_name, "", np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = CFUGridMeshVariable.identify(vars_all)
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
        result = CFUGridMeshVariable.identify(
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

        # Missing warning.
        with self.assertLogs(logger) as log:
            result = CFUGridMeshVariable.identify(vars_all, warn=False)
            self.assertEqual(0, len(log.records))
            self.assertDictEqual({}, result)

            # Default is warn=True
            result = CFUGridMeshVariable.identify(vars_all)
            rec = log.records[0]
            self.assertEqual("WARNING", rec.levelname)
            re_msg = rf"Missing CF-UGRID mesh variable {subject_name}.*"
            self.assertRegex(rec.msg, re_msg)
            self.assertDictEqual({}, result)

        # String variable warning.
        with self.assertLogs(logger, level="DEBUG") as log:
            vars_all[subject_name] = netcdf_ugrid_variable(
                subject_name, "", np.bytes_
            )
            result = CFUGridMeshVariable.identify(vars_all, warn=False)
            self.assertDictEqual({}, result)

            # Default is warn=True
            result = CFUGridMeshVariable.identify(vars_all)
            self.assertIn("is a CF-netCDF label variable", log.output[0])
            self.assertDictEqual({}, result)

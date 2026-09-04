# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFMeasureVariable`."""

import warnings

import pytest

from iris.fileformats.cf import CFMeasureVariable
import iris.warnings

CF_IDENTITY = "cell_measures"


class TestIdentify:
    def test_one_measure_ref(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, f"area: {subject_name}")
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: CFMeasureVariable(subject_name, ref_subject, "area")}
        result = CFMeasureVariable.identify(vars_all)
        assert result == expected

    def test_measure_stored_on_instance(self, named_variable):
        subject_name = "ref_subject"
        ref_subject = named_variable(subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, f"volume: {subject_name}")
        vars_all = {
            subject_name: ref_subject,
            "ref_source": ref_source,
        }

        result = CFMeasureVariable.identify(vars_all)
        assert result[subject_name].cf_measure == "volume"

    def test_multi_term(self, named_variable):
        subject_names = ("ref_area", "ref_volume")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}
        ref_source = named_variable("ref_source")
        setattr(
            ref_source,
            CF_IDENTITY,
            f"area: {subject_names[0]} volume: {subject_names[1]}",
        )
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        result = CFMeasureVariable.identify(vars_all)
        assert set(result.keys()) == set(subject_names)
        assert result[subject_names[0]].cf_measure == "area"
        assert result[subject_names[1]].cf_measure == "volume"

    def test_two_refs(self, named_variable):
        """Two source variables each referencing a different measure variable."""
        subject_names = ("ref_area", "ref_volume")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, f"area: {subject_names[ix]}")
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected = {
            name: CFMeasureVariable(name, var, "area")
            for name, var in ref_subject_vars.items()
        }
        result = CFMeasureVariable.identify(vars_all)
        assert result == expected

    def test_self_reference_ignored(self, named_variable):
        """A variable cannot reference itself as a cell measure."""
        nc_var = named_variable("self_ref")
        setattr(nc_var, CF_IDENTITY, "area: self_ref")
        vars_all = {
            "self_ref": nc_var,
        }

        result = CFMeasureVariable.identify(vars_all)
        assert result == {}

    def test_ignore(self, named_variable):
        subject_names = ("ref_area", "ref_volume")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source_vars = {
            name: named_variable(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, f"area: {subject_names[ix]}")
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: CFMeasureVariable(
                expected_name, ref_subject_vars[expected_name], "area"
            )
        }
        result = CFMeasureVariable.identify(vars_all, ignore=subject_names[1])
        assert result == expected

    def test_target(self, named_variable):
        subject_names = ("ref_area", "ref_volume")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: named_variable(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, f"area: {subject_names[ix]}")
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        expected_name = subject_names[0]
        expected = {
            expected_name: CFMeasureVariable(
                expected_name, ref_subject_vars[expected_name], "area"
            )
        }
        result = CFMeasureVariable.identify(vars_all, target=source_names[0])
        assert result == expected

    def test_target_unknown_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFMeasureVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self, named_variable):
        vars_all = {"ref_source": named_variable("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFMeasureVariable.identify(vars_all, target=object())

    def test_warn(self, named_variable, assert_warning_gated):
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, CF_IDENTITY, f"area: {subject_name}")
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            CFMeasureVariable.identify(vars_all, warn=warn)

        warn_regex = rf"Missing CF-netCDF measure variable {subject_name!r}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )

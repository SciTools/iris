# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf._CFFormulaTermsVariable`."""

import warnings

import pytest

from iris.fileformats.cf import _CFFormulaTermsVariable
import iris.warnings

from .identify_mixins import _NetCDFVar, assert_warning_gated

CF_IDENTITY = "formula_terms"


class TestIdentify:
    def test_single_formula_term(self):
        subject_name = "ref_sigma"
        ref_subject = _NetCDFVar(subject_name)
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, f"sigma: {subject_name}")
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = _CFFormulaTermsVariable.identify(vars_all)
        assert subject_name in result
        assert result[subject_name].cf_terms_by_root == {"ref_source": "sigma"}

    def test_multiple_terms_one_source(self):
        subject_names = ("ref_sigma", "ref_ps")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}
        ref_source = _NetCDFVar("ref_source")
        setattr(
            ref_source,
            CF_IDENTITY,
            f"sigma: {subject_names[0]} ps: {subject_names[1]}",
        )
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        result = _CFFormulaTermsVariable.identify(vars_all)
        assert set(result.keys()) == set(subject_names)
        assert result[subject_names[0]].cf_terms_by_root == {"ref_source": "sigma"}
        assert result[subject_names[1]].cf_terms_by_root == {"ref_source": "ps"}

    def test_term_name_lowercased(self):
        """Formula term names must be normalised to lowercase."""
        subject_name = "ref_sigma"
        ref_subject = _NetCDFVar(subject_name)
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, f"SIGMA: {subject_name}")
        vars_all = {
            subject_name: ref_subject,
            "ref_source": ref_source,
        }

        result = _CFFormulaTermsVariable.identify(vars_all)
        assert result[subject_name].cf_terms_by_root == {"ref_source": "sigma"}

    def test_same_variable_multiple_roots_aggregates(self):
        """Same variable referenced by two roots accumulates both terms."""
        subject_name = "ref_sigma"
        ref_subject = _NetCDFVar(subject_name)

        source_vars = {
            name: _NetCDFVar(name) for name in ("ref_source_1", "ref_source_2")
        }
        source_vars["ref_source_1"].formula_terms = f"sigma: {subject_name}"
        setattr(source_vars["ref_source_1"], CF_IDENTITY, f"sigma: {subject_name}")
        setattr(source_vars["ref_source_2"], CF_IDENTITY, f"eta: {subject_name}")

        vars_all = {
            subject_name: ref_subject,
            **source_vars,
        }

        result = _CFFormulaTermsVariable.identify(vars_all)
        assert subject_name in result
        terms = result[subject_name].cf_terms_by_root
        assert terms.get("ref_source_1") == "sigma"
        assert terms.get("ref_source_2") == "eta"

    def test_ignore(self):
        subject_names = ("ref_sigma", "ref_ps")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        ref_source_vars = {
            name: _NetCDFVar(name) for name in ("ref_source_1", "ref_source_2")
        }
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, f"sigma: {subject_names[ix]}")
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        result = _CFFormulaTermsVariable.identify(vars_all, ignore=subject_names[1])
        assert subject_names[0] in result
        assert subject_names[1] not in result

    def test_target(self):
        subject_names = ("ref_sigma", "ref_ps")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        source_names = ("ref_source_1", "ref_source_2")
        ref_source_vars = {name: _NetCDFVar(name) for name in source_names}
        for ix, var in enumerate(ref_source_vars.values()):
            setattr(var, CF_IDENTITY, f"sigma: {subject_names[ix]}")
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            **ref_subject_vars,
            **ref_source_vars,
        }

        result = _CFFormulaTermsVariable.identify(vars_all, target=source_names[0])
        assert subject_names[0] in result
        assert subject_names[1] not in result

    def test_target_unknown_raises(self):
        vars_all = {"ref_source": _NetCDFVar("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            _CFFormulaTermsVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self):
        vars_all = {"ref_source": _NetCDFVar("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            _CFFormulaTermsVariable.identify(vars_all, target=object())

    def test_warn(self):
        subject_name = "ref_sigma"
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, f"sigma: {subject_name}")
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
        }

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            _CFFormulaTermsVariable.identify(vars_all, warn=warn)

        warn_regex = rf"Missing CF-netCDF formula term variable {subject_name!r}.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )


class TestRepr:
    def test_repr_contains_terms_by_root(self):
        subject_name = "ref_sigma"
        ref_subject = _NetCDFVar(subject_name)
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, f"sigma: {subject_name}")
        vars_all = {
            subject_name: ref_subject,
            "ref_source": ref_source,
        }

        result = _CFFormulaTermsVariable.identify(vars_all)
        repr_str = repr(result[subject_name])
        assert "sigma" in repr_str
        assert "ref_source" in repr_str

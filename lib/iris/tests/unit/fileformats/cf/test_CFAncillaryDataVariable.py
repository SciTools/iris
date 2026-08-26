# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFAncillaryDataVariable`."""

from iris.fileformats.cf import CFAncillaryDataVariable

from .identify_catalogue import IdentifyByAttributeCatalog


class TestIdentify(IdentifyByAttributeCatalog):
    __test__ = True

    CF_CLASS = CFAncillaryDataVariable
    CF_IDENTITY = "ancillary_variables"
    MISSING_WARN_REGEX = r"Missing CF-netCDF ancillary data variable {subject!r}.*"

    def test_two_refs(self, named_variable):
        # Ancillary vars are commonly referenced as a space-delimited list
        # on a single source variable.
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: named_variable(name) for name in subject_names}

        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITY, " ".join(subject_names))
        vars_all = {
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        expected = {
            name: self._expected_var(name, var)
            for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

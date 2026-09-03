# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFClimatologyVariable`."""

from iris.fileformats.cf import CFClimatologyVariable

from .identify_catalogue import (
    IdentifyByAttributeCatalog,
    SpansCatalog,
)


class TestIdentify(IdentifyByAttributeCatalog):
    __test__ = True

    CF_CLASS = CFClimatologyVariable
    CF_IDENTITIES = ["climatology"]
    IDENTITY_SUPPORTS_MULTIPLE_REFS = False
    MISSING_WARN_REGEX = r"Missing CF-netCDF climatology variable {subject!r}.*"

    def test_whitespace_padded_ref(self, named_variable):
        # CF climatology references accept surrounding whitespace.
        subject_name = "ref_subject"
        ref_subject = self._make_subject(named_variable, subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], f"  {subject_name}  ")
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: self.CF_CLASS(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result


class TestSpans(SpansCatalog):
    """Tests for CFClimatologyVariable.spans()."""

    __test__ = True
    CF_CLASS = CFClimatologyVariable

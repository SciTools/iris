# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFAuxiliaryCoordinateVariable`."""

import numpy as np

from iris.fileformats.cf import CFAuxiliaryCoordinateVariable

from .identify_catalogue import IdentifyByAttributeCatalog


class TestIdentify(IdentifyByAttributeCatalog):
    __test__ = True

    CF_CLASS = CFAuxiliaryCoordinateVariable
    CF_IDENTITY = "coordinates"
    MISSING_WARN_REGEX = (
        r"Missing CF-netCDF auxiliary coordinate variable {subject!r}.*"
    )

    def test_two_refs(self, named_variable):
        # Auxiliary coordinates can be listed space-delimited on one source.
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
            name: self.CF_CLASS(name, var) for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result

    def test_string_type_ignored(self, named_variable):
        # Coordinate-variable identify should reject label/string subjects.
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITY, subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert {} == result

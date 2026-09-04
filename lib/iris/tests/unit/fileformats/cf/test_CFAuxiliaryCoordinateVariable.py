# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFAuxiliaryCoordinateVariable`."""

import numpy as np

from iris.fileformats.cf import CFAuxiliaryCoordinateVariable

from .identify_mixins import IdentifyByAttributeMixin


class TestIdentify(IdentifyByAttributeMixin):
    __test__ = True

    CF_CLASS = CFAuxiliaryCoordinateVariable
    CF_IDENTITIES = ["coordinates"]
    MISSING_WARN_REGEX = (
        r"Missing CF-netCDF auxiliary coordinate variable {subject!r}.*"
    )

    def test_string_type_ignored(self, named_variable):
        # Coordinate-variable identify should reject label/string subjects.
        subject_name = "ref_subject"
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], subject_name)
        vars_all = {
            subject_name: named_variable(subject_name, dtype=np.bytes_),
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert result == {}

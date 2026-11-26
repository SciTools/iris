# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFAncillaryDataVariable`."""

from iris.fileformats.cf import CFAncillaryDataVariable

from .identify_mixins import IdentifyByAttributeMixin


class TestIdentify(IdentifyByAttributeMixin):
    __test__ = True

    CF_CLASS = CFAncillaryDataVariable
    CF_IDENTITIES = ["ancillary_variables"]
    MISSING_WARN_REGEX = r"Missing CF-netCDF ancillary data variable {subject!r}.*"

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridAuxiliaryCoordinateVariable`."""

from iris.fileformats.cf import CFUGridAuxiliaryCoordinateVariable

from .identify_catalogue import IdentifyByAttributeListCatalog


class TestIdentify(IdentifyByAttributeListCatalog):
    __test__ = True

    CF_CLASS = CFUGridAuxiliaryCoordinateVariable
    CF_IDENTITIES = [
        "node_coordinates",
        "edge_coordinates",
        "face_coordinates",
        "volume_coordinates",
    ]
    MISSING_WARN_REGEX = r"Missing CF-netCDF auxiliary coordinate variable {subject}.*"

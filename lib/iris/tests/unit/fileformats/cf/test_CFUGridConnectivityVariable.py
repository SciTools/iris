# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridConnectivityVariable`."""

from iris.fileformats.cf import CFUGridConnectivityVariable
from iris.mesh import Connectivity

from .identify_catalogue import IdentifyByAttributeListCatalog


class TestIdentify(IdentifyByAttributeListCatalog):
    __test__ = True

    CF_CLASS = CFUGridConnectivityVariable
    CF_IDENTITIES = Connectivity.UGRID_CF_ROLES
    IDENTITY_SUPPORTS_MULTIPLE_REFS = False
    MISSING_WARN_REGEX = r"Missing CF-UGRID connectivity variable {subject}.*"

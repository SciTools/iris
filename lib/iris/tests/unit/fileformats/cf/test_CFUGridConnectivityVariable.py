# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridConnectivityVariable`."""

from iris.fileformats.cf import CFUGridConnectivityVariable
from iris.mesh import Connectivity

from .identify_mixins import IdentifyByAttributeListMixin, _NetCDFVar


class TestIdentify(IdentifyByAttributeListMixin):
    __test__ = True

    CF_CLASS = CFUGridConnectivityVariable
    CF_IDENTITIES = Connectivity.UGRID_CF_ROLES
    IDENTITY_SUPPORTS_MULTIPLE_REFS = False
    MISSING_WARN_REGEX = r"Missing CF-UGRID connectivity variable {subject}.*"

    def test_two_part_ref_forbidden(self):
        """Test that space-separated refs in a single attribute are rejected."""
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], " ".join(subject_names))
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        result = self.CF_CLASS.identify(vars_all)
        assert result == {}

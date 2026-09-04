# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFUGridAuxiliaryCoordinateVariable`."""

from iris.fileformats.cf import CFUGridAuxiliaryCoordinateVariable

from .identify_mixins import IdentifyByAttributeListMixin, _NetCDFVar


class TestIdentify(IdentifyByAttributeListMixin):
    __test__ = True

    CF_CLASS = CFUGridAuxiliaryCoordinateVariable
    CF_IDENTITIES = [
        "node_coordinates",
        "edge_coordinates",
        "face_coordinates",
        "volume_coordinates",
    ]
    MISSING_WARN_REGEX = r"Missing CF-netCDF auxiliary coordinate variable {subject}.*"

    def test_two_part_ref(self):
        # UGRID auxiliary coordinate attributes can contain multiple refs.
        subject_names = ("ref_subject_1", "ref_subject_2")
        ref_subject_vars = {name: _NetCDFVar(name) for name in subject_names}

        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, self.CF_IDENTITIES[0], " ".join(subject_names))
        vars_all = {
            "ref_not_subject": _NetCDFVar("ref_not_subject"),
            "ref_source": ref_source,
            **ref_subject_vars,
        }

        expected = {
            name: self.CF_CLASS(name, var) for name, var in ref_subject_vars.items()
        }
        result = self.CF_CLASS.identify(vars_all)
        assert result == expected

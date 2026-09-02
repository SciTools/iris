# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFBoundaryVariable`."""

from iris.fileformats.cf import CFBoundaryVariable

from .identify_catalogue import IdentifyByAttributeCatalog


class _NetCDFVarWithDimensions:
    """Stub with a dimensions attribute for spans() tests."""

    def __init__(self, name, dimensions, dtype=int):
        import numpy as np

        self.name = name
        self.dtype = np.dtype(dtype)
        self.dimensions = dimensions

    def ncattrs(self):
        return [
            attr
            for attr in self.__dict__
            if not attr.startswith("_") and attr not in ["name", "dtype", "dimensions"]
        ]


class TestIdentify(IdentifyByAttributeCatalog):
    __test__ = True

    CF_CLASS = CFBoundaryVariable
    CF_IDENTITY = "bounds"
    MISSING_WARN_REGEX = r"Missing CF-netCDF boundary variable {subject!r}.*"

    def test_whitespace_padded_ref(self, named_variable):
        # CF boundary references accept surrounding whitespace.
        subject_name = "ref_subject"
        ref_subject = self._make_subject(named_variable, subject_name)
        ref_source = named_variable("ref_source")
        setattr(ref_source, self.CF_IDENTITY, f"  {subject_name}  ")
        vars_all = {
            subject_name: ref_subject,
            "ref_not_subject": named_variable("ref_not_subject"),
            "ref_source": ref_source,
        }

        expected = {subject_name: self.CF_CLASS(subject_name, ref_subject)}
        result = self.CF_CLASS.identify(vars_all)
        assert expected == result


class TestSpans:
    """Tests for CFBoundaryVariable.spans()."""

    def _make_cf_var(self, name, dimensions):
        stub = _NetCDFVarWithDimensions(name, dimensions)
        return CFBoundaryVariable(name, stub)

    def test_empty_dimensions_spans(self):
        """Scalar boundary variable always spans the target."""
        cf_boundary = self._make_cf_var("bounds_var", ())
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_boundary.spans(cf_target)

    def test_source_trailing_subset_spans(self):
        """source[:-1] is a subset of target dimensions => spans."""
        # bounds_var has dims (x, y, bounds_extent); data_var has (x, y)
        cf_boundary = self._make_cf_var("bounds_var", ("x", "y", "bounds_extent"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_boundary.spans(cf_target)

    def test_source_leading_subset_spans(self):
        """source[1:] is a subset of target dimensions => spans."""
        # bounds_var has dims (bounds_extent, x, y); data_var has (x, y)
        cf_boundary = self._make_cf_var("bounds_var", ("bounds_extent", "x", "y"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert cf_boundary.spans(cf_target)

    def test_non_spanning(self):
        """Dimensions that don't fit either slice => does not span."""
        cf_boundary = self._make_cf_var("bounds_var", ("x", "b", "c"))
        cf_target = self._make_cf_var("data_var", ("x", "y"))
        assert not cf_boundary.spans(cf_target)

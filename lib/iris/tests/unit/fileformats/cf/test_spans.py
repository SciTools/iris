# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""`CFVariable.spans`, asked of both storage backends.

Dimension matching is a pair, `(name, size)`, not a bare name - which is a
widening to admit what Zarr allows, not a change to what netCDF does. These
tests run against both so that the second half of that sentence is checked
rather than asserted.
"""

import numpy as np
import pytest

from iris.fileformats.cf import (
    CFBoundaryVariable,
    CFDataVariable,
    CFLabelVariable,
)

zarr = pytest.importorskip("zarr")

from iris.fileformats.netcdf._dataset import NetCDFDataset  # noqa: E402
from iris.fileformats.zarr._dataset import ZarrDataset  # noqa: E402


def build_netcdf(path, described):
    """Write `described` as a netCDF file and return its open dataset."""
    dataset = NetCDFDataset(path, mode="w")
    sizes = {}
    for dimensions, shape in described.values():
        for name, size in zip(dimensions, shape):
            sizes.setdefault(name, size)
    for name, size in sizes.items():
        dataset.create_dimension(name, size)
    for name, (dimensions, shape) in described.items():
        dataset.create_variable(name, np.dtype("f4"), dimensions)
    return dataset


def build_zarr(path, described):
    """Write `described` as a Zarr store and return its open dataset."""
    group = zarr.open_group(store=path, mode="w", zarr_format=3)
    for name, (dimensions, shape) in described.items():
        group.create_array(name, shape=shape, dtype="f4", dimension_names=dimensions)
    return ZarrDataset(str(path))


@pytest.fixture(params=[build_netcdf, build_zarr], ids=["netcdf", "zarr"])
def build(request, tmp_path):
    """Return a callable giving `{name: CFDataVariable}` for a description."""
    suffix = ".nc" if request.param is build_netcdf else ".zarr"

    def make(described):
        dataset = request.param(tmp_path / f"spans{suffix}", described)
        return {
            name: CFDataVariable(name, dataset.variables[name]) for name in described
        }

    return make


def test_identical_dimensions_span(build):
    variables = build({"data": (("y", "x"), (3, 2)), "aux": (("y", "x"), (3, 2))})
    assert variables["aux"].spans(variables["data"]) is True


def test_a_subset_spans(build):
    variables = build({"data": (("y", "x"), (3, 2)), "aux": (("y",), (3,))})
    assert variables["aux"].spans(variables["data"]) is True


def test_a_different_name_does_not_span(build):
    variables = build({"data": (("y", "x"), (3, 2)), "aux": (("t",), (3,))})
    assert variables["aux"].spans(variables["data"]) is False


def test_a_scalar_always_spans(build):
    variables = build({"data": (("y", "x"), (3, 2)), "grid": ((), ())})
    assert variables["grid"].spans(variables["data"]) is True


def test_a_bounds_variable_ignores_its_extent_dimension(build):
    variables = build({"data": (("y",), (3,)), "bounds": (("y", "bnds"), (3, 2))})
    bounds = CFBoundaryVariable("bounds", variables["bounds"].cf_data)
    assert bounds.spans(variables["data"]) is True


def test_a_label_variable_ignores_its_string_length_dimension(build):
    variables = build({"data": (("y",), (3,)), "label": (("y", "strlen"), (3, 8))})
    label = CFLabelVariable("label", variables["label"].cf_data)
    assert label.spans(variables["data"]) is True


def test_a_name_at_two_sizes_does_not_link(tmp_path):
    """Zarr-only: a name guarantees nothing across arrays.

    The version 3 core specification "does not place any restrictions on the
    use of the same dimension name across multiple arrays within the same Zarr
    hierarchy", so two arrays may mean different things by one name. Matching
    on the name alone would produce a cube whose coordinate is the wrong
    length. netCDF cannot express this: a dimension name has one size per
    group.
    """
    group = zarr.open_group(store=tmp_path / "s.zarr", mode="w", zarr_format=3)
    group.create_array("data", shape=(3,), dtype="f4", dimension_names=("t",))
    group.create_array("aux", shape=(5,), dtype="f4", dimension_names=("t",))

    dataset = ZarrDataset(str(tmp_path / "s.zarr"))
    data = CFDataVariable("data", dataset.variables["data"])
    aux = CFDataVariable("aux", dataset.variables["aux"])
    assert aux.spans(data) is False


def test_an_anonymous_dimension_matches_nothing(tmp_path):
    """Not even another anonymous dimension, of the same size, elsewhere.

    Bare-name matching treats `None == None` as a match: `aux` would span
    `data` under the old rule, for a dimension neither variable actually
    named. There is nothing shared between two differently-sourced "no
    name"s, so claiming a match would be invention.
    """
    group = zarr.open_group(store=tmp_path / "s.zarr", mode="w", zarr_format=3)
    group.create_array("data", shape=(3,), dtype="f4")
    group.create_array("aux", shape=(3,), dtype="f4")

    dataset = ZarrDataset(str(tmp_path / "s.zarr"))
    data = CFDataVariable("data", dataset.variables["data"])
    aux = CFDataVariable("aux", dataset.variables["aux"])
    assert aux.spans(data) is False


def test_two_anonymous_dimensions_of_one_size_do_not_match(tmp_path):
    """Not even each other, and not within one variable.

    There is no information to match on, so claiming a match would be
    invention.
    """
    group = zarr.open_group(store=tmp_path / "s.zarr", mode="w", zarr_format=3)
    group.create_array("data", shape=(3, 3), dtype="f4")
    group.create_array("aux", shape=(3,), dtype="f4")

    dataset = ZarrDataset(str(tmp_path / "s.zarr"))
    data = CFDataVariable("data", dataset.variables["data"])
    aux = CFDataVariable("aux", dataset.variables["aux"])
    assert aux.spans(data) is False


def test_an_anonymous_dimension_is_not_a_coordinate_variable(tmp_path):
    """A one-dimensional array is a coordinate variable by name coincidence.

    `CFCoordinateVariable.identify` requires the variable's own name to be
    among its dimension names, which no anonymous dimension satisfies - so an
    anonymous one-dimensional array is never mistaken for a coordinate
    variable, whatever it is named.
    """
    from iris.fileformats.cf import CFCoordinateVariable

    group = zarr.open_group(store=tmp_path / "s.zarr", mode="w", zarr_format=3)
    group.create_array("t", shape=(3,), dtype="f4")

    dataset = ZarrDataset(str(tmp_path / "s.zarr"))
    assert CFCoordinateVariable.identify(dataset.variables) == {}


def test_a_partly_named_variable_matches_on_the_named_part(tmp_path):
    # "A variable may name some of its dimensions and not others." The named
    # part ("y") is present in the target and would be enough to span under
    # the old rule once the anonymous part is set aside - so `data` carries
    # an anonymous dimension of its own, of the same size as `aux`'s, to show
    # that one does not rescue the match either: bare-name matching would
    # read both `None`s as the same dimension and wrongly call this a span.
    group = zarr.open_group(store=tmp_path / "s.zarr", mode="w", zarr_format=3)
    group.create_array(
        "data", shape=(3, 2, 2), dtype="f4", dimension_names=("y", "x", None)
    )
    group.create_array("aux", shape=(3, 2), dtype="f4", dimension_names=("y", None))

    dataset = ZarrDataset(str(tmp_path / "s.zarr"))
    data = CFDataVariable("data", dataset.variables["data"])
    aux = CFDataVariable("aux", dataset.variables["aux"])
    assert aux.spans(data) is False


def test_a_non_spanning_variable_is_dropped_and_promoted(tmp_path):
    """An anonymous-dimension variable becomes a cube of its own.

    This is what keeps `None` out of `iris.fileformats._nc_load_rules.helpers`,
    which matches dimensions by name and would read `None in (None, "x")` as a
    match. `CFReader._build_cf_groups` drops any variable whose `spans` is
    False from the referrer's group and promotes it.
    """
    from iris.fileformats.cf import CFReader

    group = zarr.open_group(store=tmp_path / "s.zarr", mode="w", zarr_format=3)
    data = group.create_array("data", shape=(3,), dtype="f4", dimension_names=("t",))
    data.attrs["coordinates"] = "aux"
    group.create_array("aux", shape=(3,), dtype="f4")

    with CFReader(ZarrDataset(str(tmp_path / "s.zarr"))) as reader:
        assert "aux" not in reader.cf_group["data"].cf_group
        assert "aux" in reader.cf_group.promoted

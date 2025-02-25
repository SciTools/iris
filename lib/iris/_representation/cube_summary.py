# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides objects describing cube summaries."""

import re

import numpy as np

import iris._lazy_data as _lazy
from iris.common.metadata import hexdigest
import iris.util


class DimensionHeader:
    def __init__(self, cube):
        if cube.shape == ():
            self.scalar = True
            self.dim_names = []
            self.shape = []
            self.contents = ["scalar cube"]
        else:
            self.scalar = False
            self.dim_names = []
            for dim in range(len(cube.shape)):
                dim_coords = cube.coords(contains_dimension=dim, dim_coords=True)
                if dim_coords:
                    self.dim_names.append(dim_coords[0].name())
                else:
                    self.dim_names.append("-- ")
            self.shape = list(cube.shape)
            self.contents = [
                name + ": %d" % dim_len
                for name, dim_len in zip(self.dim_names, self.shape)
            ]


class FullHeader:
    def __init__(self, cube, name_padding=35):
        self.name = cube.name()
        self.unit = cube.units
        self.nameunit = "{name} / ({units})".format(name=self.name, units=self.unit)
        self.name_padding = name_padding
        self.dimension_header = DimensionHeader(cube)


def string_repr(text, quote_strings=False, clip_strings=False):
    """Produce a one-line printable form of a text string."""
    # Convert any np.str_ instances to plain strings.
    text = str(text)
    force_quoted = re.findall("[\n\t]", text) or quote_strings
    if force_quoted:
        # Replace the string with its repr (including quotes).
        text = repr(text)
    if clip_strings:
        # First check for quotes.
        # N.B. not just 'quote_strings', but also array values-as-strings
        has_quotes = text[0] in "\"'"
        if has_quotes:
            # Strip off (and store) any outer quotes before clipping.
            pre_quote, post_quote = text[0], text[-1]
            text = text[1:-1]
        # clipping : use 'rider' with extra space in case it ends in a '.'
        text = iris.util.clip_string(text, rider=" ...")
        if has_quotes:
            # Replace in original quotes
            text = pre_quote + text + post_quote
    return text


def array_repr(arr):
    """Produce a single-line printable repr of an array."""
    # First take whatever numpy produces..
    text = repr(arr)
    # ..then reduce any multiple spaces and newlines.
    text = re.sub("[ \t\n]+", " ", text)
    text = string_repr(text, quote_strings=False, clip_strings=True)
    return text


def value_repr(value, quote_strings=False, clip_strings=False):
    """Produce a single-line printable version of an attribute or scalar value."""
    if hasattr(value, "dtype"):
        value = array_repr(value)
    elif isinstance(value, str):
        value = string_repr(
            value, quote_strings=quote_strings, clip_strings=clip_strings
        )
    value = str(value)
    return value


class CoordSummary:
    def _summary_coord_extra(self, cube, coord):
        # Returns the text needed to ensure this coordinate can be
        # distinguished from all others with the same name.
        extra = ""
        similar_coords = cube.coords(coord.name())
        if len(similar_coords) > 1:
            # Find all the attribute keys
            keys = set()
            for similar_coord in similar_coords:
                keys.update(similar_coord.attributes.keys())
            # Look for any attributes that vary
            vary = set()
            attributes = {}
            for key in keys:
                for similar_coord in similar_coords:
                    if key not in similar_coord.attributes:
                        vary.add(key)
                        break
                    value = similar_coord.attributes[key]
                    # Like "if attributes.setdefault(key, value) != value:"
                    # ..except setdefault fails if values are numpy arrays.
                    if key not in attributes:
                        attributes[key] = value
                    elif hexdigest(attributes[key]) != hexdigest(value):
                        # NOTE: fast and array-safe comparison, as used in
                        # :mod:`iris.common.metadata`.
                        vary.add(key)
                        break
            keys = sorted(vary & set(coord.attributes.keys()))
            bits = [
                "{}={}".format(
                    key, value_repr(coord.attributes[key], quote_strings=True)
                )
                for key in keys
            ]
            if bits:
                extra = ", ".join(bits)
        return extra


class VectorSummary(CoordSummary):
    def __init__(self, cube, vector, iscoord):
        self.name = iris.util.clip_string(vector.name())
        dims = vector.cube_dims(cube)
        self.dim_chars = ["x" if dim in dims else "-" for dim in range(len(cube.shape))]
        if iscoord:
            extra = self._summary_coord_extra(cube, vector)
            self.extra = iris.util.clip_string(extra)
        else:
            self.extra = ""


class ScalarCoordSummary(CoordSummary):
    def __init__(self, cube, coord):
        self.name = coord.name()
        if (
            coord.units in ["1", "no_unit", "unknown"]
            or coord.units.is_time_reference()
        ):
            self.unit = ""
        else:
            self.unit = " {!s}".format(coord.units)

        # Don't print values of lazy coords, as computing them could cost a lot.
        safe_to_print = not _lazy.is_lazy_data(coord.core_points())
        if not safe_to_print:
            # However there is a special case: If it is a *factory* coord, then those
            # are generally lazy.  If all the dependencies are real, then it is useful
            # (and safe) to compute + print the value.
            for factory in cube._aux_factories:
                # Note : a factory doesn't have a ".metadata" which can be matched
                # against a coord.  For now, just assume that it has a 'standard_name'
                # property (also not actually guaranteed), and require them to match.
                if coord.standard_name == factory.standard_name:
                    all_deps_real = True
                    for dependency_coord in factory.dependencies.values():
                        if (
                            dependency_coord.has_lazy_points()
                            or dependency_coord.has_lazy_bounds()
                        ):
                            all_deps_real = False

                    if all_deps_real:
                        safe_to_print = True

        if safe_to_print:
            coord_cell = coord.cell(0)
        else:
            coord_cell = None

        if coord.dtype.type is np.str_:
            self.string_type = True
            if coord_cell is not None:
                # 'lines' is value split on '\n', and _each one_ length-clipped.
                self.lines = [
                    iris.util.clip_string(str(item))
                    for item in coord_cell.point.split("\n")
                ]
                # 'content' contains a one-line printable version of the string,
                content = string_repr(coord_cell.point)
                content = iris.util.clip_string(content)
            else:
                content = "<lazy>"
                self.lines = [content]
            self.point = None
            self.bound = None
            self.content = content
        else:
            self.string_type = False
            self.lines = None
            coord_cell_cbound = None
            if coord_cell is not None:
                self.point = "{!s}".format(coord_cell.point)
                coord_cell_cbound = coord_cell.bound
            else:
                self.point = "<lazy>"

            if coord_cell_cbound is not None:
                self.bound = "({})".format(
                    ", ".join(str(val) for val in coord_cell_cbound)
                )
                self.content = "{}{}, bound={}{}".format(
                    self.point, self.unit, self.bound, self.unit
                )
            elif coord.has_bounds():
                self.bound = "+bound"
                self.content = "{}{}".format(self.point, self.bound)
            else:
                self.bound = None
                self.content = "{}{}".format(self.point, self.unit)
        extra = self._summary_coord_extra(cube, coord)
        self.extra = iris.util.clip_string(extra)


class Section:
    def is_empty(self):
        return self.contents == []


class VectorSection(Section):
    def __init__(self, title, cube, vectors, iscoord):
        self.title = title
        self.contents = [VectorSummary(cube, vector, iscoord) for vector in vectors]


class ScalarCoordSection(Section):
    def __init__(self, title, cube, scalars):
        self.title = title
        self.contents = [ScalarCoordSummary(cube, scalar) for scalar in scalars]


class ScalarCellMeasureSection(Section):
    def __init__(self, title, cell_measures):
        self.title = title
        self.contents = [cm.name() for cm in cell_measures]


class ScalarAncillaryVariableSection(Section):
    def __init__(self, title, ancillary_variables):
        self.title = title
        self.contents = [av.name() for av in ancillary_variables]


class AttributeSection(Section):
    def __init__(self, title, attributes):
        self.title = title
        self.names = []
        self.values = []
        self.contents = []
        for name, value in sorted(attributes.items()):
            value = value_repr(value, quote_strings=True, clip_strings=True)
            self.names.append(name)
            self.values.append(value)
            content = "{}: {}".format(name, value)
            self.contents.append(content)


class ScalarMeshSection(AttributeSection):
    # This happens to behave just like an attribute sections, but it
    # initialises direct from the cube.
    def __init__(self, title, cube):
        self.title = title
        self.names = []
        self.values = []
        self.contents = []
        if cube.mesh is not None:
            self.names.extend(["name", "location"])
            self.values.extend([cube.mesh.name(), cube.location])
            self.contents.extend(
                [
                    "{}: {}".format(name, value)
                    for name, value in zip(self.names, self.values)
                ]
            )


class CellMethodSection(Section):
    def __init__(self, title, cell_methods):
        self.title = title
        self.names = []
        self.values = []
        self.contents = []
        for index, method in enumerate(cell_methods):
            value = str(method)
            self.names.append(str(index))
            self.values.append(value)
            content = "{}: {}".format(index, value)
            self.contents.append(content)


class CubeSummary:
    """Provide a structure for output representations of an Iris cube."""

    def __init__(self, cube, name_padding=35):
        self.header = FullHeader(cube, name_padding)

        # Cache the derived coords so we can rely on consistent
        # object IDs.
        derived_coords = cube.derived_coords
        # Determine the cube coordinates that are scalar (single-valued)
        # AND non-dimensioned.
        dim_coords = cube.dim_coords
        aux_coords = cube.aux_coords
        all_coords = dim_coords + aux_coords + derived_coords
        scalar_coords = [
            coord
            for coord in all_coords
            if not cube.coord_dims(coord) and coord.shape == (1,)
        ]
        # Determine the cube coordinates that are not scalar BUT
        # dimensioned.
        scalar_coord_ids = set(map(id, scalar_coords))
        vector_dim_coords = [
            coord for coord in dim_coords if id(coord) not in scalar_coord_ids
        ]
        if cube.mesh is None:
            mesh_coords = []
        else:
            mesh_coords = [coord for coord in aux_coords if hasattr(coord, "mesh")]

        vector_aux_coords = [
            coord
            for coord in aux_coords
            if (id(coord) not in scalar_coord_ids and coord not in mesh_coords)
        ]
        vector_derived_coords = [
            coord for coord in derived_coords if id(coord) not in scalar_coord_ids
        ]

        # Ancillary Variables
        vector_ancillary_variables = []
        scalar_ancillary_variables = []
        for av, av_dims in cube._ancillary_variables_and_dims:
            if av_dims:
                vector_ancillary_variables.append(av)
            else:
                scalar_ancillary_variables.append(av)

        # Cell Measures
        vector_cell_measures = []
        scalar_cell_measures = []
        for cm, cm_dims in cube._cell_measures_and_dims:
            if cm_dims:
                vector_cell_measures.append(cm)
            else:
                scalar_cell_measures.append(cm)

        # Sort scalar coordinates by name.
        scalar_coords.sort(key=lambda coord: coord.name())
        # Sort vector coordinates by data dimension and name.
        vector_dim_coords.sort(key=lambda coord: (cube.coord_dims(coord), coord.name()))
        vector_aux_coords.sort(key=lambda coord: (cube.coord_dims(coord), coord.name()))
        vector_derived_coords.sort(
            key=lambda coord: (cube.coord_dims(coord), coord.name())
        )

        self.vector_sections = {}

        def add_vector_section(title, contents, iscoord=True):
            self.vector_sections[title] = VectorSection(title, cube, contents, iscoord)

        add_vector_section("Dimension coordinates:", vector_dim_coords)
        add_vector_section("Mesh coordinates:", mesh_coords)
        add_vector_section("Auxiliary coordinates:", vector_aux_coords)
        add_vector_section("Derived coordinates:", vector_derived_coords)
        add_vector_section("Cell measures:", vector_cell_measures, False)
        add_vector_section("Ancillary variables:", vector_ancillary_variables, False)

        self.scalar_sections = {}

        def add_scalar_section(section_class, title, *args):
            self.scalar_sections[title] = section_class(title, *args)

        add_scalar_section(ScalarMeshSection, "Mesh:", cube)

        add_scalar_section(
            ScalarCoordSection, "Scalar coordinates:", cube, scalar_coords
        )
        add_scalar_section(
            ScalarCellMeasureSection,
            "Scalar cell measures:",
            scalar_cell_measures,
        )
        add_scalar_section(
            ScalarAncillaryVariableSection,
            "Scalar ancillary variables:",
            scalar_ancillary_variables,
        )
        add_scalar_section(CellMethodSection, "Cell methods:", cube.cell_methods)
        add_scalar_section(AttributeSection, "Attributes:", cube.attributes)

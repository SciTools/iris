# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides objects describing cube summaries.
"""

import iris.util


def sorted_axes(axes):
    """
    Returns the axis names sorted alphabetically, with the exception that
    't', 'z', 'y', and, 'x' are sorted to the end.
    """
    return sorted(
        axes,
        key=lambda name: ({"x": 4, "y": 3, "z": 2, "t": 1}.get(name, 0), name),
    )


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
                dim_coords = cube.coords(
                    contains_dimension=dim, dim_coords=True
                )
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
        self.nameunit = "{name} / ({units})".format(
            name=self.name, units=self.unit
        )
        self.name_padding = name_padding
        self.dimension_header = DimensionHeader(cube)


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
                    if attributes.setdefault(key, value) != value:
                        vary.add(key)
                        break
            keys = sorted(vary & set(coord.attributes.keys()))
            bits = [
                "{}={!r}".format(key, coord.attributes[key]) for key in keys
            ]
            if bits:
                extra = ", ".join(bits)
        return extra


class VectorSummary(CoordSummary):
    def __init__(self, cube, vector, iscoord):
        self.name = iris.util.clip_string(vector.name())
        dims = vector.cube_dims(cube)
        self.dim_chars = [
            "x" if dim in dims else "-" for dim in range(len(cube.shape))
        ]
        if iscoord:
            extra = self._summary_coord_extra(cube, vector)
            self.extra = iris.util.clip_string(extra)
        else:
            self.extra = ""


class ScalarSummary(CoordSummary):
    def __init__(self, cube, coord):
        self.name = coord.name()
        if (
            coord.units in ["1", "no_unit", "unknown"]
            or coord.units.is_time_reference()
        ):
            self.unit = ""
        else:
            self.unit = " {!s}".format(coord.units)
        coord_cell = coord.cell(0)
        if isinstance(coord_cell.point, str):
            self.string_type = True
            self.lines = [
                iris.util.clip_string(str(item))
                for item in coord_cell.point.split("\n")
            ]
            self.point = None
            self.bound = None
            self.content = "\n".join(self.lines)
        else:
            self.string_type = False
            self.lines = None
            self.point = "{!s}".format(coord_cell.point)
            coord_cell_cbound = coord_cell.bound
            if coord_cell_cbound is not None:
                self.bound = "({})".format(
                    ", ".join(str(val) for val in coord_cell_cbound)
                )
                self.content = "{}{}, bound={}{}".format(
                    self.point, self.unit, self.bound, self.unit
                )
            else:
                self.bound = None
                self.content = "{}{}".format(self.point, self.unit)
        extra = self._summary_coord_extra(cube, coord)
        self.extra = iris.util.clip_string(extra)


class Section:
    def _init_(self):
        self.contents = []

    def is_empty(self):
        return self.contents == []


class VectorSection(Section):
    def __init__(self, title, cube, vectors, iscoord):
        self.title = title
        self.contents = [
            VectorSummary(cube, vector, iscoord) for vector in vectors
        ]


class ScalarSection(Section):
    def __init__(self, title, cube, scalars):
        self.title = title
        self.contents = [ScalarSummary(cube, scalar) for scalar in scalars]


class ScalarCellMeasureSection(Section):
    def __init__(self, title, cell_measures):
        self.title = title
        self.contents = [cm.name() for cm in cell_measures]


class AttributeSection(Section):
    def __init__(self, title, attributes):
        self.title = title
        self.names = []
        self.values = []
        self.contents = []
        for name, value in sorted(attributes.items()):
            value = iris.util.clip_string(str(value))
            self.names.append(name)
            self.values.append(value)
            content = "{}: {}".format(name, value)
            self.contents.append(content)


class CellMethodSection(Section):
    def __init__(self, title, cell_methods):
        self.title = title
        self.contents = [str(cm) for cm in cell_methods]


class CubeSummary:
    def __init__(self, cube, shorten=False, name_padding=35):
        self.section_indent = 5
        self.item_indent = 10
        self.extra_indent = 13
        self.shorten = shorten
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
        vector_aux_coords = [
            coord for coord in aux_coords if id(coord) not in scalar_coord_ids
        ]
        vector_derived_coords = [
            coord
            for coord in derived_coords
            if id(coord) not in scalar_coord_ids
        ]

        # cell measures
        vector_cell_measures = [
            cm for cm in cube.cell_measures() if cm.shape != (1,)
        ]

        # Ancillary Variables
        vector_ancillary_variables = [av for av in cube.ancillary_variables()]

        # Sort scalar coordinates by name.
        scalar_coords.sort(key=lambda coord: coord.name())
        # Sort vector coordinates by data dimension and name.
        vector_dim_coords.sort(
            key=lambda coord: (cube.coord_dims(coord), coord.name())
        )
        vector_aux_coords.sort(
            key=lambda coord: (cube.coord_dims(coord), coord.name())
        )
        vector_derived_coords.sort(
            key=lambda coord: (cube.coord_dims(coord), coord.name())
        )
        scalar_cell_measures = [
            cm for cm in cube.cell_measures() if cm.shape == (1,)
        ]

        self.vector_sections = {}

        def add_vector_section(title, contents, iscoord=True):
            self.vector_sections[title] = VectorSection(
                title, cube, contents, iscoord
            )

        add_vector_section("Dimension coordinates:", vector_dim_coords)
        add_vector_section("Auxiliary coordinates:", vector_aux_coords)
        add_vector_section("Derived coordinates:", vector_derived_coords)
        add_vector_section("Cell Measures:", vector_cell_measures, False)
        add_vector_section(
            "Ancillary Variables:", vector_ancillary_variables, False
        )

        self.scalar_sections = {}

        def add_scalar_section(section_class, title, *args):
            self.scalar_sections[title] = section_class(title, *args)

        add_scalar_section(
            ScalarSection, "Scalar Coordinates:", cube, scalar_coords
        )
        add_scalar_section(
            ScalarCellMeasureSection,
            "Scalar cell measures:",
            scalar_cell_measures,
        )
        add_scalar_section(AttributeSection, "Attributes:", cube.attributes)
        add_scalar_section(
            CellMethodSection, "Cell methods:", cube.cell_methods
        )

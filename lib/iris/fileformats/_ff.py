# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides UK Met Office Fields File (FF) format specific capabilities."""

import os
import warnings

import numpy as np

from iris.exceptions import (
    IrisDefaultingWarning,
    IrisLoadWarning,
    NotYetImplementedError,
)
from iris.fileformats._ff_cross_references import STASH_TRANS

from . import pp

IMDI = -32768

FF_HEADER_DEPTH = 256  # In words (64-bit).
DEFAULT_FF_WORD_DEPTH = 8  # In bytes.

# UM marker to signify empty lookup table entry.
_FF_LOOKUP_TABLE_TERMINATE = -99

# UM FieldsFile fixed length header names and positions.
UM_FIXED_LENGTH_HEADER = [
    ("data_set_format_version", (1,)),
    ("sub_model", (2,)),
    ("vert_coord_type", (3,)),
    ("horiz_grid_type", (4,)),
    ("dataset_type", (5,)),
    ("run_identifier", (6,)),
    ("experiment_number", (7,)),
    ("calendar", (8,)),
    ("grid_staggering", (9,)),
    ("time_type", (10,)),
    ("projection_number", (11,)),
    ("model_version", (12,)),
    ("obs_file_type", (14,)),
    ("last_fieldop_type", (15,)),
    ("first_validity_time", (21, 22, 23, 24, 25, 26, 27)),
    ("last_validity_time", (28, 29, 30, 31, 32, 33, 34)),
    ("misc_validity_time", (35, 36, 37, 38, 39, 40, 41)),
    ("integer_constants", (100, 101)),
    ("real_constants", (105, 106)),
    ("level_dependent_constants", (110, 111, 112)),
    ("row_dependent_constants", (115, 116, 117)),
    ("column_dependent_constants", (120, 121, 122)),
    ("fields_of_constants", (125, 126, 127)),
    ("extra_constants", (130, 131)),
    ("temp_historyfile", (135, 136)),
    ("compressed_field_index1", (140, 141)),
    ("compressed_field_index2", (142, 143)),
    ("compressed_field_index3", (144, 145)),
    ("lookup_table", (150, 151, 152)),
    ("total_prognostic_fields", (153,)),
    ("data", (160, 161, 162)),
]

# Offset value to convert from UM_FIXED_LENGTH_HEADER positions to
# FF_HEADER offsets.
UM_TO_FF_HEADER_OFFSET = 1
# Offset the UM_FIXED_LENGTH_HEADER positions to FF_HEADER offsets.
FF_HEADER = [
    (name, tuple(position - UM_TO_FF_HEADER_OFFSET for position in positions))
    for name, positions in UM_FIXED_LENGTH_HEADER
]

# UM FieldsFile fixed length header pointer names.
_FF_HEADER_POINTERS = [
    "integer_constants",
    "real_constants",
    "level_dependent_constants",
    "row_dependent_constants",
    "column_dependent_constants",
    "fields_of_constants",
    "extra_constants",
    "temp_historyfile",
    "compressed_field_index1",
    "compressed_field_index2",
    "compressed_field_index3",
    "lookup_table",
    "data",
]

_LBUSER_DTYPE_LOOKUP = {
    1: ">f{word_depth}",
    2: ">i{word_depth}",
    3: ">i{word_depth}",
    "default": ">f{word_depth}",
}

#: Codes used in STASH_GRID which indicate the x coordinate is on the
#: edge of the cell.
X_COORD_U_GRID = (11, 18, 27)

#: Codes used in STASH_GRID which indicate the y coordinate is on the
#: edge of the cell.
Y_COORD_V_GRID = (11, 19, 28)

#: Grid codes found in the STASH master which are currently known to be
#: handled correctly. A warning is issued if a grid is found which is not
#: handled.
HANDLED_GRIDS = (1, 2, 3, 4, 5, 21, 26, 29) + X_COORD_U_GRID + Y_COORD_V_GRID

# REAL constants header names as described by UM documentation paper F3.
# NB. These are zero-based indices as opposed to the one-based indices
# used in F3.
REAL_EW_SPACING = 0
REAL_NS_SPACING = 1
REAL_FIRST_LAT = 2
REAL_FIRST_LON = 3
REAL_POLE_LAT = 4
REAL_POLE_LON = 5


class _WarnComboLoadingDefaulting(IrisDefaultingWarning, IrisLoadWarning):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class Grid:
    """An abstract class representing the default/file-level grid
    definition for a FieldsFile.

    """

    def __init__(
        self,
        column_dependent_constants,
        row_dependent_constants,
        real_constants,
        horiz_grid_type,
    ):
        """Create a Grid from the relevant sections of the FFHeader.

        Parameters
        ----------
        column_dependent_constants : numpy.ndarray
            The `column_dependent_constants` from a FFHeader.
        row_dependent_constants : numpy.ndarray
            The `row_dependent_constants` from a FFHeader.
        real_constants : numpy.ndarray
            The `real_constants` from a FFHeader.
        horiz_grid_type : int
            `horiz_grid_type` from a FFHeader.

        """
        self.column_dependent_constants = column_dependent_constants
        self.row_dependent_constants = row_dependent_constants
        self.ew_spacing = real_constants[REAL_EW_SPACING]
        self.ns_spacing = real_constants[REAL_NS_SPACING]
        self.first_lat = real_constants[REAL_FIRST_LAT]
        self.first_lon = real_constants[REAL_FIRST_LON]
        self.pole_lat = real_constants[REAL_POLE_LAT]
        self.pole_lon = real_constants[REAL_POLE_LON]
        self.horiz_grid_type = horiz_grid_type

    def _x_vectors(self, subgrid):
        # Abstract method to return the X vector for the given sub-grid.
        raise NotImplementedError()

    def _y_vectors(self, subgrid):
        # Abstract method to return the X vector for the given sub-grid.
        raise NotImplementedError()

    def regular_x(self, subgrid):
        # Abstract method to return BZX, BDX for the given sub-grid.
        raise NotImplementedError()

    def regular_y(self, subgrid):
        # Abstract method to return BZY, BDY for the given sub-grid.
        raise NotImplementedError()

    def vectors(self, subgrid):
        """Return the X and Y coordinate vectors for the given sub-grid of
        this grid.

        Parameters
        ----------
        subgrid : int
            A "grid type code" as described in UM documentation paper C4.

        Returns
        -------
            A 2-tuple of X-vector, Y-vector.

        """
        x_p, x_u = self._x_vectors()
        y_p, y_v = self._y_vectors()
        x = x_p
        y = y_p
        if subgrid in X_COORD_U_GRID:
            x = x_u
        if subgrid in Y_COORD_V_GRID:
            y = y_v
        return x, y


class ArakawaC(Grid):
    """An abstract class representing an Arakawa C-grid."""

    def _x_vectors(self):
        x_p, x_u = None, None
        if self.column_dependent_constants is not None:
            x_p = self.column_dependent_constants[:, 0]
            if self.column_dependent_constants.shape[1] == 2:
                # Wrap around for global field
                if self.horiz_grid_type == 0:
                    x_u = self.column_dependent_constants[:-1, 1]
                else:
                    x_u = self.column_dependent_constants[:, 1]
        return x_p, x_u

    def regular_x(self, subgrid):
        """Return the "zeroth" value and step for the X coordinate on the
        given sub-grid of this grid.

        Parameters
        ----------
        subgrid : int
            A "grid type code" as described in UM documentation paper C4.

        Returns
        -------
            A 2-tuple of BZX, BDX.

        """
        bdx = self.ew_spacing
        bzx = self.first_lon - bdx
        if subgrid in X_COORD_U_GRID:
            bzx += 0.5 * bdx
        return bzx, bdx

    def regular_y(self, subgrid):
        """Return the "zeroth" value and step for the Y coordinate on the
        given sub-grid of this grid.

        Parameters
        ----------
        subgrid : int
            A "grid type code" as described in UM documentation paper C4.

        Returns
        -------
            A 2-tuple of BZY, BDY.

        """
        bdy = self.ns_spacing
        bzy = self.first_lat - bdy
        if subgrid in Y_COORD_V_GRID:
            bzy += self._v_offset * bdy
        return bzy, bdy


class NewDynamics(ArakawaC):
    """An Arakawa C-grid as used by UM New Dynamics.

    The theta and u points are at the poles.

    """

    _v_offset = 0.5

    def _y_vectors(self):
        y_p, y_v = None, None
        if self.row_dependent_constants is not None:
            y_p = self.row_dependent_constants[:, 0]
            if self.row_dependent_constants.shape[1] == 2:
                y_v = self.row_dependent_constants[:-1, 1]
        return y_p, y_v


class ENDGame(ArakawaC):
    """An Arakawa C-grid as used by UM ENDGame.

    The v points are at the poles.

    """

    _v_offset = -0.5

    def _y_vectors(self):
        y_p, y_v = None, None
        if self.row_dependent_constants is not None:
            y_p = self.row_dependent_constants[:-1, 0]
            if self.row_dependent_constants.shape[1] == 2:
                y_v = self.row_dependent_constants[:, 1]
        return y_p, y_v


class FFHeader:
    """A class to represent the FIXED_LENGTH_HEADER section of a FieldsFile."""

    GRID_STAGGERING_CLASS = {3: NewDynamics, 6: ENDGame}

    def __init__(self, filename, word_depth=DEFAULT_FF_WORD_DEPTH):
        """Create a FieldsFile header instance by reading the
        FIXED_LENGTH_HEADER section of the FieldsFile, making the names
        defined in FF_HEADER available as attributes of a FFHeader instance.

        Parameters
        ----------
        filename : str
            Specify the name of the FieldsFile.

        Returns
        -------
        FFHeader object.

        """
        #: File name of the FieldsFile.
        self.ff_filename = filename
        self._word_depth = word_depth

        # Read the FF header data
        with open(filename, "rb") as ff_file:
            # typically 64-bit words (aka. int64 or ">i8")
            header_data = _parse_binary_stream(
                ff_file,
                dtype=">i{0}".format(word_depth),
                count=FF_HEADER_DEPTH,
            )
            header_data = tuple(header_data)
            # Create FF instance attributes
            for name, offsets in FF_HEADER:
                if len(offsets) == 1:
                    value = header_data[offsets[0]]
                else:
                    value = header_data[offsets[0] : offsets[-1] + 1]
                setattr(self, name, value)

            # Turn the pointer values into real arrays.
            for elem in _FF_HEADER_POINTERS:
                if elem not in ["data", "lookup_table"]:
                    if self._attribute_is_pointer_and_needs_addressing(elem):
                        addr = getattr(self, elem)
                        ff_file.seek((addr[0] - 1) * word_depth, os.SEEK_SET)
                        if len(addr) == 2:
                            if elem == "integer_constants":
                                res = _parse_binary_stream(
                                    ff_file,
                                    dtype=">i{0}".format(word_depth),
                                    count=addr[1],
                                )
                            else:
                                res = _parse_binary_stream(
                                    ff_file,
                                    dtype=">f{0}".format(word_depth),
                                    count=addr[1],
                                )
                        elif len(addr) == 3:
                            res = _parse_binary_stream(
                                ff_file,
                                dtype=">f{0}".format(word_depth),
                                count=addr[1] * addr[2],
                            )
                            res = res.reshape((addr[1], addr[2]), order="F")
                        else:
                            raise ValueError(
                                "ff header element {} is not"
                                "handled correctly".format(elem)
                            )
                    else:
                        res = None
                    setattr(self, elem, res)

    def __str__(self):
        attributes = []
        for name, _ in FF_HEADER:
            attributes.append("    {}: {}".format(name, getattr(self, name)))
        return "FF Header:\n" + "\n".join(attributes)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.ff_filename)

    def _attribute_is_pointer_and_needs_addressing(self, name):
        if name in _FF_HEADER_POINTERS:
            attr = getattr(self, name)

            # Check that we haven't already addressed this pointer,
            # that the pointer is actually referenceable (i.e. >0)
            # and that the attribute is not marked as missing.
            is_referenceable = (
                isinstance(attr, tuple) and attr[0] > 0 and attr[0] != IMDI
            )
        else:
            msg = "{!r} object does not have pointer attribute {!r}"
            raise AttributeError(msg.format(self.__class__.__name__, name))
        return is_referenceable

    def shape(self, name):
        """Return the dimension shape of the FieldsFile FIXED_LENGTH_HEADER
        pointer attribute.

        Parameters
        ----------
        name : str
            Specify the name of the FIXED_LENGTH_HEADER attribute.

        Returns
        -------
        Dimension tuple.

        """
        if name in _FF_HEADER_POINTERS:
            value = getattr(self, name)[1:]
        else:
            msg = "{!r} object does not have pointer address {!r}"
            raise AttributeError(msg.format(self.__class_.__name__, name))
        return value

    def grid(self):
        """Return the Grid definition for the FieldsFile."""
        grid_class = self.GRID_STAGGERING_CLASS.get(self.grid_staggering)
        if grid_class is None:
            grid_class = NewDynamics
            warnings.warn(
                "Staggered grid type: {} not currently interpreted, assuming "
                "standard C-grid".format(self.grid_staggering),
                category=_WarnComboLoadingDefaulting,
            )
        grid = grid_class(
            self.column_dependent_constants,
            self.row_dependent_constants,
            self.real_constants,
            self.horiz_grid_type,
        )
        return grid


class FF2PP:
    """A class to extract the individual PPFields from within a FieldsFile."""

    def __init__(self, filename, read_data=False, word_depth=DEFAULT_FF_WORD_DEPTH):
        """Create a FieldsFile to Post Process instance that returns a generator
        of PPFields contained within the FieldsFile.

        Parameters
        ----------
        filename : str
            Specify the name of the FieldsFile.
        read_data : bool, optional
            Specify whether to read the associated PPField data within
            the FieldsFile.  Default value is False.

        Returns
        -------
        PPField generator.

        Examples
        --------
        ::

            >>> for field in ff.FF2PP(filename):
            ...     print(field)

        """
        self._ff_header = FFHeader(filename, word_depth=word_depth)
        self._word_depth = word_depth
        self._filename = filename
        self._read_data = read_data

    def _payload(self, field):
        """Calculate the payload data depth (in bytes) and type."""
        lbpack_n1 = field.raw_lbpack % 10
        if lbpack_n1 == 0:
            word_depth = self._word_depth
            # Data payload is not packed.
            data_words = field.lblrec - field.lbext
            # Determine PP field 64-bit payload datatype.
            lookup = _LBUSER_DTYPE_LOOKUP
            dtype_template = lookup.get(field.lbuser[0], lookup["default"])
            dtype_name = dtype_template.format(word_depth=self._word_depth)
            data_type = np.dtype(dtype_name)
        else:
            word_depth = pp.PP_WORD_DEPTH
            # Data payload is packed.
            if lbpack_n1 == 1:
                # Data packed using WGDOS archive method.
                data_words = field.lbnrec * 2
            elif lbpack_n1 == 2:
                # Data packed using CRAY 32-bit method.
                data_words = field.lblrec - field.lbext
            else:
                msg = "PP fields with LBPACK of {} are not supported."
                raise NotYetImplementedError(msg.format(field.raw_lbpack))

            # Determine PP field payload datatype.
            lookup = pp.LBUSER_DTYPE_LOOKUP
            data_type = lookup.get(field.lbuser[0], lookup["default"])

        if field.boundary_packing is not None:
            if lbpack_n1 not in (0, 2):
                # Can't handle the usual packing methods with LBC data.
                raise ValueError(
                    "LBC data has LBPACK = {:d}, but packed LBC data is not "
                    "supported.".format(field.raw_lbpack)
                )
            # Adjust to packed data size, for LBC data.
            # NOTE: logic duplicates that in pp._data_bytes_to_shaped_array.
            pack_dims = field.boundary_packing
            boundary_height = pack_dims.y_halo + pack_dims.rim_width
            boundary_width = pack_dims.x_halo + pack_dims.rim_width
            y_height, x_width = field.lbrow, field.lbnpt
            mid_height = y_height - 2 * boundary_height
            data_words = boundary_height * x_width * 2 + boundary_width * mid_height * 2

        data_depth = data_words * word_depth
        return data_depth, data_type

    def _det_border(self, field_dim, halo_dim):
        # Update field coordinates for a variable resolution LBC file where
        # the resolution of the very edge (within the rim width) is assumed to
        # be same as the halo.
        def range_order(range1, range2, resolution):
            # Handles whether increasing/decreasing ranges.
            if np.sign(resolution) > 0:
                lower = range1
                upper = range2
            else:
                upper = range1
                lower = range2
            return lower, upper

        # Ensure that the resolution is the same on both edges.
        res_low = field_dim[1] - field_dim[0]
        res_high = field_dim[-1] - field_dim[-2]
        if not np.allclose(res_low, res_high):
            msg = (
                "The x or y coordinates of your boundary condition field "
                "may be incorrect, not having taken into account the "
                "boundary size."
            )
            warnings.warn(msg, category=IrisLoadWarning)
        else:
            range2 = field_dim[0] - res_low
            range1 = field_dim[0] - halo_dim * res_low
            lower, upper = range_order(range1, range2, res_low)
            extra_before = np.linspace(lower, upper, halo_dim)

            range1 = field_dim[-1] + res_high
            range2 = field_dim[-1] + halo_dim * res_high
            lower, upper = range_order(range1, range2, res_high)
            extra_after = np.linspace(lower, upper, halo_dim)

            field_dim = np.concatenate([extra_before, field_dim, extra_after])
        return field_dim

    def _adjust_field_for_lbc(self, field):
        """Make an LBC field look like a 'normal' field for rules processing."""
        # Set LBTIM to indicate the specific time encoding for LBCs,
        # i.e. t1=forecast, t2=reference
        lbtim_default = 11
        if field.lbtim not in (0, lbtim_default):
            raise ValueError(
                "LBC field has LBTIM of {:d}, expected only 0 or {:d}.".format(
                    field.lbtim, lbtim_default
                )
            )
        field.lbtim = lbtim_default

        # Set LBVC to indicate the specific height encoding for LBCs,
        # i.e. hybrid height layers.
        lbvc_default = 65
        if field.lbvc not in (0, lbvc_default):
            raise ValueError(
                "LBC field has LBVC of {:d}, expected only 0 or {:d}.".format(
                    field.lbvc, lbvc_default
                )
            )
        field.lbvc = lbvc_default
        # Specifying a vertical encoding scheme means a usable vertical
        # coordinate can be produced, because we also record the level
        # number in each result field:  Thus they are located, and can
        # be stacked, in a vertical dimension.
        # See per-layer loop (below).

        # Calculate field packing details.
        name_mapping = dict(
            rim_width=slice(4, 6), y_halo=slice(2, 4), x_halo=slice(0, 2)
        )
        boundary_packing = pp.SplittableInt(field.lbuser[2], name_mapping)
        # Store packing on the field, which affects how it gets the data.
        field.boundary_packing = boundary_packing
        # Fix the lbrow and lbnpt to be the actual size of the data
        # array, since the field is no longer a "boundary" fields file
        # field.
        # Note: The documentation states that lbrow (y) doesn't
        # contain the halo rows, but no such comment exists at UM v8.5
        # for lbnpt (x). Experimentation has shown that lbnpt also
        # excludes the halo size.
        field.lbrow += 2 * boundary_packing.y_halo
        field.lbnpt += 2 * boundary_packing.x_halo

        # Update the x and y coordinates for this field. Note: it may
        # be that this needs to update x and y also, but that is yet
        # to be confirmed.
        if field.bdx in (0, field.bmdi) or field.bdy in (0, field.bmdi):
            field.x = self._det_border(field.x, boundary_packing.x_halo)
            field.y = self._det_border(field.y, boundary_packing.y_halo)
        else:
            if field.bdy < 0:
                warnings.warn(
                    "The LBC has a bdy less than 0. No "
                    "case has previously been seen of "
                    "this, and the decompression may be "
                    "erroneous.",
                    category=IrisLoadWarning,
                )
            field.bzx -= field.bdx * boundary_packing.x_halo
            field.bzy -= field.bdy * boundary_packing.y_halo

    def _fields_over_all_levels(self, field):
        """Replicate the field over all model levels, setting LBLEV for each.

        This is appropriate for LBC data.
        Yields an iterator producing a sequence of distinct field objects.

        """
        n_all_levels = self._ff_header.level_dependent_constants.shape[0]
        levels_count = field.lbhem - 100
        if levels_count < 1:
            raise ValueError(
                "LBC field has LBHEM of {:d}, but this should be (100 "
                "+ levels-per-field-type), hence >= 101.".format(field.lbhem)
            )
        if levels_count > n_all_levels:
            raise ValueError(
                "LBC field has LBHEM of (100 + levels-per-field-type) "
                "= {:d}, but this is more than the total number of levels "
                "in the file = {:d}).".format(field.lbhem, n_all_levels)
            )

        for i_model_level in range(levels_count):
            # Make subsequent fields alike, but distinct.
            if i_model_level > 0:
                field = field.copy()
            # Provide the correct "model level" value.
            field.lblev = i_model_level
            # TODO: as LBC lookup headers cover multiple layers, they
            # contain no per-layer height values, which are all 0.
            # So the field's "height" coordinate will be useless.
            # It should be possible to fix this here, by calculating
            # per-layer Z and C values from the file header, and
            # setting blev/brlev/brsvd1 and bhlev/bhrlev/brsvd2 here,
            # but we don't yet do this.

            yield field

    def _extract_field(self):
        # FF table pointer initialisation based on FF LOOKUP table
        # configuration.

        lookup_table = self._ff_header.lookup_table
        table_index, table_entry_depth, table_count = lookup_table
        table_offset = (table_index - 1) * self._word_depth  # in bytes
        table_entry_depth = table_entry_depth * self._word_depth  # in bytes
        # Open the FF for processing.
        with open(self._ff_header.ff_filename, "rb") as ff_file:
            ff_file_seek = ff_file.seek

            is_boundary_packed = self._ff_header.dataset_type == 5

            grid = self._ff_header.grid()

            # Process each FF LOOKUP table entry.
            while table_count:
                table_count -= 1
                # Move file pointer to the start of the current FF LOOKUP
                # table entry.
                ff_file_seek(table_offset, os.SEEK_SET)

                # Read the current PP header entry from the FF LOOKUP table.
                header_longs = _parse_binary_stream(
                    ff_file,
                    dtype=">i{0}".format(self._word_depth),
                    count=pp.NUM_LONG_HEADERS,
                )
                # Check whether the current FF LOOKUP table entry is valid.
                if header_longs[0] == _FF_LOOKUP_TABLE_TERMINATE:
                    # There are no more FF LOOKUP table entries to read.
                    break
                header_floats = _parse_binary_stream(
                    ff_file,
                    dtype=">f{0}".format(self._word_depth),
                    count=pp.NUM_FLOAT_HEADERS,
                )
                header = tuple(header_longs) + tuple(header_floats)

                # Calculate next FF LOOKUP table entry.
                table_offset += table_entry_depth

                # Construct a PPField object and populate using the header_data
                # read from the current FF LOOKUP table.
                # (The PPField sub-class will depend on the header release
                # number.)
                # (Some Fields File fields are UM specific scratch spaces, with
                # no header release number, these will throw an exception from
                # the PP module and are skipped to enable reading of the file.
                try:
                    field = pp.make_pp_field(header)

                    # Fast stash look-up.
                    stash_s = field.lbuser[3] // 1000
                    stash_i = field.lbuser[3] % 1000
                    stash = "m{:02}s{:02}i{:03}".format(
                        field.lbuser[6], stash_s, stash_i
                    )
                    stash_entry = STASH_TRANS.get(stash, None)
                    if stash_entry is None:
                        subgrid = None
                    else:
                        subgrid = stash_entry.grid_code
                        if subgrid not in HANDLED_GRIDS:
                            warnings.warn(
                                "The stash code {} is on a grid {} "
                                "which has not been explicitly "
                                "handled by the fieldsfile loader."
                                " Assuming the data is on a P grid"
                                ".".format(stash, subgrid),
                                category=_WarnComboLoadingDefaulting,
                            )

                    field.x, field.y = grid.vectors(subgrid)

                    # Use the per-file grid if no per-field metadata is
                    # available.
                    no_x = field.bzx in (0, field.bmdi) and field.x is None
                    no_y = field.bzy in (0, field.bmdi) and field.y is None
                    if no_x and no_y:
                        if subgrid is None:
                            msg = (
                                "The STASH code {0} was not found in the "
                                "STASH to grid type mapping. Picking the P "
                                "position as the cell type".format(stash)
                            )
                            warnings.warn(
                                msg,
                                category=_WarnComboLoadingDefaulting,
                            )
                        field.bzx, field.bdx = grid.regular_x(subgrid)
                        field.bzy, field.bdy = grid.regular_y(subgrid)
                        field.bplat = grid.pole_lat
                        field.bplon = grid.pole_lon
                    elif no_x or no_y:
                        warnings.warn(
                            "Partially missing X or Y coordinate values.",
                            category=IrisLoadWarning,
                        )

                    # Check for LBC fields.
                    is_boundary_packed = self._ff_header.dataset_type == 5
                    if is_boundary_packed:
                        # Apply adjustments specific to LBC data.
                        self._adjust_field_for_lbc(field)

                    # Calculate start address of the associated PP header data.
                    data_offset = field.lbegin * self._word_depth
                    # Determine PP field payload depth and type.
                    data_depth, data_type = self._payload(field)

                    # Produce (yield) output fields.
                    if is_boundary_packed:
                        fields = self._fields_over_all_levels(field)
                    else:
                        fields = [field]
                    for result_field in fields:
                        # Add a field data element.
                        if self._read_data:
                            # Read the actual bytes. This can then be converted
                            # to a numpy array at a higher level.
                            ff_file_seek(data_offset, os.SEEK_SET)
                            result_field.data = pp.LoadedArrayBytes(
                                ff_file.read(data_depth), data_type
                            )
                        else:
                            # Provide enough context to read the data bytes
                            # later.
                            result_field.data = (
                                self._filename,
                                data_offset,
                                data_depth,
                                data_type,
                            )

                        data_offset += data_depth

                        yield result_field
                except ValueError as valerr:
                    msg = (
                        "Input field skipped as PPField creation failed :"
                        " error = {!r}"
                    )
                    warnings.warn(msg.format(str(valerr)), category=IrisLoadWarning)

    def __iter__(self):
        return pp._interpret_fields(self._extract_field())


def _parse_binary_stream(file_like, dtype=np.float64, count=-1):
    """Replacement :func:`numpy.fromfile` due to python3 performance issues.

    Parameters
    ----------
    file_like :
        Standard python file_like object.
    dtype : no.float64, optional
        Data type to be parsed out, used to work out bytes read in.
    count : optional, default=-1
        The number of values required to be generated from the parsing.
        The default is -1, which will read the entire contexts of the file_like
        object and generate as many values as possible.

    """
    # There are a wide range of types supported, we just need to know the byte
    # size of the object, so we just make sure we've go an instance of a
    # np.dtype
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    # Allocate bytearray for the file to be read into, allowing the numpy array
    # to be writable.
    _buffer = bytearray(count * dtype.itemsize)
    file_like.readinto(_buffer)

    # Let numpy do the heavy lifting once we've sorted the file reading.
    array = np.frombuffer(_buffer, dtype=dtype, count=-1)
    return array


def load_cubes(filenames, callback, constraints=None):
    """Loads cubes from a list of fields files filenames.

    Parameters
    ----------
    filenames :
        List of fields files filenames to load
    callback :
        A function which can be passed on to :func:`iris.io.run_callback`

    Notes
    -----
    .. note::

        The resultant cubes may not be in the order that they are in the
        file (order is not preserved when there is a field with
        orography references).

    """
    return pp._load_cubes_variable_loader(
        filenames, callback, FF2PP, constraints=constraints
    )


def load_cubes_32bit_ieee(filenames, callback, constraints=None):
    """Loads cubes from a list of 32bit ieee converted fieldsfiles filenames.

    See Also
    --------
    :func:`load_cubes`
        For keyword details

    """
    return pp._load_cubes_variable_loader(
        filenames, callback, FF2PP, {"word_depth": 4}, constraints=constraints
    )

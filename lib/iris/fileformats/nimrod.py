# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Provides NIMROD file format capabilities."""

import glob
import os
import struct
import sys

import numpy as np

import iris
from iris.exceptions import TranslationError
import iris.fileformats.nimrod_load_rules

# general header (int16) elements 1-31 (Fortran bytes 1-62)
general_header_int16s = (
    "vt_year",
    "vt_month",
    "vt_day",
    "vt_hour",
    "vt_minute",
    "vt_second",
    "dt_year",
    "dt_month",
    "dt_day",
    "dt_hour",
    "dt_minute",
    "datum_type",
    "datum_len",
    "experiment_num",
    "horizontal_grid_type",
    "num_rows",
    "num_cols",
    "nimrod_version",
    "field_code",
    "vertical_coord_type",
    "reference_vertical_coord_type",
    "data_specific_float32_len",
    "data_specific_int16_len",
    "origin_corner",
    "int_mdi",
    "period_minutes",
    "num_model_levels",
    "proj_biaxial_ellipsoid",
    "ensemble_member",
    "model_origin_id",
    "averagingtype",
)


# general header (float32) elements 32-59 (Fortran bytes 63-174)
general_header_float32s = (
    "vertical_coord",
    "reference_vertical_coord",
    "y_origin",
    "row_step",
    "x_origin",
    "column_step",
    "float32_mdi",
    "MKS_data_scaling",
    "data_offset",
    "x_offset",
    "y_offset",
    "true_origin_latitude",
    "true_origin_longitude",
    "true_origin_easting",
    "true_origin_northing",
    "tm_meridian_scaling",
    "threshold_value_alt",
    "threshold_value",
)


# data specific header (float32) elements 60-104 (Fortran bytes 175-354)
data_header_float32s = (
    "tl_y",
    "tl_x",
    "tr_y",
    "tr_x",
    "br_y",
    "br_x",
    "bl_y",
    "bl_x",
    "sat_calib",
    "sat_space_count",
    "ducting_index",
    "elevation_angle",
    "neighbourhood_radius",
    "threshold_vicinity_radius",
    "recursive_filter_alpha",
    "threshold_fuzziness",
    "threshold_duration_fuzziness",
)


# data specific header (char) elements 105-107 (bytes 355-410)
# units, source and title


# data specific header (int16) elements 108-159 (Fortran bytes 411-512)
data_header_int16s = (
    "threshold_type",
    "probability_method",
    "recursive_filter_iterations",
    "member_count",
    "probability_period_of_event",
    "data_header_int16_05",
    "soil_type",
    "radiation_code",
    "data_header_int16_08",
    "data_header_int16_09",
    "data_header_int16_10",
    "data_header_int16_11",
    "data_header_int16_12",
    "data_header_int16_13",
    "data_header_int16_14",
    "data_header_int16_15",
    "data_header_int16_16",
    "data_header_int16_17",
    "data_header_int16_18",
    "data_header_int16_19",
    "data_header_int16_20",
    "data_header_int16_21",
    "data_header_int16_22",
    "data_header_int16_23",
    "data_header_int16_24",
    "data_header_int16_25",
    "data_header_int16_26",
    "data_header_int16_27",
    "data_header_int16_28",
    "data_header_int16_29",
    "data_header_int16_30",
    "data_header_int16_31",
    "data_header_int16_32",
    "data_header_int16_33",
    "data_header_int16_34",
    "data_header_int16_35",
    "data_header_int16_36",
    "data_header_int16_37",
    "data_header_int16_38",
    "data_header_int16_39",
    "data_header_int16_40",
    "data_header_int16_41",
    "data_header_int16_42",
    "data_header_int16_43",
    "data_header_int16_44",
    "data_header_int16_45",
    "data_header_int16_46",
    "data_header_int16_47",
    "data_header_int16_48",
    "data_header_int16_49",
    "period_seconds",
)


def _read_chars(infile, num):
    """Read characters from the (big-endian) file."""
    instr = infile.read(num)
    result = struct.unpack(">%ds" % num, instr)[0]
    result = result.decode()
    return result


class NimrodField:
    """
    A data field from a NIMROD file.

    Capable of converting itself into a :class:`~iris.cube.Cube`

    References:
        Met Office (2003): Met Office Rain Radar Data from the NIMROD System.
        NCAS British Atmospheric Data Centre, date of citation.
        http://catalogue.ceda.ac.uk/uuid/82adec1f896af6169112d09cc1174499

    """

    def __init__(self, from_file=None):
        """
        Create a NimrodField object and optionally read from an open file.

        Example::

            with open("nimrod_file", "rb") as infile:
                field = NimrodField(infile)

        """
        if from_file is not None:
            self.read(from_file)

    def read(self, infile):
        """Read the next field from the given file object."""
        self._read_header(infile)
        self._read_data(infile)

    def _read_header_subset(self, infile, names, dtype):
        # Read contiguous header items of the same data type.
        values = np.fromfile(infile, dtype=dtype, count=len(names))
        if sys.byteorder == "little":
            values.byteswap(True)
        for i, name in enumerate(names):
            setattr(self, name, values[i])

    def _read_header(self, infile):
        """Load the 512 byte header (surrounded by 4-byte length)."""

        leading_length = struct.unpack(">L", infile.read(4))[0]
        if leading_length != 512:
            raise TranslationError("Expected header leading_length of 512")

        # general header (int16) elements 1-31 (bytes 1-62)
        self._read_header_subset(infile, general_header_int16s, np.int16)

        # general header (float32) elements 32-59 (bytes 63-174)
        self._read_header_subset(infile, general_header_float32s, np.float32)
        # skip unnamed floats
        infile.seek(4 * (28 - len(general_header_float32s)), os.SEEK_CUR)

        # data specific header (float32) elements 60-104 (bytes 175-354)
        self._read_header_subset(infile, data_header_float32s, np.float32)
        # skip unnamed floats
        infile.seek(4 * (45 - len(data_header_float32s)), os.SEEK_CUR)

        # data specific header (char) elements 105-107 (bytes 355-410)
        self.units = _read_chars(infile, 8)
        self.source = _read_chars(infile, 24)
        self.title = _read_chars(infile, 24)

        # data specific header (int16) elements 108- (bytes 411-512)
        self._read_header_subset(infile, data_header_int16s, np.int16)
        # skip unnamed int16s
        infile.seek(2 * (51 - len(data_header_int16s)), os.SEEK_CUR)

        trailing_length = struct.unpack(">L", infile.read(4))[0]
        if trailing_length != leading_length:
            raise TranslationError(
                "Expected header trailing_length of {}, "
                "got {}.".format(leading_length, trailing_length)
            )

    def _read_data(self, infile):
        """
        Read the data array: int8, int16, int32 or float32

        (surrounded by 4-byte length, at start and end)

        """
        # what are we expecting?
        num_data = int(self.num_rows) * int(self.num_cols)
        num_data_bytes = int(num_data) * int(self.datum_len)

        # format string for unpacking the file.read()
        # 0:real
        if self.datum_type == 0:
            numpy_dtype = np.float32
        # 1:int
        elif self.datum_type == 1:
            if self.datum_len == 1:
                numpy_dtype = np.int8
            elif self.datum_len == 2:
                numpy_dtype = np.int16
            elif self.datum_len == 4:
                numpy_dtype = np.int32
            else:
                raise TranslationError(
                    "Undefined datum length " "%d" % self.datum_type
                )
        # 2:byte
        elif self.datum_type == 2:
            numpy_dtype = np.byte
        else:
            raise TranslationError("Undefined data type")
        leading_length = struct.unpack(">L", infile.read(4))[0]
        if leading_length != num_data_bytes:
            raise TranslationError(
                "Expected data leading_length of %d" % num_data_bytes
            )

        self.data = np.fromfile(infile, dtype=numpy_dtype, count=num_data)

        if sys.byteorder == "little":
            self.data.byteswap(True)

        trailing_length = struct.unpack(">L", infile.read(4))[0]
        if trailing_length != leading_length:
            raise TranslationError(
                "Expected data trailing_length of %d" % num_data_bytes
            )

        # Form the correct shape.
        self.data = self.data.reshape(self.num_rows, self.num_cols)


def load_cubes(filenames, callback=None):
    """
    Loads cubes from a list of NIMROD filenames.

    Args:

    * filenames - list of NIMROD filenames to load

    Kwargs:

    * callback - a function which can be passed on to
                 :func:`iris.io.run_callback`

    .. note::

        The resultant cubes may not be in the same order as in the files.

    """
    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        for path in glob.glob(filename):
            with open(path, "rb") as infile:
                while True:
                    try:
                        field = NimrodField(infile)
                    except struct.error:
                        # End of file. Move on to the next file.
                        break

                    cube = iris.fileformats.nimrod_load_rules.run(field)

                    # Were we given a callback?
                    if callback is not None:
                        cube = iris.io.run_callback(
                            callback, cube, field, filename
                        )
                    if cube is None:
                        continue

                    yield cube

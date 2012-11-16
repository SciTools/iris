# (C) British Crown Copyright 2010 - 2012, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Provides NIMROD file format capabilities."""

import glob
import netcdftime
import numpy as np
import os
import struct
import sys
import warnings

import iris
from iris.coord_systems import OSGB
from iris.coords import DimCoord
from iris.exceptions import TranslationError


# general header (int16) elements
general_header_int16s = ("vt_year", "vt_month", "vt_day", "vt_hour", "vt_minute",
                         "vt_second", "dt_year", "dt_month", "dt_day", "dt_hour",
                         "dt_minute", "datum_type", "datum_len", "experiment_num",
                         "horizontal_grid_type", "num_rows", "num_cols",
                         "nimrod_version", "field_code", "vertical_coord_type",
                         "reference_vertical_coord_type", 
                        "data_specific_float32_len", "data_specific_int16_len",
                        "origin_corner", "int_mdi", "period_minutes",
                        "num_model_levels", "proj_biaxial_ellipsoid",
                        "ensemble_member", "spare1", "spare2")


# general header (float32) elements
general_header_float32s = ("vertical_coord", "reference_vertical_coord",
                           "y_origin", "row_step", "x_origin", "column_step",
                           "float32_mdi", "MKS_data_scaling", "data_offset",
                           "x_offset", "y_offset", "true_origin_latitude",
                           "true_origin_longitude", "true_origin_easting",
                           "true_origin_northing", "tm_meridian_scaling")


# data specific header (float32) elements
data_header_float32s = ("tl_y", "tl_x", "tr_y", "ty_x", "br_y", "br_x", "bl_y",
                        "bl_x", "sat_calib", "sat_space_count", "ducting_index",
                        "elevation_angle")


# data specific header (int16) elements
data_header_int16s = ("radar_num", "radars_bitmask", "more_radars_bitmask",
                      "clutter_map_num", "calibration_type", "bright_band_height",
                      "bright_band_intensity", "bright_band_test1",
                      "bright_band_test2", "infill_flag", "stop_elevation",
                      "int16_vertical_coord", "int16_reference_vertical_coord",
                      "int16_y_origin", "int16_row_step", "int16_x_origin",
                      "int16_column_step", "int16_float32_mdi",
                      "int16_data_scaling", "int16_data_offset", "int16_x_offset",
                      "int16_y_offset", "int16_true_origin_latitude", 
                      "int16_true_origin_longitude", "int16_tl_y", "int16_tl_x",
                      "int16_tr_y", "int16_ty_x", "int16_br_y", "int16_br_x",
                      "int16_bl_y", "int16_bl_x", "sensor_id", "meteosat_id",
                      "alphas_available")


def _read_chars(infile, num):
    """Read characters from the (big-endian) file."""
    instr = infile.read(num)
    return struct.unpack(">%ds" % num, instr)[0]


class NimrodField(object):
    """
    A data field from a NIMROD file.
    
    Capable of converting itself into a :class:`~iris.cube.Cube`
    
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
            raise TranslationError('Expected header trailing_length of {}, '
                                   'got {}.'.format(leading_length,
                                                    trailing_length))
        
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
            data_format_string = ">%df" % num_data
        # 1:int
        elif self.datum_type == 1:
            if self.datum_len == 1:
                numpy_dtype = np.int8
                data_format_string = ">%db" % num_data
            elif self.datum_len == 2:
                numpy_dtype = np.int16
                data_format_string = ">%dh" % num_data
            elif self.datum_len == 4:
                numpy_dtype = np.int32
                data_format_string = ">%di" % num_data
            else:
                raise TranslationError("Undefined datum length %d" % self.datum_type)
        # 2:byte
        elif self.datum_type == 2:
            numpy_dtype = np.byte
            data_format_string = ">%db" % num_data
        else:
            raise TranslationError("Undefined data type")

        leading_length = struct.unpack(">L", infile.read(4))[0]
        if leading_length != num_data_bytes:
            raise TranslationError("Expected data leading_length of %d" % 
                                   num_data_bytes)
        
        # TODO: Deal appropriately with MDI. Can't just create masked arrays
        #       as cube merge converts masked arrays with no masks to ndarrays,
        #       thus mergable cube can split one mergable cube into two.
        self.data = np.fromfile(infile, dtype=numpy_dtype, count=num_data)

        if sys.byteorder == "little":
            self.data.byteswap(True)

        trailing_length = struct.unpack(">L", infile.read(4))[0]
        if trailing_length != leading_length:
            raise TranslationError("Expected data trailing_length of %d" % 
                                   num_data_bytes)
        
        # Form the correct shape.
        self.data = self.data.reshape(self.num_rows, self.num_cols)
        
        # Flip the data to get origin in bottom left.
        if self.origin_corner == 0:  # top left
            # Make sure we copy, to ensure contiguous elements (e.g for crc32)
            self.data = self.data[::-1, :].copy()
        else:
            raise TranslationError("Corner {0} not yet implemented".
                                   format(self.origin_corner))
    
    def to_cube(self):
        """Return a new :class:`~iris.cube.Cube`, created from this NimrodField."""
        MISSING_INT = self.int_mdi
        MISSING_FLOAT = self.float32_mdi

        # name
        cube = iris.cube.Cube(self.data)
        cube.rename(self.title.strip())
    
        # Can we set the units?
        units = self.units.strip()
        try:
            cube.units = units
        except ValueError:
            # Just add it as an attribute.
            warnings.warn("Unhandled units '{0}' recorded in cube attributes.".
                          format(units))
            cube.attributes["invalid_units"] = units
    
        # time                
        valid_date = netcdftime.datetime(self.vt_year, self.vt_month,
                                         self.vt_day, self.vt_hour,
                                         self.vt_minute, self.vt_second)
        time_coord = DimCoord(iris.unit.date2num(valid_date,
                                                 'hours since 1970-01-01 00:00:00', 
                                                 iris.unit.CALENDAR_STANDARD), 
                              standard_name='time', units='hours')
        cube.add_aux_coord(time_coord)
    
        if self.dt_year != MISSING_INT:        
            data_date = netcdftime.datetime(self.dt_year, self.dt_month, self.dt_day, 
                                            self.dt_hour, self.dt_minute)
            ref_time_coord = DimCoord(iris.unit.date2num(data_date, 
                                                         'hours since 1970-01-01 00:00:00', 
                                                         iris.unit.CALENDAR_STANDARD), 
                                      standard_name='forecast_reference_time',
                                      units='hours')
            cube.add_aux_coord(ref_time_coord)
        
        if self.period_minutes != MISSING_INT:
            if self.period_minutes != 0:
                raise TranslationError("Period_minutes not yet handled")
        
        # experiment
        if self.experiment_num != MISSING_INT:
            cube.add_aux_coord(DimCoord(self.experiment_num, 
                                        long_name="experiment_number"))
            
        # horizontal grid
        if self.proj_biaxial_ellipsoid not in [MISSING_INT, 0]:
            raise TranslationError("Biaxial ellipsoid %d not yet handled" %
                                   self.proj_biaxial_ellipsoid)
        
        if self.tm_meridian_scaling != MISSING_INT:
            if int(self.tm_meridian_scaling*1e6) == 999601:
                pass  # This is the expected value for British National Grid
            else:
                raise TranslationError("tm_meridian_scaling not yet handled: {}".
                                       format(self.tm_meridian_scaling))
        
        if self.horizontal_grid_type == 0:
            # "NG", means osgb grid.
            osgb_cs = iris.coord_systems.OSGB()
            x_coord = DimCoord(np.arange(self.num_cols) * self.column_step +
                               self.x_origin,
                               standard_name='projection_x_coordinate',
                               units='m', coord_system=osgb_cs)
            cube.add_dim_coord(x_coord, 1)
            if self.origin_corner == 0:  # top left
                y_coord = DimCoord(np.arange(self.num_rows)[::-1] *
                                   -self.row_step + self.y_origin,
                                   standard_name='projection_y_coordinate',
                                   units='m', coord_system=osgb_cs)
                cube.add_dim_coord(y_coord, 0)
            else:
                raise TranslationError("Corner {0} not yet implemented".
                                       format(self.origin_corner))
        else:
            raise TranslationError("Grid type %d not yet implemented" %
                                   self.horizontal_grid_type)
        
        # vertical
        if self.vertical_coord_type != MISSING_INT:
            if self.field_code == 73:  # Orography data.
                # We can find values in the vertical coord, such as 9999,
                # for orography fields. Don't make a vertical coord from these.
                pass
            elif self.vertical_coord_type == 0:
                if (self.reference_vertical_coord_type == MISSING_INT or
                    self.reference_vertical_coord == MISSING_FLOAT):
                    cube.add_aux_coord(DimCoord(self.vertical_coord,
                                        standard_name="height", units="m"))
                else:
                    raise TranslationError("Bounded vertical not yet implemented")
            elif self.vertical_coord_type == 1:
                if (self.reference_vertical_coord_type == MISSING_INT or
                    self.reference_vertical_coord == MISSING_FLOAT):
                    cube.add_aux_coord(DimCoord(self.vertical_coord,
                                        standard_name="altitude", units="m"))
                else:
                    raise TranslationError("Bounded vertical not yet implemented")
            else:
                raise TranslationError("Vertical coord type %d not yet handled" %
                                       self.vertical_coord_type)
    
        # add other stuff, if present
        ensemble_member = getattr(self, "ensemble_member")
        if ensemble_member != MISSING_INT:
            cube.add_aux_coord(DimCoord(ensemble_member, "realization"))
            
        def add_attr(name):
            value = getattr(self, name)
            if value not in [MISSING_INT, MISSING_FLOAT]:
                cube.attributes[name] = value

        add_attr("nimrod_version")
        add_attr("field_code")
        add_attr("num_model_levels")
        add_attr("sat_calib")
        add_attr("sat_space_count")
        add_attr("ducting_index")
        add_attr("elevation_angle")
        add_attr("radar_num")
        add_attr("radars_bitmask")
        add_attr("more_radars_bitmask")
        add_attr("clutter_map_num")
        add_attr("calibration_type")
        add_attr("bright_band_height")
        add_attr("bright_band_intensity")
        add_attr("bright_band_test1")
        add_attr("bright_band_test2")
        add_attr("infill_flag")
        add_attr("stop_elevation")
        add_attr("sensor_id")
        add_attr("meteosat_id")
        add_attr("alphas_available")
    
        cube.attributes["source"] = self.source.strip()
        
        return cube


def load_cubes(filenames, callback=None):
    """
    Loads cubes from a list of NIMROD filenames.
    
    Args:
    
    * filenames - list of NIMROD filenames to load
    
    Kwargs:
    
    * callback - a function which can be passed on to :func:`iris.io.run_callback`
    
    .. note:: 
        The resultant cubes may not be in the same order as in the files.

    """
    if isinstance(filenames, basestring):
        filenames = [filenames]

    for filename in filenames:
        for path in glob.glob(filename):
            with open(path, "rb") as infile:
                try:
                    field = NimrodField(infile)
                except struct.error:
                    # End of file. Move on to the next file.
                    continue

                cube = field.to_cube()

            # Were we given a callback?
            if callback is not None:
                cube = iris.io.run_callback(callback, cube, field, filename)
                if cube is None:
                    continue

            yield cube

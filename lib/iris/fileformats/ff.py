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
"""
Provides UK Met Office Fields File (FF) format specific capabilities.

"""

import os
import warnings

import numpy as np

import iris.config
import iris.fileformats.manager
import pp


FF_HEADER_DEPTH = 256  # in words (64-bit)
FF_WORD_DEPTH = 8      # in bytes

# UM marker to signify empty lookup table entry.
_FF_LOOKUP_TABLE_TERMINATE = -99

# UM FieldsFile fixed length header names and positions.
UM_FIXED_LENGTH_HEADER = [
        ('data_set_format_version',    (1, )),
        ('sub_model',                  (2, )),
        ('vert_coord_type',            (3, )),
        ('horiz_grid_type',            (4, )),
        ('dataset_type',               (5, )),
        ('run_identifier',             (6, )),
        ('experiment_number',          (7, )),
        ('calendar',                   (8, )),
        ('grid_staggering',            (9, )),
        ('time_type',                  (10, )),
        ('projection_number',          (11, )),
        ('model_version',              (12, )),
        ('obs_file_type',              (14, )),
        ('last_fieldop_type',          (15, )),
        ('first_validity_time',        (21, 22, 23, 24, 25, 26, 27, )),
        ('last_validity_time',         (28, 29, 30, 31, 32, 33, 34, )),
        ('misc_validity_time',         (35, 36, 37, 38, 39, 40, 41, )),
        ('integer_constants',          (100, 101, )),
        ('real_constants',             (105, 106, )),
        ('level_dependent_constants',  (110, 111, 112, )),
        ('row_dependent_constants',    (115, 116, 117, )),
        ('column_dependent_constants', (120, 121, 122, )),
        ('fields_of_constants',        (125, 126, 127, )),
        ('extra_constants',            (130, 131, )),
        ('temp_historyfile',           (135, 136, )),
        ('compressed_field_index1',    (140, 141, )), 
        ('compressed_field_index2',    (142, 143, )),
        ('compressed_field_index3',    (144, 145, )),
        ('lookup_table',               (150, 151, 152, )),
        ('total_prognostic_fields',    (153, )),
        ('data',                       (160, 161, 162, )), ]

# Offset value to convert from UM_FIXED_LENGTH_HEADER positions to FF_HEADER offsets.
UM_TO_FF_HEADER_OFFSET = 1
# Offset the UM_FIXED_LENGTH_HEADER positions to FF_HEADER offsets.
FF_HEADER = [(name, tuple([position - UM_TO_FF_HEADER_OFFSET for position in positions])) for name, positions in UM_FIXED_LENGTH_HEADER]

# UM marker to signify a null pointer address.
_FF_HEADER_POINTER_NULL = 0
# UM FieldsFile fixed length header pointer names.
_FF_HEADER_POINTERS = [
        'integer_constants',
        'real_constants',
        'level_dependent_constants',
        'row_dependent_constants',
        'column_dependent_constants',
        'fields_of_constants',
        'extra_constants',
        'temp_historyfile',
        'compressed_field_index1',
        'compressed_field_index2',
        'compressed_field_index3',
        'lookup_table',
        'data', ]


class FFHeader(object):
    """A class to represent the FIXED_LENGTH_HEADER section of a FieldsFile."""
    
    def __init__(self, filename):
        """
        Create a FieldsFile header instance by reading the FIXED_LENGTH_HEADER section of the FieldsFile.
        
        Args:
        
        * filename (string):
            Specify the name of the FieldsFile.
            
        Returns:
            FFHeader object.
            
        """
        
        self.ff_filename = filename
        '''File name of the FieldsFile.'''
        # Read the FF header data
        ff_file = open(filename, 'rb')
        header_data = np.fromfile(ff_file, dtype='>i8', count=FF_HEADER_DEPTH)  # 64-bit words.
        header_data = tuple(header_data)
        # Create FF instance attributes
        for name, offsets in FF_HEADER:
            if len(offsets) == 1:
                value = header_data[offsets[0]]
            else:
                value = header_data[offsets[0]:offsets[-1]+1]
            setattr(self, name, value)
        ff_file.close()

    def __repr__(self):
        attribute_offset = dict([(name, offsets[0]) for name, offsets in FF_HEADER])
        attribute_name_value = [(name, getattr(self, name)) for name, offsets in FF_HEADER if getattr(self, name, None) is not None]
        attributes = sorted(attribute_name_value, key=lambda name_value: attribute_offset[name_value[0]])
        return 'FF Header:' + ''.join(['\n    %s:  %s' % (name, value) for name, value in attributes])

    def valid(self, name):
        """
        Determine whether the FieldsFile FIXED_LENGTH_HEADER pointer attribute has a valid FieldsFile address.
        
        Args:
        
        * name (string):
            Specify the name of the FIXED_LENGTH_HEADER attribute.
            
        Returns:
            Boolean.
        
        """
        
        if name in _FF_HEADER_POINTERS:
            value = getattr(self, name)[0] != _FF_HEADER_POINTER_NULL
        else:
            raise AttributeError("'%s' object does not have pointer attribute '%s'" % (self.__class__.__name__, name))
        return value

    def address(self, name):
        """
        Return the byte address of the FieldsFile FIXED_LENGTH_HEADER pointer attribute.
        
        Args:
        
        * name (string):
            Specify the name of the FIXED_LENGTH_HEADER attribute.
            
        Returns:
            int.
        
        """
        
        if name in _FF_HEADER_POINTERS:
            value = getattr(self, name)[0] * FF_WORD_DEPTH
        else:
            raise AttributeError("'%s' object does not have pointer attribute '%s'" % (self.__class__.__name__, name))
        return value
    
    def shape(self, name):
        """
        Return the dimension shape of the FieldsFile FIXED_LENGTH_HEADER pointer attribute.
        
        Args:
        
        * name (string):
            Specify the name of the FIXED_LENGTH_HEADER attribute.
            
        Returns:
            Dimension tuple.
        
        """
        
        if name in _FF_HEADER_POINTERS:
            value = getattr(self, name)[1:]
        else:
            raise AttributeError("'%s' object does not have pointer address '%s'" % (self.__class_.__name__, name))
        return value


class FF2PP(object):
    """A class to extract the individual PPFields from within a FieldsFile."""

    def __init__(self, filename, read_data=False):
        """
        Create a FieldsFile to Post Process instance that returns a generator
        of PPFields contained within the FieldsFile.
        
        Args:
        
        * filename (string):
            Specify the name of the FieldsFile.
            
        Kwargs:
        
        * read_data (boolean):
            Specify whether to read the associated PPField data within the FieldsFile.
            Default value is False.
            
        Returns:
            PPField generator.
        
        For example::
    
            >>> for field in ff.FF2PP(filename):
            ...     print field
            
        """
        
        self._ff_header = FFHeader(filename)
        self._filename = filename
        self._read_data = read_data
        
    def _extract_field(self):
        # FF table pointer initialisation based on FF LOOKUP table configuration. 
        table_index, table_entry_depth, table_count = self._ff_header.lookup_table
        table_offset = (table_index - 1) * FF_WORD_DEPTH       # in bytes
        table_entry_depth = table_entry_depth * FF_WORD_DEPTH  # in bytes
        # Open the FF for processing.
        ff_file = open(self._ff_header.ff_filename, 'rb')
        ff_file_seek = ff_file.seek
        # Process each FF LOOKUP table entry.
        while table_count:
            table_count -= 1
            # Move file pointer to the start of the current FF LOOKUP table entry.
            ff_file_seek(table_offset, os.SEEK_SET)
            # Read the current PP header entry from the FF LOOKUP table.
            pp_header_integers = np.fromfile(ff_file, dtype='>i8', count=pp.NUM_LONG_HEADERS)  # 64-bit words.
            pp_header_floats = np.fromfile(ff_file, dtype='>f8', count=pp.NUM_FLOAT_HEADERS)   # 64-bit words.
            pp_header_data = tuple(pp_header_integers) + tuple(pp_header_floats)
            # Check whether the current FF LOOKUP table entry is valid.
            if pp_header_data[0] == _FF_LOOKUP_TABLE_TERMINATE:
                # There are no more FF LOOKUP table entries to read. 
                break
            # Calculate next FF LOOKUP table entry.
            table_offset += table_entry_depth
            # Construct a PPField object and populate using the pp_header_data
            # read from the current FF LOOKUP table.
            # (The PPField sub-class will depend on the header release number.)
            pp_field = pp.make_pp_field(pp_header_data)
            # Calculate file pointer address for the start of the associated PP header data. 
            data_offset = pp_field.lbegin * FF_WORD_DEPTH
            # Determine PP field payload depth.
            pp_data_extra_depth = pp_field.lbext
            if pp_field.lbpack:
                # Convert PP field LBNREC, representing a count in 64-bit words,
                # into its associated count in bytes.
                pp_data_depth = ((pp_field.lbnrec * 2) - 1) * pp.PP_WORD_DEPTH  # in bytes
            else:
                pp_data_depth = (pp_field.lblrec - pp_data_extra_depth) * pp.PP_WORD_DEPTH  # in bytes 
            
            # Determine PP field payload datatype.
            pp_data_type = pp.LBUSER_DTYPE_LOOKUP.get(pp_field.lbuser[0], pp.LBUSER_DTYPE_LOOKUP['default'])

            # Determine PP field data shape.
            pp_data_shape = (pp_field.lbrow, pp_field.lbnpt)
            # Determine whether to read the associated PP field data.
            if self._read_data:
                # Move file pointer to the start of the current PP field data.
                ff_file_seek(data_offset, os.SEEK_SET)
                # Get the PP field data.
                data = pp_field.read_data(ff_file, pp_data_depth, pp_data_shape, pp_data_type)
                pp_field._data = data
                pp_field._data_manager = None
            else:
                pp_field._data = np.array(pp.PPDataProxy(self._ff_header.ff_filename, data_offset, pp_data_depth, pp_field.lbpack))
                pp_field._data_manager = iris.fileformats.manager.DataManager(pp_data_shape, pp_data_type, pp_field.bmdi)
            yield pp_field
        ff_file.close()
        return
        
    def __iter__(self):
        return self._extract_field()


def load_cubes(filenames, callback):
    """
    Loads cubes from a list of fields files filenames.
    
    Args:
    
    * filenames - list of fields files filenames to load
    
    Kwargs:
    
    * callback - a function which can be passed on to :func:`iris.io.run_callback`
    
    .. note:: 
        The resultant cubes may not be in the order that they are in the file (order 
        is not preserved when there is a field with orography references).
         
    """
    return pp._load_cubes_variable_loader(filenames, callback, FF2PP)

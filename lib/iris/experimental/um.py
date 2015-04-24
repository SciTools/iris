# (C) British Crown Copyright 2014 - 2015, Met Office
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
Low level support for UM FieldsFile variants.

"""

from __future__ import (absolute_import, division, print_function)

from contextlib import contextmanager
import os
import os.path
import tempfile

import numpy as np

# Borrow some definitions...
from iris.fileformats.ff import _FF_HEADER_POINTERS, FF_HEADER as _FF_HEADER
from iris.fileformats.pp import _header_defn

try:
    import mo_pack
except ImportError:
    mo_pack = None

DEFAULT_WORD_SIZE = 8  # In bytes.


def _make_getter(attr_name, index):
    if isinstance(index, slice):
        def getter(self):
            return tuple(getattr(self, attr_name)[index])
    else:
        def getter(self):
            return getattr(self, attr_name)[index]
    return getter


def _make_setter(attr_name, index):
    def setter(self, value):
        getattr(self, attr_name)[index] = value
    return setter


class _HeaderMetaclass(type):
    """
    Adds convenience get/set properties to the target class
    corresponding to single-valued entries in _FF_HEADER.
        e.g. FixedLengthHeader.sub_model

    Also adds "<name>_start" and "<name>_shape" convenience properties
    corresponding to the "pointer" entries in _FF_HEADER, with the
    exception of the lookup and data components.
        e.g. FixedLengthHeader.integer_constants_start
             FixedLengthHeader.integer_constants_shape

    """
    def __new__(metacls, classname, bases, class_dict):
        def add_property(name, index):
            class_dict[name] = property(_make_getter('_integers', index),
                                        _make_setter('_integers', index))

        for name, offsets in _FF_HEADER:
            if len(offsets) == 1:
                add_property(name, offsets[0])
            elif name in _FF_HEADER_POINTERS:
                if name == 'lookup_table':
                    # Rename for consistency with UM documentation paper F3
                    name = 'lookup'
                elif name == 'data':
                    # Bug fix - data is only 1-dimensional.
                    offsets = offsets[:-1]

                add_property(name + '_start', offsets[0])
                first_offset = offsets[1]
                last_offset = offsets[-1] + 1
                add_property(name + '_shape', slice(first_offset, last_offset))
            else:
                # The remaining multi-value items are 'first_validity_time',
                # 'last_validity_time', and 'misc_validity_time'.
                # But, from the wider perspective of FieldsFile variants
                # these names do not make sense - so we skip them.
                pass

        # Complement to 1-dimensional data bug fix
        add_property('max_length', 161)

        return super(_HeaderMetaclass, metacls).__new__(metacls, classname,
                                                        bases, class_dict)


class FixedLengthHeader(object):
    """
    Represents the FIXED_LENGTH_HEADER component of a UM FieldsFile
    variant.

    Access to simple header items is provided via named attributes,
    e.g. fixed_length_header.sub_model. Other header items can be
    accessed via the :attr:`raw` attribute which provides a simple array
    view of the header.

    """

    __metaclass__ = _HeaderMetaclass

    NUM_WORDS = 256
    IMDI = -32768

    @classmethod
    def empty(cls, word_size=DEFAULT_WORD_SIZE):
        integers = np.empty(cls.NUM_WORDS, dtype='>i{}'.format(word_size))
        integers[:] = cls.IMDI
        return cls(integers)

    @classmethod
    def from_file(cls, source, word_size=DEFAULT_WORD_SIZE):
        """
        Create a FixedLengthHeader from a file-like object.

        Args:

        * source:
            The file-like object to read from.

        Kwargs:

        * word_size:
            The number of bytes in each word of the header.

        """
        integers = np.fromfile(source, dtype='>i{}'.format(word_size),
                               count=cls.NUM_WORDS)
        return cls(integers)

    def __init__(self, integers):
        """
        Create a FixedLengthHeader from the given sequence of integer
        values.

        """
        if len(integers) != self.NUM_WORDS:
            raise ValueError('Incorrect number of words - given {} but should '
                             'be {}.'.format(len(integers), self.NUM_WORDS))
        self._integers = np.asarray(integers)

    def __eq__(self, other):
        try:
            eq = np.all(self._integers == other._integers)
        except AttributeError:
            eq = NotImplemented
        return eq

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    @property
    def raw(self):
        return self._integers.view()


# The number of integer header items.
_NUM_FIELD_INTS = 45


class _FieldMetaclass(type):
    """
    Adds human-readable get/set properties derived from a _HEADER_DEFN
    attribute on the target class.
        e.g. field.lbproc, field.blev

    "Array-style" header items, such as LBUSER, result in multiple
    single-valued properties with a one-based numeric suffix.
        e.g. field.lbuser1, field.lbuser7

    """
    def __new__(metacls, classname, bases, class_dict):
        defn = class_dict.get('_HEADER_DEFN')
        if defn is not None:
            for name, indices in defn:
                if len(indices) == 1:
                    names = [name]
                else:
                    names = [name + str(i + 1) for i, _ in enumerate(indices)]
                for name, index in zip(names, indices):
                    if index < _NUM_FIELD_INTS:
                        attr_name = 'int_headers'
                    else:
                        attr_name = 'real_headers'
                        index -= _NUM_FIELD_INTS
                    class_dict[name] = property(_make_getter(attr_name, index),
                                                _make_setter(attr_name, index))
        return super(_FieldMetaclass, metacls).__new__(metacls, classname,
                                                       bases, class_dict)


# Helper function to ensure an array is big-endian and of the
# correct dtype kind and word size for writing to file.
def normalise(values, kind, word_size):
    return values.astype('>{}{}'.format(kind, word_size))


class Field(object):
    """
    Represents a single entry in the LOOKUP component and its
    corresponding section of the DATA component.

    """
    __metaclass__ = _FieldMetaclass

    #: Zero-based index for lblrec.
    LBLREC_OFFSET = 14
    #: Zero-based index for lbrel.
    LBREL_OFFSET = 21
    #: Zero-based index for lbegin.
    LBEGIN_OFFSET = 28
    #: Zero-based index for lbnrec.
    LBNREC_OFFSET = 29

    def __init__(self, int_headers, real_headers, data_provider):
        """
        Create a Field from the integer headers, the floating-point
        headers, and an object which provides access to the
        corresponding data.

        Args:

        * int_headers:
            A sequence of integer header values.
        * real_headers:
            A sequence of floating-point header values.
        * data_provider:
            Either, an object with a `read_data()` method which will
            provide the corresponding values from the DATA component,
            or a NumPy array, or None.

        """
        #: A NumPy array of integer header values.
        self.int_headers = np.asarray(int_headers)
        #: A NumPy array of floating-point header values.
        self.real_headers = np.asarray(real_headers)
        self._data_provider = data_provider

    def __eq__(self, other):
        try:
            eq = (np.all(self.int_headers == other.int_headers) and
                  np.all(self.real_headers == other.real_headers) and
                  np.all(self.get_data() == other.get_data()))
        except AttributeError:
            eq = NotImplemented
        return eq

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def num_values(self):
        """
        Return the number of values defined by this header.

        """
        return len(self.int_headers) + len(self.real_headers)

    def get_data(self):
        """
        Return a NumPy array containing the data for this field.

        Data packed with the WGDOS archive method will be unpacked and
        returned as int/float data as appropriate.

        """
        data = None
        if isinstance(self._data_provider, np.ndarray):
            data = self._data_provider
        elif self._data_provider is not None:
            data = self._data_provider.read_data()
        return data

    def _get_raw_payload_bytes(self):
        """
        Return a buffer containing the raw bytes of the data payload.

        The field data must be a deferred-data provider, not an array.
        Typically, that means a deferred data reference to an existing file.
        This enables us to handle packed data without interpreting it.

        """
        return self._data_provider._read_raw_payload_bytes()

    def set_data(self, data):
        """
        Set the data payload for this field.

        * data:
            Either, an object with a `read_data()` method which will
            provide the corresponding values from the DATA component,
            or a NumPy array, or None.

        """
        self._data_provider = data

    def _can_copy_deferred_data(self):
        """
        Return whether a field's payload contains deferred data that can be
        reused unmodified, for the specified output packing format.

        This is true if our data is a deferred provider, with a lookup from
        the original file which specifies exactly the same packing that the
        field wants.

        """
        # Check that the original data payload has not been replaced by plain
        # array data.
        compatible = hasattr(self._data_provider, 'lookup_entry')
        if compatible:
            src_lbpack = self._data_provider.lookup_entry.lbpack
            src_bacc = self._data_provider.lookup_entry.bacc

            # The packing words are compatible if nothing else is different.
            compatible = (self.lbpack == src_lbpack and
                          self.bacc == src_bacc)

        return compatible

    def _write_payload(self, output_file, word_size, words_per_sector):
        """
        Write the field data payload to the provided file.

        This includes handling supported packing types, and updating the field
        itself appropriately.

        """
        self.lbegin = output_file.tell() / word_size
        if self._can_copy_deferred_data():
            # The original, unread file data is encoded as wanted, so pass it
            # through unchanged.  In this case, we should also leave the lookup
            # controls unchanged
            # -- i.e. do not recalculate LBLREC and LBNREC.
            output_file.write(self._get_raw_payload_bytes())
        else:
            # Output in the format specified by LBPACK.
            if self.lbpack in (0, 2000, 3000):
                # Write unpacked data -- in any of the supported word types,
                # which are all equivalent.
                data = self.get_data()

                # When not compressing, preparation is just to fix the data
                # wordlength and endian-ness.
                kind = {1: 'f', 2: 'i', 3: 'i'}.get(
                    self.lbuser1, data.dtype.kind)
                data = normalise(data, kind, word_size)

                # Write data.
                output_file.write(data)

                # Size is one word per point.
                n_data_words = data.size

            elif self.lbpack in (1, 2001, 3001):
                # Write WGDOS packed results.
                if not mo_pack:
                    msg = ('cannot pack data with lbpack={} :'
                           'WGDOS packing library "mo_pack" is not available.')
                    raise ValueError(msg.format(self.lbpack))

                data = self.get_data()
                wgdos_packed_bytes = mo_pack.pack_wgdos(
                    data.astype(np.float32), self.bacc, self.bmdi)

                # Write data.
                output_file.write(wgdos_packed_bytes)

                # Pad to a whole word boundary, if needed.
                n_data_bytes = len(wgdos_packed_bytes)
                n_data_words = n_data_bytes // word_size
                n_pad_bytes = \
                    n_data_bytes - (n_data_words * word_size)
                if n_pad_bytes > 0:
                    # Add padding bytes and increment the word length.
                    pad_bytes = np.zeros(n_pad_bytes, dtype='>i1')
                    output_file.write(pad_bytes)
                    n_data_words += 1
            else:
                # Unrecognised lbpack value.
                msg = ('Cannot save data with lbpack={} : '
                       'unsupported or unknown packing type.')
                raise ValueError(msg.format(self.lbpack))

            # Record the payload size in the lookup control words.
            # NOTE: this is *not* done on pass-through of an uninterpreted
            # payload.
            n_data_sectors = np.ceil(n_data_words / words_per_sector)
            data_fullsectors_words = n_data_sectors * words_per_sector
            self.lblrec = n_data_words
            self.lbnrec = data_fullsectors_words


class Field2(Field):
    """
    Represents an entry from the LOOKUP component with a header release
    number of 2.

    """
    _HEADER_DEFN = _header_defn(2)


class Field3(Field):
    """
    Represents an entry from the LOOKUP component with a header release
    number of 3.

    """
    _HEADER_DEFN = _header_defn(3)


# Maps lbrel to a Field class.
_FIELD_CLASSES = {2: Field2, 3: Field3}


# Maps word size and then lbuser1 (i.e. the field's data type) to a dtype.
_DATA_DTYPES = {4: {1: '>f4', 2: '>i4', 3: '>i4'},
                8: {1: '>f8', 2: '>i8', 3: '>i8'}}


_CRAY32_SIZE = 4
_WGDOS_SIZE = 4


class _DataProvider(object):
    def __init__(self, sourcefile, filename, lookup, offset, word_size):
        """
        Create a provider that can load a lookup's data.

        Args:
        * sourcefile: (file)
            An open file.  This is essentially a shortcut, to avoid having to
            always open a file. If it is *not* open when get_data is called,
            a temporary file will be opened for 'filename'.
        * filename: (string)
            Path to the containing file.
        * lookup: (Field)
            The lookup which the provider relates to.  This encapsulates the
            original encoding information in the input file.
        * offset: (int)
            The data offset in the file (bytes).
        * word_size: (int)
            Number of bytes in a header word -- either 4 or 8.

        """
        self.source = sourcefile
        self.reopen_path = filename
        self.offset = offset
        self.word_size = word_size
        self.lookup_entry = lookup

    @contextmanager
    def _with_source(self):
        # Context manager to temporarily reopen the sourcefile if the original
        # provided at create time has been closed.
        reopen_required = self.source.closed
        close_required = False

        try:
            if reopen_required:
                self.source = open(self.reopen_path)
                close_required = True
            yield self.source
        finally:
            if close_required:
                self.source.close()

    def _read_raw_payload_bytes(self):
        # Return the raw data payload, as an array of bytes.
        # This is independent of the content type.
        field = self.lookup_entry
        with self._with_source():
            self.source.seek(self.offset)
            data_size = ((field.lbnrec * 2) - 1) * _WGDOS_SIZE
            # This size calculation seems rather questionable, but derives from
            # a very long code legacy, so appeal to a "sleeping dogs" policy.
            data_bytes = self.source.read(data_size)
        return data_bytes


class _NormalDataProvider(_DataProvider):
    """
    Provides access to a simple 2-dimensional array of data, corresponding
    to the data payload for a standard FieldsFile LOOKUP entry.

    """
    def read_data(self):
        field = self.lookup_entry
        with self._with_source():
            self.source.seek(self.offset)
            lbpack = field.lbpack
            # Ensure lbpack.n4 (number format) is: native, CRAY, or IEEE.
            format = (lbpack // 1000) % 10
            if format not in (0, 2, 3):
                msg = 'Unsupported number format: {}'
                raise ValueError(msg.format(format))
            lbpack = lbpack % 1000
            # NB. This comparison includes checking for the absence of any
            # compression.
            if lbpack == 0 or lbpack == 2:
                if lbpack == 0:
                    word_size = self.word_size
                else:
                    word_size = _CRAY32_SIZE
                dtype = _DATA_DTYPES[word_size][field.lbuser1]
                rows = field.lbrow
                cols = field.lbnpt
                # The data is stored in rows, so with the shape (rows, cols)
                # we don't need to invoke Fortran order.
                data = np.fromfile(self.source, dtype, count=rows * cols)
                data = data.reshape(rows, cols)
            elif lbpack == 1:
                if mo_pack is None:
                    msg = 'mo_pack is required to read WGDOS packed data'
                    raise ValueError(msg)

                data_bytes = self._read_raw_payload_bytes()
                data = mo_pack.unpack_wgdos(data_bytes, field.lbrow,
                                            field.lbnpt, field.bmdi)
            else:
                raise ValueError('Unsupported lbpack: {}'.format(field.lbpack))
        return data


class _BoundaryDataProvider(_DataProvider):
    """
    Provides access to the data payload corresponding to a LOOKUP entry
    in a lateral boundary condition FieldsFile variant.

    The data will be 2-dimensional, with the first dimension expressing
    the number of vertical levels and the second dimension being an
    "unrolled" version of all the boundary points.

    """
    def read_data(self):
        field = self.lookup_entry
        with self._with_source():
            self.source.seek(self.offset)
            lbpack = field.lbpack
            # Ensure lbpack.n4 (number format) is: native, CRAY, or IEEE.
            format = (lbpack // 1000) % 10
            if format not in (0, 2, 3):
                msg = 'Unsupported number format: {}'
                raise ValueError(msg.format(format))
            lbpack = lbpack % 1000
            if lbpack == 0 or lbpack == 2:
                if lbpack == 0:
                    word_size = self.word_size
                else:
                    word_size = _CRAY32_SIZE
                dtype = _DATA_DTYPES[word_size][field.lbuser1]
                data = np.fromfile(self.source, dtype, count=field.lblrec)
                data = data.reshape(field.lbhem - 100, -1)
            else:
                msg = 'Unsupported lbpack for LBC: {}'.format(field.lbpack)
                raise ValueError(msg)
        return data


class FieldsFileVariant(object):
    """
    Represents a single a file containing UM FieldsFile variant data.

    """

    _COMPONENTS = (('integer_constants', 'i'),
                   ('real_constants', 'f'),
                   ('level_dependent_constants', 'f'),
                   ('row_dependent_constants', 'f'),
                   ('column_dependent_constants', 'f'),
                   ('fields_of_constants', 'f'),
                   ('extra_constants', 'f'),
                   ('temp_historyfile', 'i'),
                   ('compressed_field_index1', 'i'),
                   ('compressed_field_index2', 'i'),
                   ('compressed_field_index3', 'i'))

    _WORDS_PER_SECTOR = 2048

    class _Mode(object):
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

    #: The file will be opened for read-only access.
    READ_MODE = _Mode('READ_MODE')
    #: The file will be opened for update.
    UPDATE_MODE = _Mode('UPDATE_MODE')
    #: The file will be created, overwriting the file if it already
    #: exists.
    CREATE_MODE = _Mode('CREATE_MODE')

    _MODE_MAPPING = {READ_MODE: 'rb', UPDATE_MODE: 'r+b', CREATE_MODE: 'wb'}

    def __init__(self, filename, mode=READ_MODE, word_size=DEFAULT_WORD_SIZE):
        """
        Opens the given filename as a UM FieldsFile variant.

        Args:

        * filename:
            The name of the file containing the UM FieldsFile variant.

        Kwargs:

        * mode:
            The file access mode: `READ_MODE` for read-only;
            `UPDATE_MODE` for amending; `CREATE_MODE` for creating a new
            file.

        * word_size:
            The number of byte in each word.

        """
        if mode not in self._MODE_MAPPING:
            raise ValueError('Invalid access mode: {}'.format(mode))

        self._filename = filename
        self._mode = mode
        self._word_size = word_size

        source_mode = self._MODE_MAPPING[mode]
        self._source = source = open(filename, source_mode)

        if mode is self.CREATE_MODE:
            header = FixedLengthHeader.empty(word_size)
        else:
            header = FixedLengthHeader.from_file(source, word_size)
        self.fixed_length_header = header

        def constants(name, dtype):
            start = getattr(self.fixed_length_header, name + '_start')
            if start > 0:
                source.seek((start - 1) * word_size)
                shape = getattr(self.fixed_length_header, name + '_shape')
                values = np.fromfile(source, dtype, count=np.product(shape))
                if len(shape) > 1:
                    values = values.reshape(shape, order='F')
            else:
                values = None
            return values

        for name, kind in self._COMPONENTS:
            dtype = '>{}{}'.format(kind, word_size)
            setattr(self, name, constants(name, dtype))

        int_dtype = '>i{}'.format(word_size)
        real_dtype = '>f{}'.format(word_size)

        if self.fixed_length_header.dataset_type == 5:
            data_class = _BoundaryDataProvider
        else:
            data_class = _NormalDataProvider

        lookup = constants('lookup', int_dtype)
        fields = []
        if lookup is not None:
            is_model_dump = lookup[Field.LBNREC_OFFSET, 0] == 0
            if is_model_dump:
                # A model dump has no direct addressing - only relative,
                # so we need to update the offset as we create each
                # Field.
                running_offset = ((self.fixed_length_header.data_start - 1) *
                                  word_size)

            for raw_headers in lookup.T:
                ints = raw_headers[:_NUM_FIELD_INTS]
                reals = raw_headers[_NUM_FIELD_INTS:].view(real_dtype)
                field_class = _FIELD_CLASSES.get(ints[Field.LBREL_OFFSET],
                                                 Field)
                if raw_headers[0] == -99:
                    data_provider = None
                else:
                    if is_model_dump:
                        offset = running_offset
                    else:
                        offset = raw_headers[Field.LBEGIN_OFFSET] * word_size
                    # Make a *copy* of field lookup data, as it was in the
                    # untouched original file, as a context for data loading.
                    # (N.B. most importantly, includes the original LBPACK)
                    lookup_reference = field_class(ints.copy(), reals.copy(),
                                                   None)
                    # Make a "provider" that can fetch the data on request.
                    data_provider = data_class(source, filename,
                                               lookup_reference,
                                               offset, word_size)
                field = field_class(ints, reals, data_provider)
                fields.append(field)
                if is_model_dump:
                    running_offset += (raw_headers[Field.LBLREC_OFFSET] *
                                       word_size)
        self.fields = fields

    def __del__(self):
        if hasattr(self, '_source'):
            self.close()

    def __str__(self):
        dataset_type = self.fixed_length_header.dataset_type
        items = ['dataset_type={}'.format(dataset_type)]
        for name, kind in self._COMPONENTS:
            value = getattr(self, name)
            if value is not None:
                items.append('{}={}'.format(name, value.shape))
        if self.fields:
            items.append('fields={}'.format(len(self.fields)))
        return '<FieldsFileVariant: {}>'.format(', '.join(items))

    def __repr__(self):
        fmt = '<FieldsFileVariant: dataset_type={}>'
        return fmt.format(self.fixed_length_header.dataset_type)

    @property
    def filename(self):
        return self._filename

    @property
    def mode(self):
        return self._mode

    def _update_fixed_length_header(self):
        # Set the start locations and dimension lengths(*) in the fixed
        # length header.
        # *) Except for the DATA component where we only determine
        #    the start location.
        header = self.fixed_length_header
        word_number = header.NUM_WORDS + 1  # Numbered from 1.

        # Start by dealing with the normal components.
        for name, kind in self._COMPONENTS:
            value = getattr(self, name)
            start_name = name + '_start'
            shape_name = name + '_shape'
            if value is None:
                setattr(header, start_name, header.IMDI)
                setattr(header, shape_name, header.IMDI)
            else:
                setattr(header, start_name, word_number)
                setattr(header, shape_name, value.shape)
                word_number += value.size

        # Now deal with the LOOKUP and DATA components.
        if self.fields:
            header.lookup_start = word_number
            lookup_lengths = {field.num_values() for field in self.fields}
            if len(lookup_lengths) != 1:
                msg = 'Inconsistent lookup header lengths - {}'
                raise ValueError(msg.format(lookup_lengths))
            lookup_length = lookup_lengths.pop()
            n_fields = len(self.fields)
            header.lookup_shape = (lookup_length, n_fields)

            # make space for the lookup
            word_number += lookup_length * n_fields
            # Round up to the nearest whole number of "sectors".
            offset = word_number - 1
            offset -= offset % -self._WORDS_PER_SECTOR
            header.data_start = offset + 1
        else:
            header.lookup_start = header.IMDI
            header.lookup_shape = header.IMDI
            header.data_start = header.IMDI
            header.data_shape = header.IMDI

    def _write_new(self, output_file):
        self._update_fixed_length_header()

        # Skip the fixed length header. We'll write it at the end
        # once we know how big the DATA component needs to be.
        header = self.fixed_length_header
        word_size = self._word_size
        words_per_sector = self._WORDS_PER_SECTOR
        output_file.seek(header.NUM_WORDS * self._word_size)

        # Write all the normal components which have a value.
        for name, kind in self._COMPONENTS:
            values = getattr(self, name)
            if values is not None:
                output_file.write(np.ravel(normalise(values, kind, word_size),
                                           order='F'))

        if self.fields:
            # Skip the LOOKUP component and write the DATA component.
            # We need to adjust the LOOKUP headers to match where
            # the DATA payloads end up, so to avoid repeatedly
            # seeking backwards and forwards it makes sense to wait
            # until we've adjusted them all and write them out in
            # one go.
            output_file.seek((header.data_start - 1) * self._word_size)

            for field in self.fields:
                if hasattr(field, '_HEADER_DEFN'):
                    # Output 'recognised' lookup types (not blank entries).
                    field._write_payload(output_file,
                                         word_size, words_per_sector)

                    # Pad out the data section to a whole number of sectors.
                    sector_size = words_per_sector * word_size
                    overrun = output_file.tell() % sector_size
                    if overrun != 0:
                        padding = np.zeros(sector_size - overrun, 'i1')
                        output_file.write(padding)

            # Update the fixed length header to reflect the extent
            # of the DATA component.
            dataset_type = self.fixed_length_header.dataset_type
            if dataset_type == 5:
                header.data_shape = 0
            else:
                header.data_shape = ((output_file.tell() // word_size) -
                                     header.data_start + 1)

            # Go back and write the LOOKUP component.
            output_file.seek((header.lookup_start - 1) * word_size)
            for field in self.fields:
                output_file.write(
                    normalise(field.int_headers, 'i', word_size))
                output_file.write(
                    normalise(field.real_headers, 'f', word_size))

        # Write the fixed length header - now that we know how big
        # the DATA component was.
        output_file.seek(0)
        output_file.write(
            normalise(self.fixed_length_header.raw, 'i', word_size))

    def close(self):
        """
        Write out any pending changes, and close the underlying file.

        If the file was opened for update or creation then the current
        state of the fixed length header, the constant components (e.g.
        integer_constants, level_dependent_constants), and the list of
        fields are written to the file before closing. The process of
        writing to the file also updates the values in the fixed length
        header and fields which relate to layout within the file. For
        example, `integer_constants_start` and `integer_constants_shape`
        within the fixed length header, and the `lbegin` and `lbnrec`
        elements within the fields.

        If the file was opened in read mode then no changes will be
        made.

        After calling `close()` any subsequent modifications to any of
        the attributes will have no effect on the underlying file.

        Calling `close()` more than once is allowed, but only the first
        call will have any effect.

        .. note::

            On output, each field's data is encoded according to the LBPACK
            and BACC words in the field.  A field data array defined using
            :meth:`Field.set_data` can *only* be written in an "unpacked"
            form, corresponding to LBACK=0 (or the equivalent 2000 / 3000).
            However, data from the input file can be saved in its original
            packed form, as long as the data, LBPACK and BACC remain unchanged.

        """
        if not self._source.closed:
            try:
                if self.mode in (self.UPDATE_MODE, self.CREATE_MODE):
                    # For simplicity at this stage we always create a new
                    # file and rename it once complete.
                    # At some later stage we can optimise for in-place
                    # modifications, for example if only one of the integer
                    # constants has been modified.

                    src_dir = os.path.dirname(os.path.abspath(self.filename))
                    with tempfile.NamedTemporaryFile(dir=src_dir,
                                                     delete=False) as tmp_file:
                        self._write_new(tmp_file)
                    os.unlink(self.filename)
                    os.rename(tmp_file.name, self.filename)
            finally:
                self._source.close()

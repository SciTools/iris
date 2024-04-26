# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides UK Met Office Post Process (PP) format specific capabilities."""

from abc import ABCMeta, abstractmethod
import collections
from copy import deepcopy
import operator
import os
import re
import struct
import warnings

import cf_units
import cftime
import dask
import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_concrete_data, as_lazy_data, is_lazy_data
import iris.config
import iris.coord_systems
import iris.exceptions

# NOTE: this is for backwards-compatitibility *ONLY*
# We could simply remove it for v2.0 ?
from iris.fileformats._pp_lbproc_pairs import LBPROC_MAP as lbproc_map  # noqa: F401
from iris.fileformats._pp_lbproc_pairs import LBPROC_PAIRS  # noqa: F401
import iris.fileformats.pp_load_rules
from iris.fileformats.pp_save_rules import verify
import iris.fileformats.rules
import iris.warnings

try:
    import mo_pack
except ImportError:
    mo_pack = None


__all__ = [
    "EARTH_RADIUS",
    "PPField",
    "STASH",
    "as_fields",
    "load",
    "load_cubes",
    "load_pairs_from_fields",
    "save",
    "save_fields",
    "save_pairs_from_cube",
]


EARTH_RADIUS = 6371229.0


PP_HEADER_DEPTH = 256
PP_WORD_DEPTH = 4
NUM_LONG_HEADERS = 45
NUM_FLOAT_HEADERS = 19

# The header definition for header release 2.
#: A list of (header_name, position_in_header(tuple of)) pairs for
#: header release 2 - using the one-based UM/FORTRAN indexing convention.
UM_HEADER_2 = [
    ("lbyr", (1,)),
    ("lbmon", (2,)),
    ("lbdat", (3,)),
    ("lbhr", (4,)),
    ("lbmin", (5,)),
    ("lbday", (6,)),
    ("lbyrd", (7,)),
    ("lbmond", (8,)),
    ("lbdatd", (9,)),
    ("lbhrd", (10,)),
    ("lbmind", (11,)),
    ("lbdayd", (12,)),
    ("lbtim", (13,)),
    ("lbft", (14,)),
    ("lblrec", (15,)),
    ("lbcode", (16,)),
    ("lbhem", (17,)),
    ("lbrow", (18,)),
    ("lbnpt", (19,)),
    ("lbext", (20,)),
    ("lbpack", (21,)),
    ("lbrel", (22,)),
    ("lbfc", (23,)),
    ("lbcfc", (24,)),
    ("lbproc", (25,)),
    ("lbvc", (26,)),
    ("lbrvc", (27,)),
    ("lbexp", (28,)),
    ("lbegin", (29,)),
    ("lbnrec", (30,)),
    ("lbproj", (31,)),
    ("lbtyp", (32,)),
    ("lblev", (33,)),
    ("lbrsvd", (34, 35, 36, 37)),
    ("lbsrce", (38,)),
    ("lbuser", (39, 40, 41, 42, 43, 44, 45)),
    ("brsvd", (46, 47, 48, 49)),
    ("bdatum", (50,)),
    ("bacc", (51,)),
    ("blev", (52,)),
    ("brlev", (53,)),
    ("bhlev", (54,)),
    ("bhrlev", (55,)),
    ("bplat", (56,)),
    ("bplon", (57,)),
    ("bgor", (58,)),
    ("bzy", (59,)),
    ("bdy", (60,)),
    ("bzx", (61,)),
    ("bdx", (62,)),
    ("bmdi", (63,)),
    ("bmks", (64,)),
]

# The header definition for header release 3.
#: A list of (header_name, position_in_header(tuple of)) pairs for
#: header release 3 - using the one-based UM/FORTRAN indexing convention.
UM_HEADER_3 = [
    ("lbyr", (1,)),
    ("lbmon", (2,)),
    ("lbdat", (3,)),
    ("lbhr", (4,)),
    ("lbmin", (5,)),
    ("lbsec", (6,)),
    ("lbyrd", (7,)),
    ("lbmond", (8,)),
    ("lbdatd", (9,)),
    ("lbhrd", (10,)),
    ("lbmind", (11,)),
    ("lbsecd", (12,)),
    ("lbtim", (13,)),
    ("lbft", (14,)),
    ("lblrec", (15,)),
    ("lbcode", (16,)),
    ("lbhem", (17,)),
    ("lbrow", (18,)),
    ("lbnpt", (19,)),
    ("lbext", (20,)),
    ("lbpack", (21,)),
    ("lbrel", (22,)),
    ("lbfc", (23,)),
    ("lbcfc", (24,)),
    ("lbproc", (25,)),
    ("lbvc", (26,)),
    ("lbrvc", (27,)),
    ("lbexp", (28,)),
    ("lbegin", (29,)),
    ("lbnrec", (30,)),
    ("lbproj", (31,)),
    ("lbtyp", (32,)),
    ("lblev", (33,)),
    ("lbrsvd", (34, 35, 36, 37)),
    ("lbsrce", (38,)),
    ("lbuser", (39, 40, 41, 42, 43, 44, 45)),
    ("brsvd", (46, 47, 48, 49)),
    ("bdatum", (50,)),
    ("bacc", (51,)),
    ("blev", (52,)),
    ("brlev", (53,)),
    ("bhlev", (54,)),
    ("bhrlev", (55,)),
    ("bplat", (56,)),
    ("bplon", (57,)),
    ("bgor", (58,)),
    ("bzy", (59,)),
    ("bdy", (60,)),
    ("bzx", (61,)),
    ("bdx", (62,)),
    ("bmdi", (63,)),
    ("bmks", (64,)),
]

# A map from header-release-number to header definition
UM_HEADERS = {2: UM_HEADER_2, 3: UM_HEADER_3}

# Offset value to convert from UM_HEADER positions to PP_HEADER offsets.
UM_TO_PP_HEADER_OFFSET = 1

#: A dictionary mapping IB values to their names.
EXTRA_DATA = {
    1: "x",
    2: "y",
    3: "lower_y_domain",
    4: "lower_x_domain",
    5: "upper_y_domain",
    6: "upper_x_domain",
    7: "lower_z_domain",
    8: "upper_z_domain",
    10: "field_title",
    11: "domain_title",
    12: "x_lower_bound",
    13: "x_upper_bound",
    14: "y_lower_bound",
    15: "y_upper_bound",
}


#: Maps lbuser[0] to numpy data type. "default" will be interpreted if
#: no match is found, providing a warning in such a case.
LBUSER_DTYPE_LOOKUP = {
    1: np.dtype(">f4"),
    2: np.dtype(">i4"),
    3: np.dtype(">i4"),
    -1: np.dtype(">f4"),
    -2: np.dtype(">i4"),
    -3: np.dtype(">i4"),
    "default": np.dtype(">f4"),
}


class _WarnComboLoadingMask(
    iris.warnings.IrisLoadWarning,
    iris.warnings.IrisMaskValueMatchWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboLoadingDefaulting(
    iris.warnings.IrisDefaultingWarning,
    iris.warnings.IrisLoadWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class _WarnComboIgnoringLoad(
    iris.warnings.IrisIgnoringWarning,
    iris.warnings.IrisLoadWarning,
):
    """One-off combination of warning classes - enhances user filtering."""

    pass


class STASH(collections.namedtuple("STASH", "model section item")):
    """A class to hold a single STASH code.

    Create instances using:
        >>> model = 1
        >>> section = 2
        >>> item = 3
        >>> my_stash = iris.fileformats.pp.STASH(model, section, item)

    Access the sub-components via:
        >>> my_stash.model
        1
        >>> my_stash.section
        2
        >>> my_stash.item
        3

    String conversion results in the MSI format:
        >>> print(iris.fileformats.pp.STASH(1, 16, 203))
        m01s16i203

    A stash object can be compared directly
    to its string representation:
    >>> iris.fileformats.pp.STASH(1, 0, 4) == 'm01s0i004'
    True

    """

    __slots__ = ()

    def __new__(cls, model, section, item):
        """Create namedtuple STASH instance.

        Parameters
        ----------
        model :
            A positive integer less than 100, or None.
        section :
            A non-negative integer less than 100, or None.
        item :
            A positive integer less than 1000, or None.

        """
        model = cls._validate_member("model", model, 1, 99)
        section = cls._validate_member("section", section, 0, 99)
        item = cls._validate_member("item", item, 1, 999)
        return super().__new__(cls, model, section, item)

    @staticmethod
    def from_msi(msi):
        """Convert a STASH code MSI string to a STASH instance."""
        if not isinstance(msi, str):
            raise TypeError("Expected STASH code MSI string, got %r" % (msi,))

        msi_match = re.match(
            r"^\s*m(\d+|\?+)s(\d+|\?+)i(\d+|\?+)\s*$", msi, re.IGNORECASE
        )

        if msi_match is None:
            raise ValueError(
                'Expected STASH code MSI string "mXXsXXiXXX", ' "got %r" % (msi,)
            )

        return STASH(*msi_match.groups())

    @staticmethod
    def _validate_member(name, value, lower_limit, upper_limit):
        # Returns a valid integer or None.
        try:
            value = int(value)
            if not lower_limit <= value <= upper_limit:
                value = None
        except (TypeError, ValueError):
            value = None
        return value

    def __str__(self):
        model = self._format_member(self.model, 2)
        section = self._format_member(self.section, 2)
        item = self._format_member(self.item, 3)
        return "m{}s{}i{}".format(model, section, item)

    def _format_member(self, value, num_digits):
        if value is None:
            result = "?" * num_digits
        else:
            format_spec = "0" + str(num_digits)
            result = format(value, format_spec)
        return result

    def lbuser3(self):
        """Return the lbuser[3] value that this stash represents."""
        return (self.section or 0) * 1000 + (self.item or 0)

    def lbuser6(self):
        """Return the lbuser[6] value that this stash represents."""
        return self.model or 0

    @property
    def is_valid(self):
        return "?" not in str(self)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if isinstance(other, str):
            return super().__eq__(STASH.from_msi(other))
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class SplittableInt:
    """A class to hold integers which can easily get each decimal digit individually.

    >>> three_six_two = SplittableInt(362)
    >>> print(three_six_two)
    362
    >>> print(three_six_two[0])
    2
    >>> print(three_six_two[2])
    3

    Notes
    -----
    No support for negative numbers.

    """

    def __init__(self, value, name_mapping_dict=None):
        """Build a SplittableInt given the positive integer value provided.

        Parameters
        ----------
        name_mapping_dict : dict
            A special mapping to provide name based access to specific
            integer positions.

        Examples
        --------
            >>> a = SplittableInt(1234, {'hundreds': 2})
            >>> print(a.hundreds)
            2
            >>> a.hundreds = 9
            >>> print(a.hundreds)
            9
            >>> print(a)
            1934


        """
        if value < 0:
            raise ValueError(
                "Negative numbers not supported with splittable integers object"
            )

        # define the name lookup first (as this is the way __setattr__ is
        # plumbed)
        #: A dictionary mapping special attribute names on this object
        #: to the slices/indices required to access them.
        self._name_lookup = name_mapping_dict or {}
        self._value = value

        self._calculate_str_value_from_value()

    def __int__(self):
        return int(self._value)

    def _calculate_str_value_from_value(self):
        # Reverse the string to get the appropriate index when getting the
        # sliced value
        self._strvalue = [int(c) for c in str(self._value)[::-1]]

        # Associate the names in the lookup table to attributes
        for name, index in self._name_lookup.items():
            object.__setattr__(self, name, self[index])

    def _calculate_value_from_str_value(self):
        self._value = np.sum([10**i * val for i, val in enumerate(self._strvalue)])

    def __len__(self):
        return len(self._strvalue)

    def __getitem__(self, key):
        try:
            val = self._strvalue[key]
        except IndexError:
            val = 0

        # if the key returns a list of values, then combine them together
        # to an integer
        if isinstance(val, list):
            val = sum([10**i * val for i, val in enumerate(val)])

        return val

    def __setitem__(self, key, value):
        # The setitem method has been overridden so that assignment using
        # ``val[0] = 1`` style syntax updates
        # the entire object appropriately.

        if not isinstance(value, int) or value < 0:
            raise ValueError("Can only set %s as a positive integer value." % key)

        if isinstance(key, slice):
            if (
                (key.start is not None and key.start < 0)
                or (key.step is not None and key.step < 0)
                or (key.stop is not None and key.stop < 0)
            ):
                raise ValueError(
                    "Cannot assign a value with slice objects"
                    " containing negative indices."
                )

            # calculate the current length of the value of this string
            current_length = len(range(*key.indices(len(self))))

            # get indices for as many digits as have been requested. Putting
            # the upper limit on the number of digits at 100.
            indices = range(*key.indices(100))
            if len(indices) < len(str(value)):
                raise ValueError(
                    "Cannot put %s into %s as it has too many digits." % (value, key)
                )

            # Iterate over each of the indices in the slice,
            # zipping them together with the associated digit
            for index, digit in zip(indices, str(value).zfill(current_length)[::-1]):
                # assign each digit to the associated index
                self.__setitem__(index, int(digit))

        else:
            # If we are trying to set to an index which does not currently
            # exist in _strvalue then extend it to the
            # appropriate length
            if (key + 1) > len(self):
                new_str_value = [0] * (key + 1)
                new_str_value[: len(self)] = self._strvalue
                self._strvalue = new_str_value

            self._strvalue[key] = value

            for name, index in self._name_lookup.items():
                if index == key:
                    object.__setattr__(self, name, value)

            self._calculate_value_from_str_value()

    def __setattr__(self, name, value):
        # if the attribute is a special value, update the index value which
        # will in turn update the attribute value
        if name != "_name_lookup" and name in self._name_lookup:
            self[self._name_lookup[name]] = value
        else:
            object.__setattr__(self, name, value)

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return "SplittableInt(%r, name_mapping_dict=%r)" % (
            self._value,
            self._name_lookup,
        )

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, SplittableInt):
            result = self._value == other._value
        elif isinstance(other, int):
            result = self._value == other
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def _compare(self, other, op):
        result = NotImplemented
        if isinstance(other, SplittableInt):
            result = op(self._value, other._value)
        elif isinstance(other, int):
            result = op(self._value, other)
        return result

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.ge)


class PPDataProxy:
    """A reference to the data payload of a single PP field."""

    __slots__ = (
        "shape",
        "src_dtype",
        "path",
        "offset",
        "data_len",
        "_lbpack",
        "boundary_packing",
        "mdi",
    )

    def __init__(
        self,
        shape,
        src_dtype,
        path,
        offset,
        data_len,
        lbpack,
        boundary_packing,
        mdi,
    ):
        self.shape = shape
        self.src_dtype = src_dtype
        self.path = path
        self.offset = offset
        self.data_len = data_len
        self.lbpack = lbpack
        self.boundary_packing = boundary_packing
        self.mdi = mdi

    # lbpack
    @property
    def lbpack(self):
        value = self._lbpack
        if not isinstance(self._lbpack, SplittableInt):
            mapping = dict(n5=slice(4, None), n4=3, n3=2, n2=1, n1=0)
            value = SplittableInt(self._lbpack, mapping)
        return value

    @lbpack.setter
    def lbpack(self, value):
        self._lbpack = value

    @property
    def dtype(self):
        return self.src_dtype.newbyteorder("=")

    @property
    def fill_value(self):
        return self.mdi

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dask_meta(self):
        return np.empty((0,) * self.ndim, dtype=self.dtype)

    def __getitem__(self, keys):
        with open(self.path, "rb") as pp_file:
            pp_file.seek(self.offset, os.SEEK_SET)
            data_bytes = pp_file.read(self.data_len)
            data = _data_bytes_to_shaped_array(
                data_bytes,
                self.lbpack,
                self.boundary_packing,
                self.shape,
                self.src_dtype,
                self.mdi,
            )
        result = data.__getitem__(keys)

        return np.asanyarray(result, dtype=self.dtype)

    def __repr__(self):
        fmt = (
            "<{self.__class__.__name__} shape={self.shape}"
            " src_dtype={self.dtype!r} path={self.path!r}"
            " offset={self.offset}>"
        )
        return fmt.format(self=self)

    def __getstate__(self):
        # Because we have __slots__, this is needed to support Pickle.dump()
        return [(name, getattr(self, name)) for name in self.__slots__]

    def __setstate__(self, state):
        # Because we have __slots__, this is needed to support Pickle.load()
        # (Use setattr, as there is no object dictionary.)
        for key, value in state:
            setattr(self, key, value)

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, PPDataProxy):
            result = True
            for attr in self.__slots__:
                if getattr(self, attr) != getattr(other, attr):
                    result = False
                    break
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result


def _data_bytes_to_shaped_array(
    data_bytes, lbpack, boundary_packing, data_shape, data_type, mdi, mask=None
):
    """Convert binary payload into a numpy array.

    Convert the already read binary data payload into a numpy array, unpacking
    and decompressing as per the F3 specification.
    """
    if lbpack.n1 in (0, 2):
        data = np.frombuffer(data_bytes, dtype=data_type)
    elif lbpack.n1 == 1:
        if mo_pack is not None:
            try:
                decompress_wgdos = mo_pack.decompress_wgdos
            except AttributeError:
                decompress_wgdos = mo_pack.unpack_wgdos
        else:
            msg = (
                "Unpacking PP fields with LBPACK of {} "
                "requires mo_pack to be installed".format(lbpack.n1)
            )
            raise ValueError(msg)
        data = decompress_wgdos(data_bytes, data_shape[0], data_shape[1], mdi)
    elif lbpack.n1 == 4:
        if mo_pack is not None and hasattr(mo_pack, "decompress_rle"):
            decompress_rle = mo_pack.decompress_rle
        else:
            msg = (
                "Unpacking PP fields with LBPACK of {} "
                "requires mo_pack to be installed".format(lbpack.n1)
            )
            raise ValueError(msg)
        data = decompress_rle(data_bytes, data_shape[0], data_shape[1], mdi)
    else:
        raise iris.exceptions.NotYetImplementedError(
            "PP fields with LBPACK of %s are not yet supported." % lbpack
        )

    # Ensure we have a writeable data buffer.
    # NOTE: "data.setflags(write=True)" is not valid for numpy >= 1.16.0.
    if not data.flags["WRITEABLE"]:
        data = data.copy()

    # Ensure the data is in the native byte order
    if not data.dtype.isnative:
        data.byteswap(True)
        data.dtype = data.dtype.newbyteorder("=")

    if boundary_packing is not None:
        # Convert a long string of numbers into a "lateral boundary
        # condition" array, which is split into 4 quartiles, North
        # East, South, West and where North and South contain the corners.
        compressed_data = data
        data = np.ma.masked_all(data_shape)
        data.fill_value = mdi

        boundary_height = boundary_packing.y_halo + boundary_packing.rim_width
        boundary_width = boundary_packing.x_halo + boundary_packing.rim_width
        y_height, x_width = data_shape
        # The height of the east and west components.
        mid_height = y_height - 2 * boundary_height

        n_s_shape = boundary_height, x_width
        e_w_shape = mid_height, boundary_width

        # Keep track of our current position in the array.
        current_posn = 0

        north = compressed_data[: boundary_height * x_width]
        current_posn += len(north)
        data[-boundary_height:, :] = north.reshape(*n_s_shape)

        east = compressed_data[
            current_posn : current_posn + boundary_width * mid_height
        ]
        current_posn += len(east)
        data[boundary_height:-boundary_height, -boundary_width:] = east.reshape(
            *e_w_shape
        )

        south = compressed_data[current_posn : current_posn + boundary_height * x_width]
        current_posn += len(south)
        data[:boundary_height, :] = south.reshape(*n_s_shape)

        west = compressed_data[
            current_posn : current_posn + boundary_width * mid_height
        ]
        current_posn += len(west)
        data[boundary_height:-boundary_height, :boundary_width] = west.reshape(
            *e_w_shape
        )

    elif lbpack.n2 == 2:
        if mask is None:
            # If we are given no mask to apply, then just return raw data, even
            # though it does not have the correct shape.
            # For dask-delayed loading, this means that mask, data and the
            # combination can be properly handled within a dask graph.
            # However, we still mask any MDI values in the array (below).
            pass
        else:
            land_mask = mask.data.astype(np.bool_)
            sea_mask = ~land_mask
            new_data = np.ma.masked_all(land_mask.shape)
            new_data.fill_value = mdi
            if lbpack.n3 == 1:
                # Land mask packed data.
                # Sometimes the data comes in longer than it should be (i.e. it
                # looks like the compressed data is compressed, but the
                # trailing data hasn't been clipped off!).
                new_data[land_mask] = data[: land_mask.sum()]
            elif lbpack.n3 == 2:
                # Sea mask packed data.
                new_data[sea_mask] = data[: sea_mask.sum()]
            else:
                raise ValueError("Unsupported mask compression.")
            data = new_data

    else:
        # Reform in row-column order
        actual_length = np.prod(data.shape)
        if (expected_length := np.prod(data_shape)) != actual_length:
            if (expected_length < actual_length) and (data.ndim == 1):
                # known use case where mule adds padding to data payload
                # for a collapsed field.
                data = data[:expected_length]
            else:
                emsg = (
                    f"PP field data containing {actual_length} words does not "
                    f"match expected length of {expected_length} words."
                )
                raise ValueError(emsg)
        data.shape = data_shape

    # Mask the array
    if mdi in data:
        data = ma.masked_values(data, mdi, copy=False)

    return data


# The special headers of the PPField classes which get some improved
# functionality
_SPECIAL_HEADERS = (
    "lbtim",
    "lbcode",
    "lbpack",
    "lbproc",
    "data",
    "stash",
    "t1",
    "t2",
)


def _header_defn(release_number):
    """Return zero-indexed header definition for a particular release of a PPField."""
    um_header = UM_HEADERS[release_number]
    offset = UM_TO_PP_HEADER_OFFSET
    return [
        (name, tuple(position - offset for position in positions))
        for name, positions in um_header
    ]


def _pp_attribute_names(header_defn):
    """Return the allowed attributes of a PPField.

    Return the allowed attributes of a PPField:
    all of the normal headers (i.e. not the _SPECIAL_HEADERS),
    the _SPECIAL_HEADERS with '_' prefixed,
    the possible extra data headers.

    """
    normal_headers = list(
        name for name, positions in header_defn if name not in _SPECIAL_HEADERS
    )
    special_headers = list("_" + name for name in _SPECIAL_HEADERS)
    extra_data = list(EXTRA_DATA.values())
    special_attributes = [
        "_raw_header",
        "raw_lbtim",
        "raw_lbpack",
        "boundary_packing",
        "_index_in_structured_load_file",
    ]
    return normal_headers + special_headers + extra_data + special_attributes


class PPField(metaclass=ABCMeta):
    """Base class for PP fields.

    A generic class for PP fields - not specific to a particular
    header release number.

    A PPField instance can easily access the PP header "words" as attributes
    with some added useful capabilities::

        for field in iris.fileformats.pp.load(filename):
            print(field.lbyr)
            print(field.lbuser)
            print(field.lbuser[0])
            print(field.lbtim)
            print(field.lbtim.ia)
            print(field.t1)

    """

    # NB. Subclasses must define the attribute HEADER_DEFN to be their
    # zero-based header definition. See PPField2 and PPField3 for examples.

    __slots__ = ()

    def __init__(self, header=None):
        # Combined header longs and floats data cache.
        self._raw_header = header
        self.raw_lbtim = None
        self.raw_lbpack = None
        self.boundary_packing = None
        self._index_in_structured_load_file = None
        if header is not None:
            self.raw_lbtim = header[self.HEADER_DICT["lbtim"][0]]
            self.raw_lbpack = header[self.HEADER_DICT["lbpack"][0]]

    def __getattr__(self, key):
        """Return the value of the key.

        Method supports deferred attribute creation, which offers a
        significant loading optimisation, particularly when not all attributes
        are referenced and therefore created on the instance.

        When an 'ordinary' HEADER_DICT attribute is required, its associated
        header offset is used to lookup the data value/s from the combined
        header longs and floats data cache. The attribute is then set with this
        value/s on the instance. Thus future lookups for this attribute will be
        optimised, avoiding the __getattr__ lookup mechanism again.

        When a 'special' HEADER_DICT attribute (leading underscore) is
        required, its associated 'ordinary' (no leading underscore) header
        offset is used to lookup the data value/s from the combined header
        longs and floats data cache. The 'ordinary' attribute is then set
        with this value/s on the instance. This is required as 'special'
        attributes have supporting property convenience functionality base on
        the attribute value e.g. see 'lbpack' and 'lbtim'. Note that, for
        'special' attributes the interface is via the 'ordinary' attribute but
        the underlying attribute value is stored within the 'special'
        attribute.

        """
        try:
            loc = self.HEADER_DICT[key]
        except KeyError:
            if key[0] == "_" and key[1:] in self.HEADER_DICT:
                # Must be a special attribute.
                loc = self.HEADER_DICT[key[1:]]
            else:
                cls = self.__class__.__name__
                msg = "{!r} object has no attribute {!r}".format(cls, key)
                raise AttributeError(msg)

        if len(loc) == 1:
            value = self._raw_header[loc[0]]
        else:
            start = loc[0]
            stop = loc[-1] + 1
            value = tuple(self._raw_header[start:stop])

        # Now cache the attribute value on the instance.
        if key[0] == "_":
            # First we need to assign to the attribute so that the
            # special attribute is calculated, then we retrieve it.
            setattr(self, key[1:], value)
            value = getattr(self, key)
        else:
            setattr(self, key, value)
        return value

    @property
    @abstractmethod
    def t1(self):
        pass

    @property
    @abstractmethod
    def t2(self):
        pass

    def __repr__(self):
        """Return a string representation of the PP field."""
        # Define an ordering on the basic header names
        attribute_priority_lookup = {name: loc[0] for name, loc in self.HEADER_DEFN}

        # With the attributes sorted the order will remain stable if extra
        # attributes are added.
        public_attribute_names = list(attribute_priority_lookup.keys()) + list(
            EXTRA_DATA.values()
        )
        self_attrs = [
            (name, getattr(self, name, None)) for name in public_attribute_names
        ]
        self_attrs = [pair for pair in self_attrs if pair[1] is not None]

        self_attrs.append(("data", self.data))

        # sort the attributes by position in the pp header followed,
        # then by alphabetical order.
        attributes = sorted(
            self_attrs,
            key=lambda pair: (
                attribute_priority_lookup.get(pair[0], 999),
                pair[0],
            ),
        )

        return (
            "PP Field" + "".join(["\n   %s: %s" % (k, v) for k, v in attributes]) + "\n"
        )

    @property
    def stash(self):
        """Stash property giving access to the associated STASH object.

        Now supporting __eq__.

        """
        if (
            not hasattr(self, "_stash")
            or self.lbuser[6] != self._stash.lbuser6()
            or self.lbuser[3] != self._stash.lbuser3()
        ):
            self._stash = STASH(
                self.lbuser[6], self.lbuser[3] // 1000, self.lbuser[3] % 1000
            )
        return self._stash

    @stash.setter
    def stash(self, stash):
        if isinstance(stash, str):
            self._stash = STASH.from_msi(stash)
        elif isinstance(stash, STASH):
            self._stash = stash
        else:
            raise ValueError("Cannot set stash to {!r}".format(stash))

        # Keep the lbuser up to date.
        self.lbuser = list(self.lbuser)
        self.lbuser[6] = self._stash.lbuser6()
        self.lbuser[3] = self._stash.lbuser3()

    @property
    def lbtim(self):
        return self._lbtim

    @lbtim.setter
    def lbtim(self, value):
        value = int(value)
        self.raw_lbtim = value
        self._lbtim = SplittableInt(value, {"ia": slice(2, None), "ib": 1, "ic": 0})

    # lbcode
    @property
    def lbcode(self):
        return self._lbcode

    @lbcode.setter
    def lbcode(self, new_value):
        if not isinstance(new_value, SplittableInt):
            # add the ix/iy values for lbcode
            new_value = SplittableInt(new_value, {"iy": slice(0, 2), "ix": slice(2, 4)})
        self._lbcode = new_value

    # lbpack
    @property
    def lbpack(self):
        return self._lbpack

    @lbpack.setter
    def lbpack(self, new_value):
        if not isinstance(new_value, SplittableInt):
            self.raw_lbpack = new_value
            # add the n1/n2/n3/n4/n5 values for lbpack
            name_mapping = dict(n5=slice(4, None), n4=3, n3=2, n2=1, n1=0)
            new_value = SplittableInt(new_value, name_mapping)
        else:
            self.raw_lbpack = new_value._value
        self._lbpack = new_value

    @property
    def lbproc(self):
        return self._lbproc

    @lbproc.setter
    def lbproc(self, value):
        value = int(value)
        if value < 0:
            raise ValueError("PPField.lbproc cannot be a negative number.")
        self._lbproc = value

    @property
    def data(self):
        """:class:`numpy.ndarray` representing multidimensional data of the pp file."""
        if is_lazy_data(self._data):
            # Replace with real data on the first access.
            self._data = as_concrete_data(self._data)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def core_data(self):
        return self._data

    @property
    def calendar(self):
        """Return the calendar of the field."""
        # TODO #577 What calendar to return when ibtim.ic in [0, 3]
        calendar = cf_units.CALENDAR_STANDARD
        if self.lbtim.ic == 2:
            calendar = cf_units.CALENDAR_360_DAY
        elif self.lbtim.ic == 4:
            calendar = cf_units.CALENDAR_365_DAY
        return calendar

    def _read_extra_data(self, pp_file, file_reader, extra_len, little_ended=False):
        """Read the extra data section and update the self appropriately."""
        dtype_endian_char = "<" if little_ended else ">"
        # While there is still extra data to decode run this loop
        while extra_len > 0:
            dtype = "%cL" % dtype_endian_char
            extra_int_code = struct.unpack_from(dtype, file_reader(PP_WORD_DEPTH))[0]
            extra_len -= PP_WORD_DEPTH

            ib = extra_int_code % 1000
            ia = extra_int_code // 1000

            data_len = ia * PP_WORD_DEPTH

            if ib == 10:
                dtype = "%c%ds" % (dtype_endian_char, data_len)
                field_title = struct.unpack_from(dtype, file_reader(data_len))
                self.field_title = field_title[0].rstrip(b"\00").decode()
            elif ib == 11:
                dtype = "%c%ds" % (dtype_endian_char, data_len)
                domain_title = struct.unpack_from(dtype, file_reader(data_len))
                self.domain_title = domain_title[0].rstrip(b"\00").decode()
            elif ib in EXTRA_DATA:
                attr_name = EXTRA_DATA[ib]
                dtype = np.dtype("%cf%d" % (dtype_endian_char, PP_WORD_DEPTH))
                values = np.fromfile(pp_file, dtype=dtype, count=ia)
                # Ensure the values are in the native byte order
                if not values.dtype.isnative:
                    values.byteswap(True)
                    values.dtype = values.dtype.newbyteorder("=")
                setattr(self, attr_name, values)
            else:
                raise ValueError("Unknown IB value for extra data: %s" % ib)

            extra_len -= data_len

    @property
    def x_bounds(self):
        if hasattr(self, "x_lower_bound") and hasattr(self, "x_upper_bound"):
            return np.column_stack((self.x_lower_bound, self.x_upper_bound))

    @property
    def y_bounds(self):
        if hasattr(self, "y_lower_bound") and hasattr(self, "y_upper_bound"):
            return np.column_stack((self.y_lower_bound, self.y_upper_bound))

    def save(self, file_handle):
        """Save the PPField to the given file object.

        Typically created with :func:`open`::

            # to append the field to a file
            with open(filename, 'ab') as fh:
                a_pp_field.save(fh)

            # to overwrite/create a file
            with open(filename, 'wb') as fh:
                a_pp_field.save(fh)

        Notes
        -----
        The fields which are automatically calculated are: 'lbext',
        'lblrec' and 'lbuser[0]'. Some fields are not currently
        populated, these are: 'lbegin', 'lbnrec', 'lbuser[1]'.

        """
        # Get the actual data content.
        data = self.data
        mdi = self.bmdi
        if np.any(data == mdi):
            msg = (
                "PPField data contains unmasked points equal to the fill "
                "value, {}. As saved, these points will read back as "
                "missing data. To save these as normal values, please "
                "set the field BMDI not equal to any valid data points."
            )
            warnings.warn(
                msg.format(mdi),
                category=_WarnComboLoadingMask,
            )
        if isinstance(data, ma.MaskedArray):
            if ma.is_masked(data):
                data = data.filled(fill_value=mdi)
            else:
                data = data.data

        # Make sure the data is big-endian
        if data.dtype.newbyteorder(">") != data.dtype:
            # take a copy of the data when byteswapping
            data = data.byteswap(False)
            data.dtype = data.dtype.newbyteorder(">")

        # Create the arrays which will hold the header information
        lb = np.empty(shape=NUM_LONG_HEADERS, dtype=np.dtype(">u%d" % PP_WORD_DEPTH))
        b = np.empty(shape=NUM_FLOAT_HEADERS, dtype=np.dtype(">f%d" % PP_WORD_DEPTH))

        # Fill in the header elements from the PPField
        for name, pos in self.HEADER_DEFN:
            try:
                header_elem = getattr(self, name)
            except AttributeError:
                raise AttributeError("PPField.save() could not find %s" % name)
            if pos[0] <= NUM_LONG_HEADERS - UM_TO_PP_HEADER_OFFSET:
                index = slice(pos[0], pos[-1] + 1)
                if isinstance(header_elem, SplittableInt):
                    header_elem = int(header_elem)
                lb[index] = header_elem
            else:
                index = slice(pos[0] - NUM_LONG_HEADERS, pos[-1] - NUM_LONG_HEADERS + 1)
                b[index] = header_elem

        # Although all of the elements are now populated, we still need to
        # update some of the elements in case
        # things have changed (for example, the data length etc.)

        # Set up a variable to represent the datalength of this PPField in
        # WORDS.
        len_of_data_payload = 0

        # set up a list to hold the extra data which will need to be encoded
        # at the end of the data
        extra_items = []
        # iterate through all of the possible extra data fields
        for ib, extra_data_attr_name in EXTRA_DATA.items():
            # try to get the extra data field, returning None if it doesn't
            # exist
            extra_elem = getattr(self, extra_data_attr_name, None)
            if extra_elem is not None:
                # The special case of character extra data must be caught
                if isinstance(extra_elem, str):
                    ia = len(extra_elem)
                    # pad any strings up to a multiple of PP_WORD_DEPTH
                    # (this length is # of bytes)
                    ia = (PP_WORD_DEPTH - (ia - 1) % PP_WORD_DEPTH) + (ia - 1)
                    extra_elem = extra_elem.ljust(ia, "\00")

                    # ia is now the datalength in WORDS of the string
                    ia //= PP_WORD_DEPTH
                else:
                    # ia is the datalength in WORDS
                    ia = np.prod(extra_elem.shape)
                    # flip the byteorder if the data is not big-endian
                    if extra_elem.dtype.newbyteorder(">") != extra_elem.dtype:
                        # take a copy of the extra data when byte swapping
                        extra_elem = extra_elem.byteswap(False)
                        extra_elem.dtype = extra_elem.dtype.newbyteorder(">")

                # add the number of bytes to the len_of_data_payload variable
                # + the extra integer which will encode ia/ib
                len_of_data_payload += PP_WORD_DEPTH * ia + PP_WORD_DEPTH
                integer_code = 1000 * ia + ib
                extra_items.append([integer_code, extra_elem])

                if ia >= 1000:
                    raise IOError(
                        "PP files cannot write extra data with more"
                        ' than 1000 elements. Tried to write "%s"'
                        " which has %s elements." % (extra_data_attr_name, ib)
                    )

        # populate lbext in WORDS
        lb[self.HEADER_DICT["lbext"][0]] = len_of_data_payload // PP_WORD_DEPTH

        # Put the data length of pp.data into len_of_data_payload (in BYTES)
        lbpack = lb[self.HEADER_DICT["lbpack"][0]]
        if lbpack == 0:
            len_of_data_payload += data.size * PP_WORD_DEPTH
        elif lbpack == 1:
            if mo_pack is not None:
                try:
                    compress_wgdos = mo_pack.compress_wgdos
                except AttributeError:
                    compress_wgdos = mo_pack.pack_wgdos
                packed_data = compress_wgdos(
                    data.astype(np.float32),
                    b[self.HEADER_DICT["bacc"][0] - 45],
                    b[self.HEADER_DICT["bmdi"][0] - 45],
                )
                len_of_data_payload += len(packed_data)
            else:
                msg = (
                    "Writing packed pp data with lbpack of {} "
                    "requires mo_pack to be installed.".format(lbpack)
                )
                raise NotImplementedError(msg)

        # populate lbrec in WORDS
        lb[self.HEADER_DICT["lblrec"][0]] = len_of_data_payload // PP_WORD_DEPTH

        # populate lbuser[0] to have the data's datatype
        if data.dtype == np.dtype(">f4"):
            lb[self.HEADER_DICT["lbuser"][0]] = 1
        elif data.dtype == np.dtype(">f8"):
            warnings.warn(
                "Downcasting array precision from float64 to float32"
                " for save.If float64 precision is required then"
                " please save in a different format",
                category=_WarnComboLoadingDefaulting,
            )
            data = data.astype(">f4")
            lb[self.HEADER_DICT["lbuser"][0]] = 1
        elif data.dtype == np.dtype(">i4"):
            # NB: there is no physical difference between lbuser[0] of 2 or 3
            # so we encode just 2
            lb[self.HEADER_DICT["lbuser"][0]] = 2
        else:
            raise IOError(
                "Unable to write data array to a PP file. "
                "The datatype was %s." % data.dtype
            )

        # NB: lbegin, lbnrec, lbuser[1] not set up

        # Now that we have done the manouvering required, write to the file...
        if not hasattr(file_handle, "write"):
            raise TypeError(
                "The file_handle argument must be an instance of a"
                " Python file object, but got %r. \n e.g. "
                'open(filename, "wb") to open a binary file with'
                " write permission." % type(file_handle)
            )

        pp_file = file_handle

        # header length
        pp_file.write(struct.pack(">L", PP_HEADER_DEPTH))

        # 45 integers
        lb.tofile(pp_file)
        # 19 floats
        b.tofile(pp_file)

        # Header length (again)
        pp_file.write(struct.pack(">L", PP_HEADER_DEPTH))

        # Data length (including extra data length)
        pp_file.write(struct.pack(">L", int(len_of_data_payload)))

        # the data itself
        if lbpack == 0:
            data.tofile(pp_file)
        elif lbpack == 1:
            pp_file.write(packed_data)
        else:
            msg = "Writing packed pp data with lbpack of {} is not supported.".format(
                lbpack
            )
            raise NotImplementedError(msg)

        # extra data elements
        for int_code, extra_data in extra_items:
            pp_file.write(struct.pack(">L", int(int_code)))
            if isinstance(extra_data, str):
                pp_file.write(
                    struct.pack(">%ss" % len(extra_data), extra_data.encode())
                )
            else:
                extra_data = extra_data.astype(np.dtype(">f4"))
                extra_data.tofile(pp_file)

        # Data length (again)
        pp_file.write(struct.pack(">L", int(len_of_data_payload)))

    ##############################################################
    #
    # From here on define helper methods for PP -> Cube conversion.
    #

    def time_unit(self, time_unit, epoch="epoch"):
        return cf_units.Unit("%s since %s" % (time_unit, epoch), calendar=self.calendar)

    def coord_system(self):
        """Return a CoordSystem for this PPField.

        Returns
        -------
        :class:`~iris.coord_systems.GeogCS` or :class:`~iris.coord_systems.RotatedGeogCS`.

        """
        geog_cs = iris.coord_systems.GeogCS(EARTH_RADIUS)

        def degrees_ne(angle, ref_angle):
            """Return whether an angle differs significantly from a set value.

            The inputs are in degrees.
            The difference is judged significant if more than 0.0001 degrees.

            """
            return abs(angle - ref_angle) > 0.0001

        if degrees_ne(self.bplat, 90.0) or (
            degrees_ne(self.bplon, 0.0) and degrees_ne(self.bplon, 180.0)
        ):
            # NOTE: when bplat,bplon=90,0 this encodes an unrotated system.
            # However, the rotated system which is *equivalent* to an unrotated
            # one actually has blat,bplon=90,180, due to a quirk in the
            # definition equations.
            # So we accept BPLON of 0 *or* 180 to mean 'unrotated'.
            geog_cs = iris.coord_systems.RotatedGeogCS(
                self.bplat, self.bplon, ellipsoid=geog_cs
            )

        return geog_cs

    def _x_coord_name(self):
        # TODO: Remove once we have the ability to derive this in the rules.
        x_name = "longitude"
        if isinstance(self.coord_system(), iris.coord_systems.RotatedGeogCS):
            x_name = "grid_longitude"
        return x_name

    def _y_coord_name(self):
        # TODO: Remove once we have the ability to derive this in the rules.
        y_name = "latitude"
        if isinstance(self.coord_system(), iris.coord_systems.RotatedGeogCS):
            y_name = "grid_latitude"
        return y_name

    def copy(self):
        """Return a deep copy of this PPField.

        Returns
        -------
        :class:`PPField`:
            A copy instance of the :class:`PPField`.

        """
        return self._deepcopy({})

    def __deepcopy__(self, memo):
        return self._deepcopy(memo)

    def _deepcopy(self, memo):
        field = self.__class__()
        for attr in self.__slots__:
            if hasattr(self, attr):
                value = getattr(self, attr)
                # Cope with inability to deepcopy a 0-d NumPy array.
                if attr == "_data" and value is not None and value.ndim == 0:
                    setattr(field, attr, np.array(deepcopy(value[()], memo)))
                else:
                    setattr(field, attr, deepcopy(value, memo))
        return field

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, PPField):
            result = True
            for attr in self.__slots__:
                attrs = [hasattr(self, attr), hasattr(other, attr)]
                if all(attrs):
                    self_attr = getattr(self, attr)
                    other_attr = getattr(other, attr)
                    if not np.all(self_attr == other_attr):
                        result = False
                        break
                elif any(attrs):
                    result = False
                    break
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result


class PPField2(PPField):
    """Hold a single field from a PP file, with a header release number of 2."""

    HEADER_DEFN = _header_defn(2)
    HEADER_DICT = dict(HEADER_DEFN)

    __slots__ = _pp_attribute_names(HEADER_DEFN)

    @property
    def t1(self):
        """A cftime.datetime object.

        cftime.datetime object consisting of the lbyr, lbmon, lbdat, lbhr,
        and lbmin attributes.

        """
        if not hasattr(self, "_t1"):
            has_year_zero = self.lbyr == 0 or None
            calendar = None if self.lbmon == 0 or self.lbdat == 0 else self.calendar
            self._t1 = cftime.datetime(
                self.lbyr,
                self.lbmon,
                self.lbdat,
                self.lbhr,
                self.lbmin,
                calendar=calendar,
                has_year_zero=has_year_zero,
            )
        return self._t1

    @t1.setter
    def t1(self, dt):
        self.lbyr = dt.year
        self.lbmon = dt.month
        self.lbdat = dt.day
        self.lbhr = dt.hour
        self.lbmin = dt.minute
        self.lbday = int(dt.strftime("%j"))
        if hasattr(self, "_t1"):
            delattr(self, "_t1")

    @property
    def t2(self):
        """A cftime.datetime object.

        cftime.datetime object consisting of the lbyrd, lbmond, lbdatd,
        lbhrd, and lbmind attributes.

        """
        if not hasattr(self, "_t2"):
            has_year_zero = self.lbyrd == 0 or None
            calendar = None if self.lbmond == 0 or self.lbdatd == 0 else self.calendar
            self._t2 = cftime.datetime(
                self.lbyrd,
                self.lbmond,
                self.lbdatd,
                self.lbhrd,
                self.lbmind,
                calendar=calendar,
                has_year_zero=has_year_zero,
            )
        return self._t2

    @t2.setter
    def t2(self, dt):
        self.lbyrd = dt.year
        self.lbmond = dt.month
        self.lbdatd = dt.day
        self.lbhrd = dt.hour
        self.lbmind = dt.minute
        self.lbdayd = int(dt.strftime("%j"))
        if hasattr(self, "_t2"):
            delattr(self, "_t2")


class PPField3(PPField):
    """Hold a single field from a PP file, with a header release number of 3."""

    HEADER_DEFN = _header_defn(3)
    HEADER_DICT = dict(HEADER_DEFN)

    __slots__ = _pp_attribute_names(HEADER_DEFN)

    @property
    def t1(self):
        """A cftime.datetime object.

        cftime.datetime object consisting of the lbyr, lbmon, lbdat, lbhr,
        lbmin, and lbsec attributes.

        """
        if not hasattr(self, "_t1"):
            has_year_zero = self.lbyr == 0 or None
            calendar = None if self.lbmon == 0 or self.lbdat == 0 else self.calendar
            self._t1 = cftime.datetime(
                self.lbyr,
                self.lbmon,
                self.lbdat,
                self.lbhr,
                self.lbmin,
                self.lbsec,
                calendar=calendar,
                has_year_zero=has_year_zero,
            )
        return self._t1

    @t1.setter
    def t1(self, dt):
        self.lbyr = dt.year
        self.lbmon = dt.month
        self.lbdat = dt.day
        self.lbhr = dt.hour
        self.lbmin = dt.minute
        self.lbsec = dt.second
        if hasattr(self, "_t1"):
            delattr(self, "_t1")

    @property
    def t2(self):
        """A cftime.datetime object.

        cftime.datetime object consisting of the lbyrd, lbmond, lbdatd,
        lbhrd, lbmind, and lbsecd attributes.

        """
        if not hasattr(self, "_t2"):
            has_year_zero = self.lbyrd == 0 or None
            calendar = None if self.lbmond == 0 or self.lbdatd == 0 else self.calendar
            self._t2 = cftime.datetime(
                self.lbyrd,
                self.lbmond,
                self.lbdatd,
                self.lbhrd,
                self.lbmind,
                self.lbsecd,
                calendar=calendar,
                has_year_zero=has_year_zero,
            )
        return self._t2

    @t2.setter
    def t2(self, dt):
        self.lbyrd = dt.year
        self.lbmond = dt.month
        self.lbdatd = dt.day
        self.lbhrd = dt.hour
        self.lbmind = dt.minute
        self.lbsecd = dt.second
        if hasattr(self, "_t2"):
            delattr(self, "_t2")


PP_CLASSES = {2: PPField2, 3: PPField3}


def make_pp_field(header):
    # Choose a PP field class from the value of LBREL
    lbrel = header[21]
    if lbrel not in PP_CLASSES:
        raise ValueError("Unsupported header release number: {}".format(lbrel))
    pp_field = PP_CLASSES[lbrel](header)
    return pp_field


LoadedArrayBytes = collections.namedtuple("LoadedArrayBytes", "bytes, dtype")


def load(filename, read_data=False, little_ended=False):
    """Return an iterator of PPFields given a filename.

    Parameters
    ----------
    filename : str
        String of the filename to load.
    read_data : bool, default=False
        Flag whether or not the data should be read, if False an empty
        data manager will be provided which can subsequently load the data
        on demand. Default False.
    little_ended : bool, default=False
        If True, file contains all little-ended words (header and data).

    Notes
    -----
    To iterate through all of the fields in a pp file::

        for field in iris.fileformats.pp.load(filename):
            print(field)

    """
    return _interpret_fields(
        _field_gen(filename, read_data_bytes=read_data, little_ended=little_ended)
    )


def _interpret_fields(fields):
    """Turn the fields read with load and FF2PP._extract_field into usable fields.

    Turn the fields read with load and FF2PP._extract_field into usable
    fields. One of the primary purposes of this function is to either convert
    "deferred bytes" into "deferred arrays" or "loaded bytes" into actual
    numpy arrays (via the _create_field_data) function.

    """
    land_mask_field = None
    landmask_compressed_fields = []
    for field in fields:
        # Store the first reference to a land mask, and use this as the
        # definitive mask for future fields in this generator.
        if (
            land_mask_field is None
            and field.lbuser[6] == 1
            and (field.lbuser[3] // 1000) == 0
            and (field.lbuser[3] % 1000) == 30
        ):
            land_mask_field = field

        # Handle land compressed data payloads, when lbpack.n2 is 2.
        if (field.raw_lbpack // 10 % 10) == 2:
            # Field uses land-mask packing, so needs a related land-mask field.
            if land_mask_field is None:
                landmask_compressed_fields.append(field)
                # Land-masked fields have their size+shape defined by the
                # reference landmask field, so we can't yield them if they
                # are encountered *before* the landmask.
                # In that case, defer them, and process them all afterwards at
                # the end of the file.
                continue

            # Land-mask compressed fields don't have an lbrow and lbnpt.
            field.lbrow, field.lbnpt = (
                land_mask_field.lbrow,
                land_mask_field.lbnpt,
            )
            _create_field_data(
                field,
                (field.lbrow, field.lbnpt),
                land_mask_field=land_mask_field,
            )
        else:
            # Field does not use land-mask packing.
            _create_field_data(field, (field.lbrow, field.lbnpt))

        yield field

    # At file end, return any land-masked fields that were deferred because
    # they were encountered before the landmask reference field.
    if landmask_compressed_fields:
        if land_mask_field is None:
            warnings.warn(
                "Landmask compressed fields existed without a "
                "landmask to decompress with. The data will have "
                "a shape of (0, 0) and will not read.",
                category=iris.warnings.IrisLoadWarning,
            )
            mask_shape = (0, 0)
        else:
            mask_shape = (land_mask_field.lbrow, land_mask_field.lbnpt)

        for field in landmask_compressed_fields:
            field.lbrow, field.lbnpt = mask_shape
            _create_field_data(field, mask_shape, land_mask_field=land_mask_field)
            yield field


def _create_field_data(field, data_shape, land_mask_field=None):
    """Modify a field's ``_data`` attribute.

    Modify a field's ``_data`` attribute either by:

    * converting a 'deferred array bytes' tuple into a lazy array,
    * converting LoadedArrayBytes into an actual numpy array.

    If 'land_mask_field' is passed (not None), then it contains the associated
    landmask, which is also a field :  Its data array is used as a template for
    `field`'s data, determining its size, shape and the locations of all the
    valid (non-missing) datapoints.

    """
    if isinstance(field.core_data(), LoadedArrayBytes):
        loaded_bytes = field.core_data()
        field.data = _data_bytes_to_shaped_array(
            loaded_bytes.bytes,
            field.lbpack,
            field.boundary_packing,
            data_shape,
            loaded_bytes.dtype,
            field.bmdi,
            land_mask_field,
        )
    else:
        # Wrap the reference to the data payload within a data proxy
        # in order to support deferred data loading.
        fname, position, n_bytes, dtype = field.core_data()
        proxy = PPDataProxy(
            data_shape,
            dtype,
            fname,
            position,
            n_bytes,
            field.raw_lbpack,
            field.boundary_packing,
            field.bmdi,
        )
        block_shape = data_shape if 0 not in data_shape else (1, 1)
        if land_mask_field is None:
            # For a "normal" (non-landsea-masked) field, the proxy can be
            # wrapped directly as a deferred array.
            field.data = as_lazy_data(proxy, meta=proxy.dask_meta, chunks=block_shape)
        else:
            # This is a landsea-masked field, and its data must be handled in
            # a different way :  Because data shape/size is not known in
            # advance, the data+mask calculation can't be represented
            # as a dask-array operation.  Instead, we make that calculation
            # 'delayed', and then use 'from_delayed' to make the result back
            # into a dask array -- because the final result shape *is* known.
            @dask.delayed
            def fetch_valid_values_array():
                # Return the data values array (shape+size unknown).
                return proxy[:]

            delayed_valid_values = fetch_valid_values_array()

            # Get the mask data-array from the landsea-mask field.
            # This is *either* a lazy or a real array, we don't actually care.
            # If this is a deferred dependency, the delayed calc can see that.
            mask_field_array = land_mask_field.core_data()

            # Check whether this field uses a land or a sea mask.
            if field.lbpack.n3 not in (1, 2):
                raise ValueError(
                    "Unsupported mask compression : lbpack.n3 = {}.".format(
                        field.lbpack.n3
                    )
                )
            if field.lbpack.n3 == 2:
                # Sea-mask packing : points are inverse of the land-mask.
                mask_field_array = ~mask_field_array

            # Define the mask+data calculation as a deferred operation.
            # NOTE: duplicates the operation in _data_bytes_to_shaped_array.
            @dask.delayed
            def calc_array(mask, values):
                # Note: "mask" is True at *valid* points, not missing ones.
                # First ensure the mask array is boolean (not int).
                mask = mask.astype(bool)
                result = ma.masked_all(mask.shape, dtype=dtype)
                # Apply the fill-value from the proxy object.
                # Note: 'values' is just 'proxy' in a dask wrapper.  This arg
                # must be a dask type so that 'delayed' can recognise it, but
                # that provides no access to the underlying fill value.
                result.fill_value = proxy.mdi
                n_values = np.sum(mask)
                if n_values > 0:
                    # Note: data field can have excess values, but not fewer.
                    result[mask] = values[:n_values]
                return result

            delayed_result = calc_array(mask_field_array, delayed_valid_values)
            meta = np.ma.array(np.empty((0,) * proxy.ndim, dtype=dtype), mask=True)
            lazy_result_array = da.from_delayed(
                delayed_result,
                shape=block_shape,
                dtype=dtype,
                meta=meta,
            )
            field.data = lazy_result_array


def _field_gen(filename, read_data_bytes, little_ended=False):
    """Return generator of "half-formed" PPField instances derived from given filename.

    A field returned by the generator is only "half-formed" because its
    `_data` attribute represents a simple one-dimensional stream of
    bytes. (Encoded either as an instance of LoadedArrayBytes or as a
    'deferred array bytes' tuple, depending on the value of `read_data_bytes`.)
    This is because fields encoded with a land/sea mask do not contain
    sufficient information within the field to determine the final
    two-dimensional shape of the data.

    The callers of this routine are the 'load' routines (both PP and FF).
    They both filter the resulting stream of fields with `_interpret_fields`,
    which replaces each field.data with an actual array (real or lazy).
    This is done separately so that `_interpret_fields` can detect land-mask
    fields and use them to construct data arrays for any fields which use
    land/sea-mask packing.

    """
    dtype_endian_char = "<" if little_ended else ">"
    with open(filename, "rb") as pp_file:
        # Get a reference to the seek method on the file
        # (this is accessed 3* #number of headers so can provide a small
        # performance boost)
        pp_file_seek = pp_file.seek
        pp_file_read = pp_file.read

        field_count = 0
        # Keep reading until we reach the end of file
        while True:
            # Move past the leading header length word
            pp_file_seek(PP_WORD_DEPTH, os.SEEK_CUR)
            # Get the LONG header entries
            dtype = "%ci%d" % (dtype_endian_char, PP_WORD_DEPTH)
            header_longs = np.fromfile(pp_file, dtype=dtype, count=NUM_LONG_HEADERS)
            # Nothing returned => EOF
            if len(header_longs) == 0:
                break
            # Get the FLOAT header entries
            dtype = "%cf%d" % (dtype_endian_char, PP_WORD_DEPTH)
            header_floats = np.fromfile(pp_file, dtype=dtype, count=NUM_FLOAT_HEADERS)
            header = tuple(header_longs) + tuple(header_floats)

            # Make a PPField of the appropriate sub-class (depends on header
            # release number)
            try:
                pp_field = make_pp_field(header)
            except ValueError as e:
                msg = (
                    "Unable to interpret field {}. {}. Skipping "
                    "the remainder of the file.".format(field_count, str(e))
                )
                warnings.warn(
                    msg,
                    category=_WarnComboIgnoringLoad,
                )
                break

            # Skip the trailing 4-byte word containing the header length
            pp_file_seek(PP_WORD_DEPTH, os.SEEK_CUR)

            # Read the word telling me how long the data + extra data is
            # This value is # of bytes
            len_of_data_plus_extra = struct.unpack_from(
                "%cL" % dtype_endian_char, pp_file_read(PP_WORD_DEPTH)
            )[0]
            if len_of_data_plus_extra != pp_field.lblrec * PP_WORD_DEPTH:
                wmsg = (
                    "LBLREC has a different value to the integer recorded "
                    "after the header in the file ({} and {}). "
                    "Skipping the remainder of the file."
                )
                warnings.warn(
                    wmsg.format(
                        pp_field.lblrec * PP_WORD_DEPTH, len_of_data_plus_extra
                    ),
                    category=_WarnComboIgnoringLoad,
                )
                break

            # calculate the extra length in bytes
            extra_len = pp_field.lbext * PP_WORD_DEPTH

            # Derive size and datatype of payload
            data_len = len_of_data_plus_extra - extra_len
            dtype = LBUSER_DTYPE_LOOKUP.get(
                pp_field.lbuser[0], LBUSER_DTYPE_LOOKUP["default"]
            )
            if little_ended:
                # Change data dtype for a little-ended file.
                dtype = str(dtype)
                if dtype[0] != ">":
                    msg = "Unexpected dtype {!r} can't be converted to little-endian"
                    raise ValueError(msg)

                dtype = np.dtype("<" + dtype[1:])

            if read_data_bytes:
                # Read the actual bytes. This can then be converted to a numpy
                # array at a higher level.
                pp_field.data = LoadedArrayBytes(pp_file.read(data_len), dtype)
            else:
                # Provide enough context to read the data bytes later on,
                # as a 'deferred array bytes' tuple.
                # N.B. this used to be a namedtuple called DeferredArrayBytes,
                # but it no longer is. Possibly for performance reasons?
                pp_field.data = (filename, pp_file.tell(), data_len, dtype)
                # Seek over the actual data payload.
                pp_file_seek(data_len, os.SEEK_CUR)

            # Do we have any extra data to deal with?
            if extra_len:
                pp_field._read_extra_data(
                    pp_file, pp_file_read, extra_len, little_ended=little_ended
                )

            # Skip that last 4 byte record telling me the length of the field I
            # have already read
            pp_file_seek(PP_WORD_DEPTH, os.SEEK_CUR)
            field_count += 1
            yield pp_field


# Stash codes not to be filtered (reference altitude and pressure fields).
_STASH_ALLOW = [STASH(1, 0, 33), STASH(1, 0, 1)]


def _convert_constraints(constraints):
    """Convert known constraints from Iris semantics to PP semantics.

    Convert known constraints from Iris semantics to PP semantics
    ignoring all unknown constraints.

    """
    constraints = iris._constraints.list_of_constraints(constraints)
    pp_constraints = {}
    unhandled_constraints = False

    def _make_func(stashobj):
        """Provide unique name-space for each lambda function's stashobj variable."""
        return lambda stash: stash == stashobj

    for con in constraints:
        if isinstance(con, iris.AttributeConstraint) and list(
            con._attributes.keys()
        ) == ["STASH"]:
            # Convert a STASH constraint.
            # The attribute can be a STASH object, a stashcode string, or a
            # callable.
            stashobj = con._attributes["STASH"]
            if callable(stashobj):
                call_func = stashobj
            elif isinstance(stashobj, (str, STASH)):
                call_func = _make_func(stashobj)
            else:
                raise TypeError(
                    "STASH constraints should be either a"
                    " callable, string or STASH object"
                )

            if "stash" not in pp_constraints:
                pp_constraints["stash"] = [call_func]
            else:
                pp_constraints["stash"].append(call_func)
        else:
            # only keep the pp constraints set if they are all handled as
            # pp constraints
            unhandled_constraints = True

    def pp_filter(field):
        """Return True if field is to be kept, False if field does not match filter."""
        res = True
        if field.stash not in _STASH_ALLOW:
            if pp_constraints.get("stash"):
                res = False
                for call_func in pp_constraints["stash"]:
                    if call_func(str(field.stash)):
                        res = True
                        break
        return res

    if pp_constraints and not unhandled_constraints:
        result = pp_filter
    else:
        result = None
    return result


def load_cubes(filenames, callback=None, constraints=None):
    """Load cubes from a list of pp filenames.

    Parameters
    ----------
    filenames :
        List of pp filenames to load.
    callback : optional
        A function which can be passed on to :func:`iris.io.run_callback`.
    constraints : optional
        A list of Iris constraints.

    Notes
    -----
    The resultant cubes may not be in the order that they are in the file
    (order is not preserved when there is a field with orography
    references)

    """
    return _load_cubes_variable_loader(
        filenames, callback, load, constraints=constraints
    )


def load_cubes_little_endian(filenames, callback=None, constraints=None):
    """Load cubes from a list of pp filenames containing little-endian data.

    Parameters
    ----------
    filenames :
        List of pp filenames to load.
    callback : optional
        A function which can be passed on to :func:`iris.io.run_callback`.
    constraints : optional
        A list of Iris constraints.

    Notes
    -----
    The resultant cubes may not be in the order that they are in the file
    (order is not preserved when there is a field with orography
    references).

    """
    return _load_cubes_variable_loader(
        filenames,
        callback,
        load,
        {"little_ended": True},
        constraints=constraints,
    )


def load_pairs_from_fields(pp_fields):
    r"""Convert an iterable of PP fields into an iterable of tuples of (Cubes, PPField).

    Parameters
    ----------
    pp_fields :
        An iterable of :class:`iris.fileformats.pp.PPField`.

    Returns
    -------
    :class:`iris.cube.Cube`
        An iterable of :class:`iris.cube.Cube`\ s.

    Notes
    -----
    This capability can be used to filter out fields before they are passed to
    the load pipeline, and amend the cubes once they are created, using
    PP metadata conditions.  Where this filtering
    removes a significant number of fields, the speed up to load can be
    significant:

        >>> import iris
        >>> from iris.fileformats.pp import load_pairs_from_fields
        >>> filename = iris.sample_data_path('E1.2098.pp')
        >>> filtered_fields = []
        >>> for field in iris.fileformats.pp.load(filename):
        ...     if field.lbproc == 128:
        ...         filtered_fields.append(field)
        >>> cube_field_pairs = load_pairs_from_fields(filtered_fields)
        >>> for cube, field in cube_field_pairs:
        ...     cube.attributes['lbproc'] = field.lbproc
        ...     print(cube.attributes['lbproc'])
        128

    This capability can also be used to alter fields before they are passed to
    the load pipeline.  Fields with out of specification header elements can
    be cleaned up this way and cubes created:

        >>> filename = iris.sample_data_path('E1.2098.pp')
        >>> cleaned_fields = list(iris.fileformats.pp.load(filename))
        >>> for field in cleaned_fields:
        ...     if field.lbrel == 0:
        ...         field.lbrel = 3
        >>> cubes_field_pairs = list(load_pairs_from_fields(cleaned_fields))

    """
    load_pairs_from_fields = iris.fileformats.rules.load_pairs_from_fields
    return load_pairs_from_fields(pp_fields, iris.fileformats.pp_load_rules.convert)


def _load_cubes_variable_loader(
    filenames,
    callback,
    loading_function,
    loading_function_kwargs=None,
    constraints=None,
):
    import iris.fileformats.um._fast_load as um_fast_load

    pp_filter = None
    if constraints is not None:
        pp_filter = _convert_constraints(constraints)
    if um_fast_load.STRUCTURED_LOAD_CONTROLS.loads_use_structured:
        # For structured loads, pass down the pp_filter function as an extra
        # keyword to the low-level generator function.
        loading_function_kwargs = loading_function_kwargs or {}
        loading_function_kwargs["pp_filter"] = pp_filter
        # Also do *not* use this filter in generic rules processing, as for
        # structured loading, the 'field' of rules processing is no longer a
        # PPField but a FieldCollation.
        pp_filter = None
        # Make a loader object for the generic rules code.
        loader = iris.fileformats.rules.Loader(
            um_fast_load._basic_load_function,
            loading_function_kwargs,
            um_fast_load._convert_collation,
        )
    else:
        loader = iris.fileformats.rules.Loader(
            loading_function,
            loading_function_kwargs or {},
            iris.fileformats.pp_load_rules.convert,
        )

    result = iris.fileformats.rules.load_cubes(filenames, callback, loader, pp_filter)

    if um_fast_load.STRUCTURED_LOAD_CONTROLS.loads_use_structured:
        # We need an additional concatenate-like operation to combine cubes
        # from different files.  Unfortunately, the 'merge' call provided in
        # the iris_load_xx functions cannot do this.
        result = um_fast_load._combine_structured_cubes(result)

    return result


def save(cube, target, append=False, field_coords=None):
    """Use the PP saving rules (and any user rules) to save a cube to a PP file.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
    target
        A filename or open file handle.
    append : bool, default=False
        Whether to start a new file afresh or add the cube(s)
        to the end of the file.
        Only applicable when target is a filename, not a file
        handle.
        Default is False.
    field_coords : optional
        List of 2 coords or coord names which are to be used
        for reducing the given cube into 2d slices,
        which will ultimately determine the x and y
        coordinates of the resulting fields.
        If None, the final two  dimensions are chosen
        for slicing.

    Notes
    -----
    See also :func:`iris.io.save`. Note that :func:`iris.save` is the preferred
    method of saving. This allows a :class:`iris.cube.CubeList` or a sequence
    of cubes to be saved to a PP file.

    """
    fields = as_fields(cube, field_coords)
    save_fields(fields, target, append=append)


def save_pairs_from_cube(cube, field_coords=None):
    """Use the PP saving rules to generate (2D cube, PP field) pairs from a cube.

    Parameters
    ----------
    cube :
        A :class:`iris.cube.Cube`.
    field_coords : optional
        List of 2 coords or coord names which are to be used for
        reducing the given cube into 2d slices, which will ultimately
        determine the x and y coordinates of the resulting fields.
        If None, the final two  dimensions are chosen for slicing.

    Yields
    ------
    :class:`iris.cube.Cube`, :class:`iris.fileformats.pp.PPField`.
        2-dimensional slices of the input cube together with their associated pp-fields.

    """
    # Open issues
    # Could use rules in "sections" ... e.g. to process the extensive
    # dimensions; ...?
    # Could pre-process the cube to add extra convenient terms?
    #    e.g. x-coord, y-coord ... but what about multiple coordinates on the
    #    dimension?

    # How to perform the slicing?
    #   Do we always slice in the last two dimensions?
    #   Not all source data will contain lat-lon slices.
    # What do we do about dimensions with multiple coordinates?

    # Deal with:
    #   LBLREC - Length of data record in words (incl. extra data)
    #       Done on save(*)
    #   LBUSER[0] - Data type
    #       Done on save(*)
    #   LBUSER[1] - Start address in DATA (?! or just set to "null"?)
    #   BLEV - Level - the value of the coordinate for LBVC

    # *) With the current on-save way of handling LBLREC and LBUSER[0] we can't
    # check if they've been set correctly without *actually* saving as a binary
    # PP file. That also means you can't use the same reference.txt file for
    # loaded vs saved fields (unless you re-load the saved field!).

    # Set to (or leave as) "null":
    #   LBEGIN - Address of start of field in direct access dataset
    #   LBEXP - Experiment identification
    #   LBPROJ - Fields file projection number
    #   LBTYP - Fields file field type code
    #   LBLEV - Fields file level code / hybrid height model level

    # Build confidence by having a PP object that records which header items
    # have been set, and only saves if they've all been set?
    #   Watch out for extra-data.

    # On the flip side, record which Cube metadata has been "used" and flag up
    # unused?

    n_dims = len(cube.shape)
    if n_dims < 2:
        raise ValueError("Unable to save a cube of fewer than 2 dimensions.")

    if field_coords is not None:
        # cast the given coord/coord names into cube coords
        field_coords = cube._as_list_of_coords(field_coords)
        if len(field_coords) != 2:
            raise ValueError(
                "Got %s coordinates in field_coords, expecting"
                " exactly 2." % len(field_coords)
            )
    else:
        # default to the last two dimensions
        # (if result of coords is an empty list, will raise an IndexError)
        # NB watch out for the ordering of the dimensions
        field_coords = (
            cube.coords(dimensions=n_dims - 2)[0],
            cube.coords(dimensions=n_dims - 1)[0],
        )

    # Save each named or latlon slice2D in the cube
    for slice2D in cube.slices(field_coords):
        # Start with a blank PPField
        pp_field = PPField3()

        # Set all items to 0 because we need lbuser, lbtim
        # and some others to be present before running the rules.
        for name, positions in pp_field.HEADER_DEFN:
            # Establish whether field name is integer or real
            default = (
                0 if positions[0] <= NUM_LONG_HEADERS - UM_TO_PP_HEADER_OFFSET else 0.0
            )
            # Establish whether field position is scalar or composite
            if len(positions) > 1:
                default = [default] * len(positions)
            setattr(pp_field, name, default)

        # Some defaults should not be 0
        pp_field.lbrel = 3  # Header release 3.
        pp_field.lbcode = 1  # Grid code.
        pp_field.bmks = 1.0  # Some scaley thing.
        pp_field.lbproc = 0
        # Set the missing-data value to the standard default value.
        # The save code uses this header word to fill masked data points.
        pp_field.bmdi = -1e30
        # From UM doc F3: "Set to -99 if LBEGIN not known"
        pp_field.lbuser[1] = -99

        # Set the data, keeping it lazy where possible.
        pp_field.data = slice2D.core_data()

        # Run the PP save rules on the slice2D, to fill the PPField,
        # recording the rules that were used
        pp_field = verify(slice2D, pp_field)

        yield (slice2D, pp_field)


def as_fields(cube, field_coords=None):
    """Use the PP saving rules to convert a cube to an iterable of PP fields.

    Use the PP saving rules (and any user rules) to convert a cube to
    an iterable of PP fields.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube`
    field_coords : optional
        List of 2 coords or coord names which are to be used for
        reducing the given cube into 2d slices, which will ultimately
        determine the x and y coordinates of the resulting fields.
        If None, the final two  dimensions are chosen for slicing.

    """
    return (field for _, field in save_pairs_from_cube(cube, field_coords=field_coords))


def save_fields(fields, target, append: bool = False):
    """Save an iterable of PP fields to a PP file.

    Parameters
    ----------
    fields :
        An iterable of PP fields.
    target :
        A filename or open file handle.
    append : bool, default=False
        Whether to start a new file afresh or add the cube(s) to the end
        of the file.
        Only applicable when target is a filename, not a file handle.
        Default is False.

    See Also
    --------
    iris.io.save :
        Save one or more Cubes to file (or other writeable).

    """
    # Open issues

    # Deal with:
    #   LBLREC - Length of data record in words (incl. extra data)
    #       Done on save(*)
    #   LBUSER[0] - Data type
    #       Done on save(*)
    #   LBUSER[1] - Start address in DATA (?! or just set to "null"?)
    #   BLEV - Level - the value of the coordinate for LBVC

    # *) With the current on-save way of handling LBLREC and LBUSER[0] we can't
    # check if they've been set correctly without *actually* saving as a binary
    # PP file. That also means you can't use the same reference.txt file for
    # loaded vs saved fields (unless you re-load the saved field!).

    # Set to (or leave as) "null":
    #   LBEGIN - Address of start of field in direct access dataset
    #   LBEXP - Experiment identification
    #   LBPROJ - Fields file projection number
    #   LBTYP - Fields file field type code
    #   LBLEV - Fields file level code / hybrid height model level

    if isinstance(target, str):
        pp_file = open(target, "ab" if append else "wb")
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" not in target.mode:
            raise ValueError("Target not binary")
        pp_file = target
    else:
        raise ValueError("Can only save pp to filename or writable")

    try:
        # Save each field
        for pp_field in fields:
            # Write to file
            pp_field.save(pp_file)
    finally:
        if isinstance(target, str):
            pp_file.close()

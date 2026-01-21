# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Module providing to netcdf datasets with automatic character encoding.

The requirement is to convert numpy fixed-width unicode arrays on writing to a variable
which is declared as a byte (character) array with a fixed-length string dimension.

Numpy unicode string arrays are ones with dtypes of the form "U<character-width>".
Numpy character variables have the dtype "S1", and map to a fixed-length "string
dimension".

In principle, netCDF4 already performs these translations, but in practice current
releases are not functional for anything other than "ascii" encoding -- including UTF-8,
which is the most obvious and desirable "general" solution.

There is also the question of whether we should like to implement UTF-8 as our default.
Current discussions on this are inconclusive and neither CF conventions nor the NetCDF
User Guide are definite on what possible values of "_Encoding" are, or what the effective
default is, even though they do both mention the "_Encoding" attribute as a potential
way to handle the issue.

Because of this, we interpret as follows:
  * when reading bytes : in the absence of an "_Encoding" attribute, we will attempt to
    decode bytes as UTF-8
  * when writing strings : in the absence of an "_Encoding" attribute (on the Iris
    cube or coord object), we will attempt to encode data with "ascii" : If this fails,
    it raise an error prompting the user to supply an "_Encoding" attribute.

Where an "_Encoding" attribute is provided to Iris, we will honour it where possible,
identifying with "codecs.lookup" :  This means we support the encodings in the Python
Standard Library, and the name aliases which it recognises.

See:

* known problems https://github.com/Unidata/netcdf4-python/issues/1440
* suggestions for how this "ought" to work, discussed in the netcdf-c library
   * https://github.com/Unidata/netcdf-c/issues/402

"""

import codecs
import contextlib
import dataclasses
import threading
import warnings

import numpy as np

from iris.fileformats.netcdf._thread_safe_nc import (
    DatasetWrapper,
    NetCDFDataProxy,
    NetCDFWriteProxy,
    VariableWrapper,
)
import iris.warnings
from iris.warnings import IrisCfLoadWarning, IrisCfSaveWarning


def decode_bytesarray_to_stringarray(
    byte_array: np.ndarray, encoding: str, string_width: int
) -> np.ndarray:
    """Convert an array of bytes to an array of strings, with one less dimension.

    N.B. for now at least, we assume the string dim is **always the last one**.
    If 'string_width' is not given, it is set to the final dimension of 'byte_array'.
    """
    if np.ma.isMaskedArray(byte_array):
        # netCDF4-python sees zeros as "missing" -- we don't need or want that
        byte_array = byte_array.data
    bytes_shape = byte_array.shape
    var_shape = bytes_shape[:-1]
    string_dtype = f"U{string_width}"
    result = np.empty(var_shape, dtype=string_dtype)
    for ndindex in np.ndindex(var_shape):
        element_bytes = byte_array[ndindex]
        bytes = b"".join([b if b else b"\0" for b in element_bytes])
        string = bytes.decode(encoding)
        result[ndindex] = string
    return result


def encode_stringarray_as_bytearray(
    data: np.typing.ArrayLike, encoding: str, string_dimension_length: int
) -> np.ndarray:
    """Encode strings as a bytes array."""
    data = np.asanyarray(data)
    element_shape = data.shape
    result = np.zeros(element_shape + (string_dimension_length,), dtype="S1")
    right_pad = b"\0" * string_dimension_length
    for index in np.ndindex(element_shape):
        string = data[index]
        bytes = string.encode(encoding=encoding)
        n_bytes = len(bytes)
        # TODO: may want to issue warning or error if we overflow the length?
        if n_bytes > string_dimension_length:
            from iris.exceptions import TranslationError

            msg = (
                f"String {string!r} written to netcdf exceeds string dimension after "
                f"encoding : {n_bytes} > {string_dimension_length}."
            )
            raise TranslationError(msg)

        # It's all a bit nasty ...
        bytes = (bytes + right_pad)[:string_dimension_length]
        result[index] = [bytes[i : i + 1] for i in range(string_dimension_length)]

    return result


@dataclasses.dataclass
class VariableEncoder:
    """A record of encoding details which can apply them to variable data."""

    varname: str  # just for the error messages
    dtype: np.dtype
    is_chardata: bool  # just a shortcut for the dtype test
    read_encoding: str  # *always* a valid encoding from the codecs package
    write_encoding: str  # *always* a valid encoding from the codecs package
    n_chars_dim: int  # length of associated character dimension
    string_width: int  # string lengths when viewing as strings (i.e. "Uxx")

    def __init__(self, cf_var):
        """Get all the info from an netCDF4 variable (or similar wrapper object).

        Most importantly, we do *not* store 'cf_var' : instead we extract the
        necessary information and store it in this object.
        So, this object has static state + is serialisable.
        """
        self.varname = cf_var.name
        self.dtype = cf_var.dtype
        self.is_chardata = np.issubdtype(self.dtype, np.bytes_)
        self.read_encoding = self._get_encoding(cf_var, writing=False)
        self.write_encoding = self._get_encoding(cf_var, writing=True)
        self.n_chars_dim = cf_var.group().dimensions[cf_var.dimensions[-1]].size
        self.string_width = self._get_string_width(cf_var)

    @staticmethod
    def _get_encoding(cf_var, writing=False) -> str:
        """Get the byte encoding defined for this variable (or None)."""
        result = getattr(cf_var, "_Encoding", None)
        if result is not None:
            try:
                # Accept + normalise naming of encodings
                result = codecs.lookup(result).name
                # NOTE: if encoding does not suit data, errors can occur.
                # For example, _Encoding = "ascii", with non-ascii content.
            except LookupError:
                # Unrecognised encoding name : handle this as just a warning
                msg = (
                    f"Ignoring unknown encoding for variable {cf_var.name!r}: "
                    f"_Encoding = {result!r}."
                )
                warntype = IrisCfSaveWarning if writing else IrisCfLoadWarning
                warnings.warn(msg, category=warntype)
                # Proceed as if there is no specified encoding
                result = None

        if result is None:
            if writing:
                result = DEFAULT_WRITE_ENCODING
            else:
                result = DEFAULT_READ_ENCODING
        return result

    def _get_string_width(self, cf_var) -> int:
        """Return the string-length defined for this variable."""
        # Work out the actual byte width from the parent dataset dimensions.
        strlen = self.n_chars_dim
        # Convert the string dimension length (i.e. bytes) to a sufficiently-long
        #  string width, depending on the (read) encoding used.
        encoding = self.read_encoding
        if "utf-16" in encoding:
            # Each char needs at least 2 bytes -- including a terminator char
            strlen = (strlen // 2) - 1
        elif "utf-32" in encoding:
            # Each char needs exactly 4 bytes -- including a terminator char
            strlen = (strlen // 4) - 1
        # "ELSE": assume there can be (at most) as many chars as bytes
        return strlen

    def decode_bytes_to_stringarray(self, data: np.ndarray) -> np.ndarray:
        if self.is_chardata and DECODE_TO_STRINGS_ON_READ:
            # N.B. read encoding default is UTF-8 --> a "usually safe" choice
            encoding = self.read_encoding
            strlen = self.string_width
            try:
                data = decode_bytesarray_to_stringarray(data, encoding, strlen)
            except UnicodeDecodeError as err:
                msg = (
                    f"Character data in variable {self.varname!r} could not be decoded "
                    f"with the {encoding!r} encoding.  This can be fixed by setting the "
                    "variable '_Encoding' attribute to suit the content."
                )
                raise ValueError(msg) from err

        return data

    def encode_strings_as_bytearray(self, data: np.ndarray) -> np.ndarray:
        if data.dtype.kind == "U":
            # N.B. it is also possible to pass a byte array (dtype "S1"),
            #  to be written directly, without processing.
            try:
                # N.B. write encoding *default* is "ascii" --> fails bad content
                encoding = self.write_encoding
                strlen = self.n_chars_dim
                data = encode_stringarray_as_bytearray(data, encoding, strlen)
            except UnicodeEncodeError as err:
                msg = (
                    f"String data written to netcdf character variable {self.varname!r} "
                    f"could not be represented in encoding {self.write_encoding!r}.  "
                    "This can be fixed by setting a suitable variable '_Encoding' "
                    'attribute, e.g. <variable>._Encoding="UTF-8".'
                )
                raise ValueError(msg) from err
        return data


class NetcdfStringDecodeSetting(threading.local):
    def __init__(self, perform_encoding: bool = True):
        self.set(perform_encoding)

    def set(self, perform_encoding: bool):
        self.perform_encoding = perform_encoding

    def __bool__(self):
        return self.perform_encoding

    @contextlib.contextmanager
    def context(self, perform_encoding: bool):
        old_setting = self.perform_encoding
        self.perform_encoding = perform_encoding
        yield
        self.perform_encoding = old_setting


DECODE_TO_STRINGS_ON_READ = NetcdfStringDecodeSetting()
DEFAULT_READ_ENCODING = "utf-8"
DEFAULT_WRITE_ENCODING = "ascii"


class EncodedVariable(VariableWrapper):
    """A variable wrapper that translates variable data according to byte encodings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, keys):
        self._contained_instance.set_auto_chartostring(False)
        data = super().__getitem__(keys)
        # Create a coding spec : redo every time in case "_Encoding" has changed
        encoding_spec = VariableEncoder(self._contained_instance)
        data = encoding_spec.decode_bytes_to_stringarray(data)
        return data

    def __setitem__(self, keys, data):
        data = np.asanyarray(data)
        # Create a coding spec : redo every time in case "_Encoding" has changed
        encoding_spec = VariableEncoder(self._contained_instance)
        data = encoding_spec.encode_strings_as_bytearray(data)
        super().__setitem__(keys, data)

    def set_auto_chartostring(self, onoff: bool):
        msg = "auto_chartostring is not supported by Iris 'EncodedVariable' type."
        raise TypeError(msg)


class EncodedDataset(DatasetWrapper):
    """A specialised DatasetWrapper whose variables perform byte encoding."""

    VAR_WRAPPER_CLS = EncodedVariable

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_auto_chartostring(self, onoff: bool):
        msg = "auto_chartostring is not supported by Iris 'EncodedDataset' type."
        raise TypeError(msg)


class EncodedNetCDFDataProxy(NetCDFDataProxy):
    __slots__ = NetCDFDataProxy.__slots__ + ("encoding_details",)

    def __init__(self, cf_var, *args, **kwargs):
        # When creating, also capture + record the encoding to be performed.
        kwargs["use_byte_data"] = True
        super().__init__(cf_var, *args, **kwargs)
        self.encoding_details = VariableEncoder(cf_var)

    def __getitem__(self, keys):
        data = super().__getitem__(keys)
        # Apply the optional bytes-to-strings conversion
        data = self.encoding_details.decode_bytes_to_stringarray(data)
        return data


class EncodedNetCDFWriteProxy(NetCDFWriteProxy):
    def __init__(self, filepath, cf_var, file_write_lock):
        super.__init__(filepath, cf_var, file_write_lock)
        self.encoding_details = VariableEncoder(cf_var)

    def __setitem__(self, key, data):
        data = np.asanyarray(data)
        # Apply the optional strings-to-bytes conversion
        data = self.encoding_details.encode_strings_as_bytearray(data)
        super.__setitem__(key, data)

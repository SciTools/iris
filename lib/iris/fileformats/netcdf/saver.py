# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to support the saving of Iris cubes to a NetCDF file.

.. z_reference:: iris.fileformats.netcdf.saver
   :tags: topic_load_save

   API reference

netCDF's half of saving: the file to write into, the keywords netCDF4 accepts,
and the four things netCDF does differently from a generic CF store. The CF
data model lives in :mod:`iris.fileformats.cf.saver`, which this builds on.

See : `NetCDF User's Guide <https://docs.unidata.ucar.edu/nug/current/>`_
and `netCDF4 python module <https://github.com/Unidata/netcdf4-python>`_.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""

from itertools import repeat, zip_longest
import os
from pathlib import Path
import typing
from urllib.parse import urlsplit
import warnings

import dask.array as da
import numpy as np

from iris._deprecation import warn_deprecated
from iris._lazy_data import is_lazy_data
import iris.config
import iris.fileformats.cf
from iris.fileformats.cf import saver as cf_saver
from iris.fileformats.cf.dataset import CFDatasetVariable

# Beside CFSaver, these names moved to iris.fileformats.cf.saver too, and are
# imported back rather than forwarded, so that reading them here stays silent.
# SaverFillValueWarning because it is public and spec section 4.8 does not
# relocate it. The constants because they are private - not API, so owed no
# deprecation cycle - and because code elsewhere in Iris still reads them from
# here, on the load path, where a warning per cube would be intolerable. The
# next pull request repoints those readers and this import goes with them.
from iris.fileformats.cf.saver import (
    _CF_ATTRS,
    _CF_DATA_ATTRS,
    _CF_GLOBAL_ATTRS,
    _FACTORY_DEFNS,
    _UKMO_DATA_ATTRS,
    CFSaver,
    SaverFillValueWarning,
)
from iris.fileformats.netcdf import _bytecoding_datasets as bytecoding_datasets
from iris.fileformats.netcdf._bytecoding_datasets import VariableEncoder
from iris.fileformats.netcdf._dataset import NetCDFDataset, NetCDFDatasetVariable
import iris.util
import iris.warnings

# Get the logger : shared logger for all in 'iris.fileformats.netcdf'.
from . import logger

# Avoid warnings about unused imports.
# We could use an __all__, but we don't want to maintain one here
logger
SaverFillValueWarning
_CF_ATTRS
_CF_DATA_ATTRS
_CF_GLOBAL_ATTRS
_FACTORY_DEFNS
_UKMO_DATA_ATTRS

#: Names that moved to iris.fileformats.cf.saver, and are still reachable
#: here. Spec section 4.8.
_RELOCATED_NAMES = frozenset(
    {
        "CFNameCoordMap",
        "CF_CONVENTIONS_VERSION",
        "MESH_ELEMENTS",
        "SPATIO_TEMPORAL_AXES",
    }
)


def __getattr__(name):
    """Forward a relocated name to :mod:`iris.fileformats.cf.saver`.

    .. deprecated:: 3.17
        Four saver names moved to :mod:`iris.fileformats.cf.saver`, because
        they are not specific to netCDF. Reading them here still works, and
        warns.

    Parameters
    ----------
    name : str
        The attribute being looked up. Python calls this only for names that
        are not already module attributes.

    Returns
    -------
    Any
        The object of that name in :mod:`iris.fileformats.cf.saver` - the same
        object, not a copy.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the relocated names. Raised without warning,
        so that a ``hasattr`` probe that comes back False stays quiet.

    """
    if name not in _RELOCATED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn_deprecated(
        f"iris.fileformats.netcdf.saver.{name} has moved to "
        f"iris.fileformats.cf.saver.{name}, because it is not specific to "
        "netCDF. The name here will be removed in a future release.",
        # warn_deprecated's default stacklevel=2 points at the line below,
        # inside this function - PEP 562 means that line is a genuine call
        # frame, not the user's code. One level further out reaches the
        # line that actually read the attribute.
        stacklevel=3,
    )
    return getattr(cf_saver, name)


def __dir__():
    """List module attributes, including the relocated names.

    A module ``__getattr__`` forwards reads of the relocated names, but does
    not make them discoverable: PEP 562 does not extend to ``dir()``,
    :func:`inspect.getmembers`, or tab-completion, so this fills the gap.

    Returns
    -------
    list of str
        The real module globals, unioned with the relocated names.

    """
    return sorted(set(globals()) | _RELOCATED_NAMES)


# TODO: becomes a Union once a Zarr CFDataset implementation exists.
CFVariable = NetCDFDatasetVariable


class Saver(CFSaver):
    """A manager for saving NetCDF/NcZarr files."""

    def __init__(self, filename, netcdf_format, compute=True):
        """Manage saving netcdf files.

        Also supports Zarr files in the NcZarr URL format, e.g.
        ``file:///path/to/file#mode=nczarr,file``. Note that NcZarr is limited to
        Zarr Storage Specification version 2. See the
        `NcZarr docs <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`
        for more.

        Parameters
        ----------
        filename : str or netCDF4.Dataset
            Name of the NetCDF file, or an NcZarr URL, to save the cube.
            OR a writeable object supporting the :class:`netCF4.Dataset` api.
        netcdf_format : str
            Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
            'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.
        compute : bool, default=True
            If ``True``, delayed variable saves will be completed on exit from the Saver
            context (after first closing the target file), equivalent to
            :meth:`complete()`.

            If ``False``, the file is created and closed without writing the data of
            variables for which the source data was lazy.  These writes can be
            completed later, see :meth:`delayed_completion`.

            .. note::
                If ``filename`` is an open dataset, rather than a filepath, then the
                caller must specify ``compute=False``, **close the dataset**, and
                complete delayed saving afterwards.
                If ``compute`` is ``True`` in this case, an error is raised.
                This is because lazy content must be written by delayed save operations,
                which will only succeed if the dataset can be (re-)opened for writing.
                See :func:`save`.

        Returns
        -------
        None

        Examples
        --------
        >>> import iris
        >>> from iris.fileformats.netcdf.saver import Saver
        >>> cubes = iris.load(iris.sample_data_path('atlantic_profiles.nc'))
        >>> with Saver("tmp.nc", "NETCDF4") as sman:
        ...     # Iterate through the cubelist.
        ...     for cube in cubes:
        ...         sman.write(cube)


        """
        if netcdf_format not in [
            "NETCDF4",
            "NETCDF4_CLASSIC",
            "NETCDF3_CLASSIC",
            "NETCDF3_64BIT",
        ]:
            raise ValueError("Unknown netCDF file format, got %r" % netcdf_format)

        #: Whether the save target is an NCZarr URL.
        self._is_nczarr = False
        # A list of pre-close writes for NCZarr lazy saving.
        # NCZarr cannot be reopened for deferred writes, so lazy data is
        # written before __exit__ closes the file.
        self._nczarr_writes = []

        # Detect if we were passed a pre-opened dataset (or something like one)
        self._to_open_dataset = hasattr(filename, "createVariable")
        if self._to_open_dataset:
            # We were passed a *dataset*, so we don't open (or close) one of our own.
            if compute:
                msg = (
                    "Cannot save to a user-provided dataset with 'compute=True'. "
                    "Please use 'compute=False' and complete delayed saving in the "
                    "calling code after the file is closed."
                )
                raise ValueError(msg)

            # from_existing() handles the thread-safety wrapping.
            dataset = NetCDFDataset.from_existing(filename)

            # In this case the dataset gives a filepath, not the other way around.
            filepath = dataset.location

        else:
            # Given a filepath string/path : create a dataset from that
            try:
                # Lazy import to avoid circular import overhead at module import-time.
                from iris.io import _is_nczarr_fragment

                self._is_nczarr = _is_nczarr_fragment(
                    urlsplit(str(filename)).fragment or None
                )
                if self._is_nczarr:
                    # NCZarr URLs contain a #mode= fragment; Path() strips it.
                    # Keep as a plain string and pass directly to the dataset.
                    filepath = str(filename)
                else:
                    filepath = Path(filename).absolute()
                dataset = NetCDFDataset(filepath, mode="w", netcdf_format=netcdf_format)
            except RuntimeError:
                if self._is_nczarr:
                    raise
                dir_name = Path(filepath).parent
                if not dir_name.is_dir():
                    msg = "No such file or directory: {}".format(dir_name)
                    raise IOError(msg)
                if not os.access(dir_name, os.R_OK | os.W_OK):
                    msg = "Permission denied: {}".format(filepath)
                    raise IOError(msg)
                else:
                    raise

        super().__init__(
            dataset, compute=compute, owns_dataset=not self._to_open_dataset
        )

        #: Target filepath. A Path for a local file, which is public API:
        #: the base class has only the dataset's string form of it.
        self.filepath = filepath

        # One lock for the whole file, shared with every variable of it.
        # N.B. the file-write-lock *type* actually depends on the dask scheduler type.
        #: A per-file write lock to prevent dask attempting overlapping writes.
        self.file_write_lock = dataset.write_lock

    def _flush_before_close(self) -> None:
        """Write anything that cannot be written after the dataset closes."""
        if self._nczarr_writes:
            # NCZarr lazy writes must be computed while the file is still open;
            # the deferred reopen-write pattern used for netCDF is not supported.
            sources, targets = zip(*self._nczarr_writes)
            da.store(list(sources), list(targets))

    def write(
        self,
        cube,
        local_keys=None,
        unlimited_dimensions=None,
        zlib=False,
        complevel=4,
        shuffle=True,
        fletcher32=False,
        contiguous=False,
        chunksizes=None,
        endian="native",
        least_significant_digit=None,
        packing=None,
        fill_value=None,
    ):
        """Wrap for saving cubes to a NetCDF file.

        Also supports Zarr files in the NcZarr URL format, e.g.
        ``file:///path/to/file#mode=nczarr,file``. Note that NcZarr is limited to
        Zarr Storage Specification version 2. See the
        `NcZarr docs <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`
        for more.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a NetCDF/NcZarr file.
        local_keys : iterable of str, optional
            An iterable of cube attribute keys. Any cube attributes with
            matching keys will become attributes on the data variable rather
            than global attributes.

            .. note::

                Has no effect if :attr:`iris.FUTURE.save_split_attrs` is ``True``.
        unlimited_dimensions : iterable of str and/or :class:`iris.coords.Coord`, optional
            List of coordinate names (or coordinate objects)
            corresponding to coordinate dimensions of `cube` to save with the
            NetCDF dimension variable length 'UNLIMITED'. By default, no
            unlimited dimensions are saved. Only the 'NETCDF4' format
            supports multiple 'UNLIMITED' dimensions.
        zlib : bool, default=False
            If `True`, the data will be compressed in the netCDF file using
            gzip compression (default `False`).
        complevel : int, default=4
            An integer between 1 and 9 describing the level of compression
            desired (default 4). Ignored if `zlib=False`.
        shuffle : bool, default=True
            If `True`, the HDF5 shuffle filter will be applied before
            compressing the data (default `True`). This significantly improves
            compression. Ignored if `zlib=False`.
        fletcher32 : bool, default=False
            If `True`, the Fletcher32 HDF5 checksum algorithm is activated to
            detect errors. Default `False`.
        contiguous : bool, default=False
            If `True`, the variable data is stored contiguously on disk.
            Default `False`. Setting to `True` for a variable with an unlimited
            dimension will trigger an error.
        chunksizes : tuple of int, optional
            Used to manually specify the HDF5 chunksizes for each dimension of
            the variable. A detailed discussion of HDF chunking and I/O
            performance is available
            `here <https://docs.unidata.ucar.edu/nug/current/netcdf_perf_chunking.html>`__.
            Basically, you want the chunk size for each dimension to match
            as closely as possible the size of the data block that users will
            read from the file. `chunksizes` cannot be set if `contiguous=True`.
        endian : str, default="native"
            Used to control whether the data is stored in little or big endian
            format on disk. Possible values are 'little', 'big' or 'native'
            (default). The library will automatically handle endian conversions
            when the data is read, but if the data is always going to be read
            on a computer with the opposite format as the one used to create
            the file, there may be some performance advantage to be gained by
            setting the endian-ness.
        least_significant_digit : int, optional
            If `least_significant_digit` is specified, variable data will be
            truncated (quantized). In conjunction with `zlib=True` this
            produces 'lossy', but significantly more efficient compression. For
            example, if `least_significant_digit=1`, data will be quantized
            using `numpy.around(scale*data)/scale`, where `scale = 2**bits`,
            and `bits` is determined so that a precision of 0.1 is retained (in
            this case `bits=4`). From the
            `metadata conventions <https://docs.unidata.ucar.edu/nug/current/attribute_conventions.html>`__:
            "least_significant_digit -- power of ten of the smallest decimal
            place in unpacked data that is a reliable value". Default is
            `None`, or no quantization, or 'lossless' compression.
        packing : type or str or dict or list, optional
            A numpy integer datatype (signed or unsigned) or a string that
            describes a numpy integer dtype(i.e. 'i2', 'short', 'u4') or a
            dict of packing parameters as described below. This provides
            support for netCDF data packing as described
            `here <https://docs.unidata.ucar.edu/nug/current/best_practices.html#bp_Best_Practices>`__.
            If this argument is a type (or type string), appropriate values of
            scale_factor and add_offset will be automatically calculated based
            on `cube.data` and possible masking. For more control, pass a dict
            with one or more of the following keys: `dtype` (required),
            `scale_factor` and `add_offset`. Note that automatic calculation of
            packing parameters will trigger loading of lazy data; set them
            manually using a dict to avoid this. The default is `None`, in
            which case the datatype is determined from the cube and no packing
            will occur.
        fill_value : optional
            The value to use for the `_FillValue` attribute on the netCDF
            variable. If `packing` is specified the value of `fill_value`
            should be in the domain of the packed data.

        Returns
        -------
        None.

        Notes
        -----
        The `zlib`, `complevel`, `shuffle`, `fletcher32`, `contiguous`,
        `chunksizes` and `endian` keywords are silently ignored for netCDF
        3 files that do not use HDF5.

        """
        super().write(
            cube,
            local_keys=local_keys,
            unlimited_dimensions=unlimited_dimensions,
            packing=packing,
            fill_value=fill_value,
            # Applied to every variable the cube creates, not just its data.
            compression_kwargs={
                "complevel": complevel,
                "fletcher32": fletcher32,
                "shuffle": shuffle,
                "zlib": zlib,
            },
            contiguous=contiguous,
            chunksizes=chunksizes,
            endian=endian,
            least_significant_digit=least_significant_digit,
        )

    def _patch_dataset(self, cf_patch, profile, cf_var_cube) -> None:
        """Hand the site-configured patch hook the objects it expects."""
        # A reserved iris.site_configuration hook, documented since
        # Iris 1.3 as receiving netCDF4 objects - so both arguments
        # are unwrapped, not just the dataset. A CFVariable would
        # raise on setncattr() and, worse, silently swallow
        # 'variable.name = value': it defines no __setattr__.
        cf_patch(profile, self._dataset.dataset, cf_var_cube.variable)

    def _create_string_variable(
        self, cube_or_mesh, cf_name, element, element_dims, data
    ):
        """Create the char variable that will hold this unicode string data.

        netCDF has no unicode type, so string data is stored as bytes over an
        extra trailing dimension whose length is the width of the widest
        value. The name of that dimension is appended to ``element_dims``.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved. Named in the error message.
        cf_name : str
            The name the variable is created with.
        element : :class:`iris.coords._DimensionalMetadata`
            The element being saved. Supplies the ``_Encoding`` attribute.
        element_dims : list of str
            The variable's dimension names, appended to in place.
        data : :class:`numpy.ndarray`
            The string values, used for their dtype.

        Returns
        -------
        :class:`~iris.fileformats.cf.dataset.CFDatasetVariable`

        """
        # NOTE: all we are doing here is to calculate the byte dimension length,
        # based on the dtype and any encoding attribute.
        # The actual char --> byte data *translation* is done by the variable,
        # being a _bytecoding_datasets.EncodedVariable.
        string_dimension_depth = data.dtype.itemsize

        # String content (U) instead of bytes (S).
        # For numpy strings, itemsize is **always** a multiple of 4
        if string_dimension_depth % 4 != 0:
            msg = (
                "Unexpected numpy string 'dtype.itemsize' for element "
                f"{cube_or_mesh.name()!r}: "
                f"'dtype.itemsize = {string_dimension_depth}, expected "
                "a multiple of four (always)."
            )
            raise ValueError(msg)
        nchars = string_dimension_depth // 4

        encoding_attr = element.attributes.get("_Encoding", "ascii")
        # Look this up + return a supported encoding name
        # NB implements defaults and raises a warning if given not recognised.
        encoding = bytecoding_datasets._identify_encoding(
            encoding=encoding_attr, var_name=cf_name, writing=True
        )
        width_fns = bytecoding_datasets._ENCODING_WIDTH_TRANSLATIONS[encoding]
        string_dimension_depth = width_fns.nchars_2_nbytes(nchars)

        string_dimension_name = "string%d" % string_dimension_depth

        # Determine whether to create the string length dimension.
        if string_dimension_name not in self._dataset.dimensions:
            while string_dimension_name in self._dataset.variables:
                # Also avoid collision with variable names.
                # See '_get_dim_names' for reason.
                string_dimension_name = self._increment_name(string_dimension_name)
            self._dataset.create_dimension(
                string_dimension_name, string_dimension_depth
            )

        # Add the string length dimension to the variable dimensions.
        element_dims.append(string_dimension_name)

        # Create the label coordinate variable.
        return self._dataset.create_variable(cf_name, "|S1", element_dims)

    def _backend_write(
        self,
        data: np.typing.ArrayLike,
        cf_var: CFDatasetVariable,
    ) -> bool:
        """Write the data here and now, if this backend handles it itself.

        Returns
        -------
        bool
            Whether the data was written. False leaves the caller to write it
            the ordinary way, directly or by a delayed store.

        """
        # The base method takes any CFDatasetVariable, and an override may not
        # ask for less. In this module the only variable there is, is netCDF's,
        # and the members reached below are netCDF's alone.
        cf_var = typing.cast("NetCDFDatasetVariable", cf_var)

        if cf_var.is_emulated:
            # The variable is not an actual file variable, but an emulating
            # object with an attached data array (either numpy or dask), which should
            # be copied immediately to the target.  This is used as a hook to translate
            # data to/from netcdf data container objects in other packages, such as
            # xarray.
            # See https://github.com/SciTools/iris/issues/4994 "Xarray bridge".
            if cf_var.dtype.kind == "U":
                # We also need to perform string-to-bytes encoding since, in ncdata,
                #  the emulating objects don't do this, and also don't support a
                #  'set_auto_chartostring(True)'.
                # Therefore, do here what an EncodedVariable.__setitem__ would do : ..
                #  .. get details from the file (char) variable to be written ..
                encoder = VariableEncoder.from_var(cf_var.unencoded_variable)
                #  .. apply encoding to get the bytes to write.
                data = encoder.encode_strings_as_bytearray(data)

            cf_var.emulated_data_array = data
            return True

        if is_lazy_data(data) and self._is_nczarr:
            # NCZarr cannot be reopened for deferred writes.
            # Collect here; da.store() will be called in __exit__.
            self._nczarr_writes.append((data, cf_var))
            return True

        return False


def save(
    cube,
    filename,
    netcdf_format="NETCDF4",
    local_keys=None,
    unlimited_dimensions=None,
    zlib=False,
    complevel=4,
    shuffle=True,
    fletcher32=False,
    contiguous=False,
    chunksizes=None,
    endian="native",
    least_significant_digit=None,
    packing=None,
    fill_value=None,
    compute=True,
):
    r"""Save cube(s) to a NetCDF file, given the cube and the filename.

    Also supports Zarr files in the NcZarr URL format, e.g.
    ``file:///path/to/file#mode=nczarr,file``. Note that NcZarr is limited to
    Zarr Storage Specification Version 2. See the
    `NcZarr docs <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`
    for more.

    * Iris will write CF 1.7 compliant NetCDF files.
    * **If split-attribute saving is disabled**, i.e.
      :data:`iris.FUTURE` ``.save_split_attrs`` is ``False``, then attributes
      dictionaries on each cube in the saved cube list will be compared, and common
      attributes saved as NetCDF global attributes where appropriate.

      Or, **when split-attribute saving is enabled**, then ``cube.attributes.locals``
      are always saved as attributes of data-variables, and ``cube.attributes.globals``
      are saved as global (dataset) attributes, where possible.
      Since the 2 types are now distinguished : see :class:`~iris.cube.CubeAttrsDict`.
    * Keyword arguments specifying how to save the data are applied
      to each cube. To use different settings for different cubes, use
      the NetCDF Context manager (:class:`~Saver`) directly.
    * The save process will stream the data payload to the file using dask,
      enabling large data payloads to be saved and maintaining the 'lazy'
      status of the cube's data payload, unless the netcdf_format is explicitly
      specified to be 'NETCDF3' or 'NETCDF3_CLASSIC'.

    Parameters
    ----------
    cube : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
        A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or other
        iterable of cubes to be saved to a netCDF file.
    filename : str
        Name of the NetCDF file or NcZarr URL to save the cube(s).
        **Or** an open, writeable :class:`netCDF4.Dataset`, or compatible object.

        .. note::
            When saving to a dataset, ``compute`` **must** be ``False`` :
            See the ``compute`` parameter.

    netcdf_format : str, default="NETCDF"
        Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
        'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.
    local_keys : iterable of str, optional
        An iterable of cube attribute keys. Any cube attributes with
        matching keys will become attributes on the data variable rather
        than global attributes.

        .. note::
            This is *ignored* if 'split-attribute saving' is **enabled**,
            i.e. when ``iris.FUTURE.save_split_attrs`` is ``True``.

    unlimited_dimensions : iterable of str and/or :class:`iris.coords.Coord` objects, optional
        List of coordinate names (or coordinate objects) corresponding
        to coordinate dimensions of `cube` to save with the NetCDF dimension
        variable length 'UNLIMITED'. By default, no unlimited dimensions are
        saved. Only the 'NETCDF4' format supports multiple 'UNLIMITED'
        dimensions.
    zlib : bool, default=False
        If `True`, the data will be compressed in the netCDF file using gzip
        compression (default `False`).
    complevel : int, default=4
        An integer between 1 and 9 describing the level of compression desired
        (default 4). Ignored if `zlib=False`.
    shuffle : bool, default=True
        If `True`, the HDF5 shuffle filter will be applied before compressing
        the data (default `True`). This significantly improves compression.
        Ignored if `zlib=False`.
    fletcher32 : bool, default=False
        If `True`, the Fletcher32 HDF5 checksum algorithm is activated to
        detect errors. Default `False`.
    contiguous : bool, default=False
        If `True`, the variable data is stored contiguously on disk. Default
        `False`. Setting to `True` for a variable with an unlimited dimension
        will trigger an error.
    chunksizes : tuple of int, optional
        Used to manually specify the HDF5 chunksizes for each dimension of the
        variable. A detailed discussion of HDF chunking and I/O performance is
        available
        `here <https://docs.unidata.ucar.edu/nug/current/netcdf_perf_chunking.html>`__.
        Basically, you want the chunk size for each dimension to match as
        closely as possible the size of the data block that users will read
        from the file. `chunksizes` cannot be set if `contiguous=True`.
    endian : str, default="native"
        Used to control whether the data is stored in little or big endian
        format on disk. Possible values are 'little', 'big' or 'native'
        (default). The library will automatically handle endian conversions
        when the data is read, but if the data is always going to be read on a
        computer with the opposite format as the one used to create the file,
        there may be some performance advantage to be gained by setting the
        endian-ness.
    least_significant_digit : int, optional
        If `least_significant_digit` is specified, variable data will be
        truncated (quantized). In conjunction with `zlib=True` this produces
        'lossy', but significantly more efficient compression. For example, if
        `least_significant_digit=1`, data will be quantized using
        `numpy.around(scale*data)/scale`, where `scale = 2**bits`, and `bits`
        is determined so that a precision of 0.1 is retained (in this case
        `bits=4`). From

        "least_significant_digit -- power of ten of the smallest decimal place
        in unpacked data that is a reliable value". Default is `None`, or no
        quantization, or 'lossless' compression.
    packing : type or str or dict or list, optional
        A numpy integer datatype (signed or unsigned) or a string that
        describes a numpy integer dtype (i.e. 'i2', 'short', 'u4') or a dict
        of packing parameters as described below or an iterable of such types,
        strings, or dicts. This provides support for netCDF data packing as
        described in the
        `metadata conventions <https://docs.unidata.ucar.edu/nug/current/attribute_conventions.html>`__
        If this argument is a type (or type string), appropriate values of
        scale_factor and add_offset will be automatically calculated based
        on `cube.data` and possible masking. For more control, pass a dict with
        one or more of the following keys: `dtype` (required), `scale_factor`
        and `add_offset`. Note that automatic calculation of packing parameters
        will trigger loading of lazy data; set them manually using a dict to
        avoid this. The default is `None`, in which case the datatype is
        determined from the cube and no packing will occur. If this argument is
        a list it must have the same number of elements as `cube` if `cube` is
        a :class:`iris.cube.CubeList`, or one element, and each element of
        this argument will be applied to each cube separately.
    fill_value : numeric or list, optional
        The value to use for the `_FillValue` attribute on the netCDF variable.
        If `packing` is specified the value of `fill_value` should be in the
        domain of the packed data. If this argument is a list it must have the
        same number of elements as `cube` if `cube` is a
        :class:`iris.cube.CubeList`, or a single element, and each element of
        this argument will be applied to each cube separately.
    compute : bool, default=True
        Default is ``True``, meaning complete the file immediately, and return ``None``.

        When ``False``, create the output file but don't write any lazy array content to
        its variables, such as lazy cube data or aux-coord points and bounds.
        Instead return a :class:`dask.delayed.Delayed` which, when computed, will
        stream all the lazy content via :meth:`dask.store`, to complete the file.
        Several such data saves can be performed in parallel, by passing a list of them
        into a :func:`dask.compute` call.

        .. note::
            If saving to an open dataset instead of a filepath, then the caller
            **must** specify ``compute=False``, and complete delayed saves **after
            closing the dataset**.
            This is because delayed saves may be performed in other processes : These
            must (re-)open the dataset for writing, which will fail if the file is
            still open for writing by the caller.

    Returns
    -------
    None or dask.delayed.Delayed
        If `compute=True`, returns `None`.
        Otherwise returns a :class:`dask.delayed.Delayed`, which implements delayed
        writing to fill in the variables data.

    Notes
    -----
    The `zlib`, `complevel`, `shuffle`, `fletcher32`, `contiguous`,
    `chunksizes` and `endian` keywords are silently ignored for netCDF 3
    files that do not use HDF5.

    """
    from iris.cube import Cube, CubeList

    if unlimited_dimensions is None:
        unlimited_dimensions = []

    if isinstance(cube, Cube):
        cubes = CubeList()
        cubes.append(cube)
    else:
        cubes = cube

    # Decide which cube attributes will be saved as "global" attributes
    # NOTE: in 'legacy' mode, when iris.FUTURE.save_split_attrs == False, this code
    # section derives a common value for 'local_keys', which is passed to 'Saver.write'
    # when saving each input cube.  The global attributes are then created by a call
    # to "Saver.update_global_attributes" within each 'Saver.write' call (which is
    # obviously a bit redundant!), plus an extra one to add 'Conventions'.
    # HOWEVER, in `split_attrs` mode (iris.FUTURE.save_split_attrs == False), this code
    # instead constructs a 'global_attributes' dictionary, and outputs that just once,
    # after writing all the input cubes.
    if iris.FUTURE.save_split_attrs:
        # We don't actually use 'local_keys' in this case.
        # TODO: can remove this when the iris.FUTURE.save_split_attrs is removed.
        local_keys = set()

        # Find any collisions in the cube global attributes and "demote" all those to
        # local attributes (where possible, else warn they are lost).
        # N.B. "collision" includes when not all cubes *have* that attribute.
        global_names = set()
        for cube in cubes:
            global_names |= set(cube.attributes.globals.keys())

        # Fnd any global attributes which are not the same on *all* cubes.
        def attr_values_equal(val1, val2):
            # An equality test which also works when some values are numpy arrays (!)
            return iris.util._attribute_equal(val1, val2)

        cube0 = cubes[0]
        invalid_globals = set(
            [
                attrname
                for attrname in global_names
                if not all(
                    attr_values_equal(
                        cube.attributes.globals.get(attrname),
                        cube0.attributes.globals.get(attrname),
                    )
                    for cube in cubes[1:]
                )
            ]
        )

        # Establish all the global attributes which we will write to the file (at end).
        global_attributes = {
            attr: cube0.attributes.globals.get(attr)
            for attr in global_names - invalid_globals
        }
        if invalid_globals:
            # Some cubes have different global attributes: modify cubes as required.
            warnings.warn(
                f"Saving the cube global attributes {sorted(invalid_globals)} as local "
                "(i.e. data-variable) attributes, where possible, since they are not "
                "the same on all input cubes.",
                category=iris.warnings.IrisSaveWarning,
            )
            cubes = cubes.copy()  # avoiding modifying the actual input arg.
            for i_cube in range(len(cubes)):
                # We iterate over cube *index*, so we can replace the list entries with
                # with cube *copies* -- just to avoid changing our call args.
                cube = cubes[i_cube]
                demote_attrs = set(cube.attributes.globals) & invalid_globals
                if any(demote_attrs):
                    # Catch any demoted attrs where there is already a local version
                    blocked_attrs = demote_attrs & set(cube.attributes.locals)
                    if blocked_attrs:
                        warnings.warn(
                            f"Global cube attributes {sorted(blocked_attrs)} "
                            f'of cube "{cube.name()}" were not saved, overlaid '
                            "by existing local attributes with the same names.",
                            category=iris.warnings.IrisSaveWarning,
                        )
                    demote_attrs -= blocked_attrs
                    if demote_attrs:
                        # This cube contains some 'demoted' global attributes.
                        # Replace input cube with a copy, so we can modify attributes.
                        cube = cube.copy()
                        cubes[i_cube] = cube
                        for attr in demote_attrs:
                            # move global to local
                            value = cube.attributes.globals.pop(attr)
                            cube.attributes.locals[attr] = value

    else:
        # Legacy mode: calculate "local_keys" to control which attributes are local
        # and which global.
        # TODO: when iris.FUTURE.save_split_attrs is removed, this section can also be
        # removed
        message = (
            "Saving to netcdf with legacy-style attribute handling for backwards "
            "compatibility.\n"
            "This mode is deprecated since Iris 3.8, and will eventually be removed.\n"
            "Please consider enabling the new split-attributes handling mode, by "
            "setting 'iris.FUTURE.save_split_attrs = True'."
        )
        warn_deprecated(message)

        if local_keys is None:
            local_keys = set()
        else:
            local_keys = set(local_keys)

        # Determine the attribute keys that are common across all cubes and
        # thereby extend the collection of local_keys for attributes
        # that should be attributes on data variables.
        attributes = cubes[0].attributes
        common_keys = set(attributes)
        for cube in cubes[1:]:
            keys = set(cube.attributes)
            local_keys.update(keys.symmetric_difference(common_keys))
            common_keys.intersection_update(keys)
            different_value_keys = []
            for key in common_keys:
                if not iris.util._attribute_equal(
                    attributes[key], cube.attributes[key]
                ):
                    different_value_keys.append(key)
            common_keys.difference_update(different_value_keys)
            local_keys.update(different_value_keys)

    def is_valid_packspec(p):
        """Only checks that the datatype is valid."""
        if isinstance(p, dict):
            if "dtype" in p:
                return is_valid_packspec(p["dtype"])
            else:
                msg = "The argument to packing must contain the key 'dtype'."
                raise ValueError(msg)
        elif isinstance(p, str) or isinstance(p, type) or isinstance(p, str):
            pdtype = np.dtype(p)  # Does nothing if it's already a numpy dtype
            if pdtype.kind != "i" and pdtype.kind != "u":
                msg = "The packing datatype must be a numpy integer type."
                raise ValueError(msg)
            return True
        elif p is None:
            return True
        else:
            return False

    if is_valid_packspec(packing):
        packspecs = repeat(packing)
    else:
        # Assume iterable, make sure packing is the same length as cubes.
        for cube, packspec in zip_longest(cubes, packing, fillvalue=-1):
            if cube == -1 or packspec == -1:
                msg = (
                    "If packing is a list, it must have the "
                    "same number of elements as the argument to"
                    "cube."
                )
                raise ValueError(msg)
            if not is_valid_packspec(packspec):
                msg = "Invalid packing argument: {}.".format(packspec)
                raise ValueError(msg)
        packspecs = packing

    # Make fill-value(s) into an iterable over cubes.
    if isinstance(fill_value, str):
        # Strings are awkward -- handle separately.
        fill_values = repeat(fill_value)
    else:
        try:
            fill_values = tuple(fill_value)
        except TypeError:
            fill_values = repeat(fill_value)
        else:
            if len(fill_values) != len(cubes):
                msg = (
                    "If fill_value is a list, it must have the "
                    "same number of elements as the cube argument."
                )
                raise ValueError(msg)

    # Initialise Manager for saving
    # N.B. make the Saver compute=False, as we want control over creation of the
    # delayed-completion object.
    with Saver(filename, netcdf_format, compute=compute) as sman:
        # Iterate through the cubelist.
        for cube, packspec, fill_value in zip(cubes, packspecs, fill_values):
            sman.write(
                cube,
                local_keys=local_keys,
                unlimited_dimensions=unlimited_dimensions,
                zlib=zlib,
                complevel=complevel,
                shuffle=shuffle,
                fletcher32=fletcher32,
                contiguous=contiguous,
                chunksizes=chunksizes,
                endian=endian,
                least_significant_digit=least_significant_digit,
                packing=packspec,
                fill_value=fill_value,
            )

        if iris.config.netcdf.conventions_override:
            # Set to the default if custom conventions are not available.
            conventions = cube.attributes.get(
                "Conventions", cf_saver.CF_CONVENTIONS_VERSION
            )
        else:
            conventions = cf_saver.CF_CONVENTIONS_VERSION

        # Perform a CF patch of the conventions attribute.
        cf_profile_available = iris.site_configuration.get("cf_profile") not in [
            None,
            False,
        ]
        if cf_profile_available:
            conventions_patch = iris.site_configuration.get("cf_patch_conventions")
            if conventions_patch is not None:
                conventions = conventions_patch(conventions)
            else:
                msg = "cf_profile is available but no {} defined.".format(
                    "cf_patch_conventions"
                )
                warnings.warn(msg, category=iris.warnings.IrisCfSaveWarning)

        # Add conventions attribute.
        if iris.FUTURE.save_split_attrs:
            # In the "new way", we just create all the global attributes at once.
            global_attributes["Conventions"] = conventions
            sman.update_global_attributes(global_attributes)
        else:
            sman.update_global_attributes(Conventions=conventions)

    if compute:
        # No more to do, since we used Saver(compute=True).
        result = None
    else:
        # Return a delayed completion object.
        result = sman.delayed_completion()

    return result


def save_mesh(
    mesh,
    filename,
    /,
    *,
    complevel=4,
    fletcher32=False,
    netcdf_format="NETCDF4",
    shuffle=True,
    zlib=False,
):
    """Save mesh(es) to a netCDF file.

    Parameters
    ----------
    mesh : :class:`iris.mesh.MeshXY` or iterable
        Mesh(es) to save.
    filename : str
        Name of the netCDF file to create.
    complevel : int, default=4
        An integer between 1 and 9 describing the level of compression
        desired. Ignored if ``zlib=False``.
    fletcher32 : bool, default=False
        If ``True``, the Fletcher32 HDF5 checksum algorithm is activated to
        detect errors.
    netcdf_format : str, default="NETCDF4"
        Underlying netCDF file format, one of ``NETCDF4``, ``NETCDF4_CLASSIC``,
        ``NETCDF3_CLASSIC`` or ``NETCDF3_64BIT``. Default is ``NETCDF4`` format.
    shuffle : bool, default=True
        If ``True``, the HDF5 shuffle filter will be applied before
        compressing the data. This significantly improves compression.
        Ignored if ``zlib=False``.
    zlib : bool, default=False
        If ``True``, the data will be compressed in the netCDF file using
        gzip compression.

    """
    if isinstance(mesh, typing.Iterable):
        meshes = mesh
    else:
        meshes = [mesh]

    # Group the generic compression keyword arguments together.
    compression_kwargs = {
        "complevel": complevel,
        "fletcher32": fletcher32,
        "shuffle": shuffle,
        "zlib": zlib,
    }

    # Initialise Manager for saving
    with Saver(filename, netcdf_format) as sman:
        # Iterate through the list.
        for mesh in meshes:
            # Get suitable dimension names.
            mesh_dimensions, _ = sman._get_dim_names(mesh)

            # Create dimensions.
            sman._create_cf_dimensions(cube=None, dimension_names=mesh_dimensions)

            # Create the mesh components.
            sman._add_mesh(mesh, compression_kwargs=compression_kwargs)

        # Add a conventions attribute.
        # TODO: add 'UGRID' to conventions, when this is agreed with CF ?
        sman.update_global_attributes(Conventions=cf_saver.CF_CONVENTIONS_VERSION)

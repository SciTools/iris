# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to support the saving of Iris cubes to a NetCDF file.

Module to support the saving of Iris cubes to a NetCDF file, also using the CF
conventions for metadata interpretation.

See : `NetCDF User's Guide <https://docs.unidata.ucar.edu/nug/current/>`_
and `netCDF4 python module <https://github.com/Unidata/netcdf4-python>`_.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""

import collections
from itertools import repeat, zip_longest
import os
import os.path
import re
import string
import typing
import warnings

import cf_units
import dask
import dask.array as da
from dask.delayed import Delayed
import numpy as np

from iris._deprecation import warn_deprecated
from iris._lazy_data import _co_realise_lazy_arrays, is_lazy_data
from iris.aux_factory import (
    AtmosphereSigmaFactory,
    HybridHeightFactory,
    HybridPressureFactory,
    OceanSFactory,
    OceanSg1Factory,
    OceanSg2Factory,
    OceanSigmaFactory,
    OceanSigmaZFactory,
)
import iris.config
import iris.coord_systems
import iris.coords
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
import iris.exceptions
import iris.fileformats.cf
from iris.fileformats.netcdf import _dask_locks, _thread_safe_nc
import iris.io
import iris.util
import iris.warnings

# Get the logger : shared logger for all in 'iris.fileformats.netcdf'.
from . import logger

# Avoid warning about unused import.
# We could use an __all__, but we don't want to maintain one here
logger

SPATIO_TEMPORAL_AXES = ["t", "z", "y", "x"]
"""Standard CML spatio-temporal axis names."""

# The CF-meaningful attributes which may appear on a data variable.
_CF_ATTRS = [
    "add_offset",
    "ancillary_variables",
    "axis",
    "bounds",
    "calendar",
    "cell_measures",
    "cell_methods",
    "climatology",
    "compress",
    "coordinates",
    "_FillValue",
    "formula_terms",
    "grid_mapping",
    "leap_month",
    "leap_year",
    "long_name",
    "missing_value",
    "month_lengths",
    "scale_factor",
    "standard_error_multiplier",
    "standard_name",
    "units",
]

# CF attributes that should not be global.
_CF_DATA_ATTRS = [
    "flag_masks",
    "flag_meanings",
    "flag_values",
    "instance_dimension",
    "missing_value",
    "sample_dimension",
    "standard_error_multiplier",
]

# CF attributes that should only be global.
_CF_GLOBAL_ATTRS = ["conventions", "featureType", "history", "title"]

# UKMO specific attributes that should not be global.
_UKMO_DATA_ATTRS = ["STASH", "um_stash_source", "ukmo__process_flags"]

# TODO: whenever we advance to CF-1.11 we should then discuss a completion date
#  for the deprecation of Rotated Mercator in coord_systems.py and
#  _nc_load_rules/helpers.py .
CF_CONVENTIONS_VERSION = "CF-1.7"
"""CF Conventions Version."""


_FactoryDefn = collections.namedtuple(
    "_FactoryDefn", ("primary", "std_name", "formula_terms_format")
)
_FACTORY_DEFNS = {
    AtmosphereSigmaFactory: _FactoryDefn(
        primary="sigma",
        std_name="atmosphere_sigma_coordinate",
        formula_terms_format="ptop: {pressure_at_top} sigma: {sigma} "
        "ps: {surface_air_pressure}",
    ),
    HybridHeightFactory: _FactoryDefn(
        primary="delta",
        std_name="atmosphere_hybrid_height_coordinate",
        formula_terms_format="a: {delta} b: {sigma} orog: {orography}",
    ),
    HybridPressureFactory: _FactoryDefn(
        primary="delta",
        std_name="atmosphere_hybrid_sigma_pressure_coordinate",
        formula_terms_format="ap: {delta} b: {sigma} ps: {surface_air_pressure}",
    ),
    OceanSigmaZFactory: _FactoryDefn(
        primary="zlev",
        std_name="ocean_sigma_z_coordinate",
        formula_terms_format="sigma: {sigma} eta: {eta} depth: {depth} "
        "depth_c: {depth_c} nsigma: {nsigma} zlev: {zlev}",
    ),
    OceanSigmaFactory: _FactoryDefn(
        primary="sigma",
        std_name="ocean_sigma_coordinate",
        formula_terms_format="sigma: {sigma} eta: {eta} depth: {depth}",
    ),
    OceanSFactory: _FactoryDefn(
        primary="s",
        std_name="ocean_s_coordinate",
        formula_terms_format="s: {s} eta: {eta} depth: {depth} a: {a} b: {b} "
        "depth_c: {depth_c}",
    ),
    OceanSg1Factory: _FactoryDefn(
        primary="s",
        std_name="ocean_s_coordinate_g1",
        formula_terms_format="s: {s} c: {c} eta: {eta} depth: {depth} "
        "depth_c: {depth_c}",
    ),
    OceanSg2Factory: _FactoryDefn(
        primary="s",
        std_name="ocean_s_coordinate_g2",
        formula_terms_format="s: {s} c: {c} eta: {eta} depth: {depth} "
        "depth_c: {depth_c}",
    ),
}


class CFNameCoordMap:
    """Provide a simple CF name to CF coordinate mapping."""

    _Map = collections.namedtuple("_Map", ["name", "coord"])

    def __init__(self):
        self._map = []

    def append(self, name, coord):
        """Append the given name and coordinate pair to the mapping.

        Parameters
        ----------
        name :
            CF name of the associated coordinate.
        coord :
            The coordinate of the associated CF name.

        Returns
        -------
        None.

        """
        self._map.append(CFNameCoordMap._Map(name, coord))

    @property
    def names(self):
        """Return all the CF names."""
        return [pair.name for pair in self._map]

    @property
    def coords(self):
        """Return all the coordinates."""
        return [pair.coord for pair in self._map]

    def name(self, coord):
        """Return the CF name, given a coordinate, or None if not recognised.

        Parameters
        ----------
        coord :
            The coordinate of the associated CF name.

        Returns
        -------
        Coordinate or None.

        """
        result = None
        for pair in self._map:
            if coord == pair.coord:
                result = pair.name
                break
        return result

    def coord(self, name):
        """Return the coordinate, given a CF name, or None if not recognised.

        Parameters
        ----------
        name :
            CF name of the associated coordinate, or None if not recognised.

        Returns
        -------
        CF name or None.

        """
        result = None
        for pair in self._map:
            if name == pair.name:
                result = pair.coord
                break
        return result


def _bytes_if_ascii(string):
    """Convert string to a byte string (str in py2k, bytes in py3k).

    Convert the given string to a byte string (str in py2k, bytes in py3k)
    if the given string can be encoded to ascii, else maintain the type
    of the inputted string.

    Note: passing objects without an `encode` method (such as None) will
    be returned by the function unchanged.

    """
    if isinstance(string, str):
        try:
            return string.encode(encoding="ascii")
        except (AttributeError, UnicodeEncodeError):
            pass
    return string


def _setncattr(variable, name, attribute):
    """Put the given attribute on the given netCDF4 Data type.

    Put the given attribute on the given netCDF4 Data type, casting
    attributes as we go to bytes rather than unicode.

    NOTE: variable needs to be a _thread_safe_nc._ThreadSafeWrapper subclass.

    """
    assert hasattr(variable, "THREAD_SAFE_FLAG")
    attribute = _bytes_if_ascii(attribute)
    return variable.setncattr(name, attribute)


MESH_ELEMENTS = ("node", "edge", "face")
"""This matches :class:`iris.experimental.ugrid.mesh.MeshXY.ELEMENTS`
   but in the preferred order for coord/connectivity variables in the file."""


class SaverFillValueWarning(iris.warnings.IrisSaverFillValueWarning):
    """Backwards compatible form of :class:`iris.warnings.IrisSaverFillValueWarning`."""

    # TODO: remove at the next major release.
    pass


class VariableEmulator(typing.Protocol):
    """Duck-type-hinting for a ncdata object.

    https://github.com/pp-mo/ncdata
    """

    _data_array: np.typing.ArrayLike
    shape: tuple[int, ...]


_CFVariable = typing.Union[_thread_safe_nc.VariableWrapper, VariableEmulator]


class Saver:
    """A manager for saving netcdf files."""

    def __init__(self, filename, netcdf_format, compute=True):
        """Manage saving netcdf files.

        Parameters
        ----------
        filename : str or netCDF4.Dataset
            Name of the netCDF file to save the cube.
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

        Example
        -------
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

        # All persistent variables

        self._name_coord_map = CFNameCoordMap()
        """CF name mapping with iris coordinates."""

        self._dim_names_and_coords = CFNameCoordMap()
        """Map of dimensions to characteristic coordinates with which they are identified."""

        self._coord_systems = []
        """List of grid mappings added to the file."""

        self._existing_dim = {}
        """A dictionary, listing dimension names and corresponding length."""

        self._mesh_dims = {}
        """A map from meshes to their actual file dimensions (names).
           NB: might not match those of the mesh, if they were 'incremented'."""

        self._formula_terms_cache = {}
        """dictionary, mapping formula terms to owner cf variable name."""

        self.filepath = None  # this line just for the API page -- value is set later
        """Target filepath."""

        self.compute = compute
        """Whether to complete delayed saves on exit."""

        self.file_write_lock = (
            None  # this line just for the API page -- value is set later
        )
        """The file-write-lock *type* actually depends on the dask scheduler type.
           A per-file write lock to prevent dask attempting overlapping writes."""

        # A list of delayed writes for lazy saving
        # a list of couples (source, target).
        self._delayed_writes = []

        # Detect if we were passed a pre-opened dataset (or something like one)
        self._to_open_dataset = hasattr(filename, "createVariable")
        if self._to_open_dataset:
            # We were passed a *dataset*, so we don't open (or close) one of our own.
            self._dataset = filename
            if compute:
                msg = (
                    "Cannot save to a user-provided dataset with 'compute=True'. "
                    "Please use 'compute=False' and complete delayed saving in the "
                    "calling code after the file is closed."
                )
                raise ValueError(msg)

            # Put it inside a _thread_safe_nc wrapper to ensure thread-safety.
            # Except if it already is one, since they forbid "re-wrapping".
            if not hasattr(self._dataset, "THREAD_SAFE_FLAG"):
                self._dataset = _thread_safe_nc.DatasetWrapper.from_existing(
                    self._dataset
                )

            # In this case the dataset gives a filepath, not the other way around.
            self.filepath = self._dataset.filepath()

        else:
            # Given a filepath string/path : create a dataset from that
            try:
                self.filepath = os.path.abspath(filename)
                self._dataset = _thread_safe_nc.DatasetWrapper(
                    self.filepath, mode="w", format=netcdf_format
                )
            except RuntimeError:
                dir_name = os.path.dirname(self.filepath)
                if not os.path.isdir(dir_name):
                    msg = "No such file or directory: {}".format(dir_name)
                    raise IOError(msg)
                if not os.access(dir_name, os.R_OK | os.W_OK):
                    msg = "Permission denied: {}".format(self.filepath)
                    raise IOError(msg)
                else:
                    raise

        self.file_write_lock = _dask_locks.get_worker_lock(self.filepath)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Flush any buffered data to the CF-netCDF file before closing."""
        self._dataset.sync()
        if not self._to_open_dataset:
            # Only close if the Saver created it.
            self._dataset.close()
            # Complete after closing, if required
            if self.compute:
                self.complete()

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

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
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
            `here <https://www.unidata.ucar.edu/software/netcdf/documentation/NUG/netcdf_perf_chunking.html>`__.
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
            this case `bits=4`). From
            `here <https://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml>`__:
            "least_significant_digit -- power of ten of the smallest decimal
            place in unpacked data that is a reliable value". Default is
            `None`, or no quantization, or 'lossless' compression.
        packing : type or str or dict or list, optional
            A numpy integer datatype (signed or unsigned) or a string that
            describes a numpy integer dtype(i.e. 'i2', 'short', 'u4') or a
            dict of packing parameters as described below. This provides
            support for netCDF data packing as described
            `here <https://www.unidata.ucar.edu/software/netcdf/documentation/NUG/best_practices.html#bp_Packed-Data-Values>`__.
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
        # TODO: when iris.FUTURE.save_split_attrs defaults to True, we can deprecate the
        #  "local_keys" arg, and finally remove it when we finally remove the
        #  save_split_attrs switch.
        if unlimited_dimensions is None:
            unlimited_dimensions = []

        cf_profile_available = iris.site_configuration.get("cf_profile") not in [
            None,
            False,
        ]
        if cf_profile_available:
            # Perform a CF profile of the cube. This may result in an exception
            # being raised if mandatory requirements are not satisfied.
            profile = iris.site_configuration["cf_profile"](cube)

        # Ensure that attributes are CF compliant and if possible to make them
        # compliant.
        self.check_attribute_compliance(cube, cube.dtype)
        for coord in cube.coords():
            self.check_attribute_compliance(coord, coord.dtype)

        # Get suitable dimension names.
        mesh_dimensions, cube_dimensions = self._get_dim_names(cube)

        # Create all the CF-netCDF data dimensions.
        # Put mesh dims first, then non-mesh dims in cube-occurring order.
        nonmesh_dimensions = [
            dim for dim in cube_dimensions if dim not in mesh_dimensions
        ]
        all_dimensions = mesh_dimensions + nonmesh_dimensions
        self._create_cf_dimensions(cube, all_dimensions, unlimited_dimensions)

        # Create the mesh components, if there is a mesh.
        # We do this before creating the data-var, so that mesh vars precede
        # data-vars in the file.
        cf_mesh_name = self._add_mesh(cube)

        # Create the associated cube CF-netCDF data variable.
        cf_var_cube = self._create_cf_data_variable(
            cube,
            cube_dimensions,
            local_keys,
            zlib=zlib,
            complevel=complevel,
            shuffle=shuffle,
            fletcher32=fletcher32,
            contiguous=contiguous,
            chunksizes=chunksizes,
            endian=endian,
            least_significant_digit=least_significant_digit,
            packing=packing,
            fill_value=fill_value,
        )

        # Associate any mesh with the data-variable.
        # N.B. _add_mesh cannot do this, as we want to put mesh variables
        # before data-variables in the file.
        if cf_mesh_name is not None:
            _setncattr(cf_var_cube, "mesh", cf_mesh_name)
            _setncattr(cf_var_cube, "location", cube.location)

        # Add coordinate variables.
        self._add_dim_coords(cube, cube_dimensions)

        # Add the auxiliary coordinate variables and associate the data
        # variable to them
        self._add_aux_coords(cube, cf_var_cube, cube_dimensions)

        # Add the cell_measures variables and associate the data
        # variable to them
        self._add_cell_measures(cube, cf_var_cube, cube_dimensions)

        # Add the ancillary_variables variables and associate the data variable
        # to them
        self._add_ancillary_variables(cube, cf_var_cube, cube_dimensions)

        # Add the formula terms to the appropriate cf variables for each
        # aux factory in the cube.
        self._add_aux_factories(cube, cf_var_cube, cube_dimensions)

        if not iris.FUTURE.save_split_attrs:
            # In the "old" way, we update global attributes as we go.
            # Add data variable-only attribute names to local_keys.
            if local_keys is None:
                local_keys = set()
            else:
                local_keys = set(local_keys)
            local_keys.update(_CF_DATA_ATTRS, _UKMO_DATA_ATTRS)

            # Add global attributes taking into account local_keys.
            cube_attributes = cube.attributes
            global_attributes = {
                k: v
                for k, v in cube_attributes.items()
                if (k not in local_keys and k.lower() != "conventions")
            }
            self.update_global_attributes(global_attributes)

        if cf_profile_available:
            cf_patch = iris.site_configuration.get("cf_patch")
            if cf_patch is not None:
                # Perform a CF patch of the dataset.
                cf_patch(profile, self._dataset, cf_var_cube)
            else:
                msg = "cf_profile is available but no {} defined.".format("cf_patch")
                warnings.warn(msg, category=iris.warnings.IrisCfSaveWarning)

    @staticmethod
    def check_attribute_compliance(container, data_dtype):
        """Check attributte complliance."""

        def _coerce_value(val_attr, val_attr_value, data_dtype):
            val_attr_tmp = np.array(val_attr_value, dtype=data_dtype)
            if (val_attr_tmp != val_attr_value).any():
                msg = '"{}" is not of a suitable value ({})'
                raise ValueError(msg.format(val_attr, val_attr_value))
            return val_attr_tmp

        # Ensure that conflicting attributes are not provided.
        if (
            container.attributes.get("valid_min") is not None
            or container.attributes.get("valid_max") is not None
        ) and container.attributes.get("valid_range") is not None:
            msg = (
                'Both "valid_range" and "valid_min" or "valid_max" '
                "attributes present."
            )
            raise ValueError(msg)

        # Ensure correct datatype
        for val_attr in ["valid_range", "valid_min", "valid_max"]:
            val_attr_value = container.attributes.get(val_attr)
            if val_attr_value is not None:
                val_attr_value = np.asarray(val_attr_value)
                if data_dtype.itemsize == 1:
                    # Allow signed integral type
                    if val_attr_value.dtype.kind == "i":
                        continue
                new_val = _coerce_value(val_attr, val_attr_value, data_dtype)
                container.attributes[val_attr] = new_val

    def update_global_attributes(self, attributes=None, **kwargs):
        """Update the CF global attributes.

        Update the CF global attributes based on the provided
        iterable/dictionary and/or keyword arguments.

        Parameters
        ----------
        attributes : dict or iterable of key, value pairs, optional
            CF global attributes to be updated.
        """
        # TODO: when when iris.FUTURE.save_split_attrs is removed, this routine will
        # only be called once: it can reasonably be renamed "_set_global_attributes",
        # and the 'kwargs' argument can be removed.
        if attributes is not None:
            # Handle sequence e.g. [('fruit', 'apple'), ...].
            if not hasattr(attributes, "keys"):
                attributes = dict(attributes)

            for attr_name in sorted(attributes):
                _setncattr(self._dataset, attr_name, attributes[attr_name])

        for attr_name in sorted(kwargs):
            _setncattr(self._dataset, attr_name, kwargs[attr_name])

    def _create_cf_dimensions(self, cube, dimension_names, unlimited_dimensions=None):
        """Create the CF-netCDF data dimensions.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` in which to lookup coordinates.
        dimension_names :
        unlimited_dimensions : iterable of strings and/or :class:`iris.coords.Coord` objects):
            List of coordinates to make unlimited (None by default).

        Returns
        -------
        None.

        """
        unlimited_dim_names = []
        if unlimited_dimensions is not None:
            for coord in unlimited_dimensions:
                try:
                    coord = cube.coord(name_or_coord=coord, dim_coords=True)
                except iris.exceptions.CoordinateNotFoundError:
                    # coordinate isn't used for this cube, but it might be
                    # used for a different one
                    pass
                else:
                    dim_name = self._get_coord_variable_name(cube, coord)
                    unlimited_dim_names.append(dim_name)

        for dim_name in dimension_names:
            # NOTE: these dim-names have been chosen by _get_dim_names, and
            # were already checked+fixed to avoid any name collisions.
            if dim_name not in self._dataset.dimensions:
                if dim_name in unlimited_dim_names:
                    size = None
                else:
                    size = self._existing_dim[dim_name]
                self._dataset.createDimension(dim_name, size)

    def _add_mesh(self, cube_or_mesh):
        """Add the cube's mesh, and all related variables to the dataset.

        Add the cube's mesh, and all related variables to the dataset.
        Includes all the mesh-element coordinate and connectivity variables.

        .. note::

            Here, we do *not* add the relevant referencing attributes to the
            data-variable, because we want to create the data-variable later.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.

        Returns
        -------
        str or None
            The name of the mesh variable created, or None if the cube does not
            have a mesh.

        """
        cf_mesh_name = None

        # Do cube- or -mesh-based save
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            cube = cube_or_mesh
            mesh = cube.mesh
        else:
            cube = None  # The underlying routines must support this !
            mesh = cube_or_mesh

        if mesh:
            cf_mesh_name = self._name_coord_map.name(mesh)
            if cf_mesh_name is None:
                # Not already present : create it
                cf_mesh_name = self._create_mesh(mesh)
                self._name_coord_map.append(cf_mesh_name, mesh)

                cf_mesh_var = self._dataset.variables[cf_mesh_name]

                # Get the mesh-element dim names.
                mesh_dims = self._mesh_dims[mesh]

                # Add all the element coordinate variables.
                for location in MESH_ELEMENTS:
                    coords_meshobj_attr = f"{location}_coords"
                    coords_file_attr = f"{location}_coordinates"
                    mesh_coords = getattr(mesh, coords_meshobj_attr, None)
                    if mesh_coords:
                        coord_names = []
                        for coord in mesh_coords:
                            if coord is None:
                                continue  # an awkward thing that mesh.coords does
                            coord_name = self._create_generic_cf_array_var(
                                cube_or_mesh,
                                [],
                                coord,
                                element_dims=(mesh_dims[location],),
                            )
                            # Only created once per file, but need to fetch the
                            #  name later in _add_inner_related_vars().
                            self._name_coord_map.append(coord_name, coord)
                            coord_names.append(coord_name)
                        # Record the coordinates (if any) on the mesh variable.
                        if coord_names:
                            coord_names = " ".join(coord_names)
                            _setncattr(cf_mesh_var, coords_file_attr, coord_names)

                # Add all the connectivity variables.
                # pre-fetch the set + ignore "None"s, which are empty slots.
                conns = [conn for conn in mesh.all_connectivities if conn is not None]
                for conn in conns:
                    # Get the connectivity role, = "{loc1}_{loc2}_connectivity".
                    cf_conn_attr_name = conn.cf_role
                    loc_from, loc_to, _ = cf_conn_attr_name.split("_")
                    # Construct a trailing dimension name.
                    last_dim = f"{cf_mesh_name}_{loc_from}_N_{loc_to}s"
                    # Create if it does not already exist.
                    if last_dim not in self._dataset.dimensions:
                        while last_dim in self._dataset.variables:
                            # Also avoid collision with variable names.
                            # See '_get_dim_names' for reason.
                            last_dim = self._increment_name(last_dim)
                        length = conn.shape[1 - conn.location_axis]
                        self._dataset.createDimension(last_dim, length)

                    # Create variable.
                    # NOTE: for connectivities *with missing points*, this will use a
                    # fixed standard fill-value of -1.  In that case, we create the
                    # variable with a '_FillValue' property, which can only be done
                    # when it is first created.
                    loc_dim_name = mesh_dims[loc_from]
                    conn_dims = (loc_dim_name, last_dim)
                    if conn.location_axis == 1:
                        # Has the 'other' dimension order, =reversed
                        conn_dims = conn_dims[::-1]
                    if iris.util.is_masked(conn.core_indices()):
                        # Flexible mesh.
                        fill_value = -1
                    else:
                        fill_value = None
                    cf_conn_name = self._create_generic_cf_array_var(
                        cube_or_mesh,
                        [],
                        conn,
                        element_dims=conn_dims,
                        fill_value=fill_value,
                    )
                    # Add essential attributes to the Connectivity variable.
                    cf_conn_var = self._dataset.variables[cf_conn_name]
                    _setncattr(cf_conn_var, "cf_role", cf_conn_attr_name)
                    _setncattr(cf_conn_var, "start_index", conn.start_index)

                    # Record the connectivity on the parent mesh var.
                    _setncattr(cf_mesh_var, cf_conn_attr_name, cf_conn_name)
                    # If the connectivity had the 'alternate' dimension order, add the
                    # relevant dimension property
                    if conn.location_axis == 1:
                        loc_dim_attr = f"{loc_from}_dimension"
                        # Should only get here once.
                        assert loc_dim_attr not in cf_mesh_var.ncattrs()
                        _setncattr(cf_mesh_var, loc_dim_attr, loc_dim_name)

        return cf_mesh_name

    def _add_inner_related_vars(
        self, cube, cf_var_cube, dimension_names, coordlike_elements
    ):
        """Create a set of variables for aux-coords, ancillaries or cell-measures.

        Create a set of variables for aux-coords, ancillaries or cell-measures,
        and attach them to the parent data variable.

        """
        if coordlike_elements:
            # Choose the appropriate parent attribute
            elem_type = type(coordlike_elements[0])
            if elem_type in (AuxCoord, DimCoord):
                role_attribute_name = "coordinates"
            elif elem_type == AncillaryVariable:
                role_attribute_name = "ancillary_variables"
            else:
                # We *only* handle aux-coords, cell-measures and ancillaries
                assert elem_type == CellMeasure
                role_attribute_name = "cell_measures"

            # Add CF-netCDF variables for the given cube components.
            element_names = []
            for element in sorted(
                coordlike_elements, key=lambda element: element.name()
            ):
                # Reuse, or create, the associated CF-netCDF variable.
                cf_name = self._name_coord_map.name(element)
                if cf_name is None:
                    # Not already present : create it
                    cf_name = self._create_generic_cf_array_var(
                        cube, dimension_names, element
                    )
                    self._name_coord_map.append(cf_name, element)

                if role_attribute_name == "cell_measures":
                    # In the case of cell-measures, the attribute entries are not just
                    # a var_name, but each have the form "<measure>: <varname>".
                    cf_name = "{}: {}".format(element.measure, cf_name)

                element_names.append(cf_name)

            # Add CF-netCDF references to the primary data variable.
            if element_names:
                variable_names = " ".join(sorted(element_names))
                _setncattr(cf_var_cube, role_attribute_name, variable_names)

    def _add_aux_coords(self, cube, cf_var_cube, dimension_names):
        """Add aux. coordinate to the dataset and associate with the data variable.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`netcdf.netcdf_variable`
            A cf variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        from iris.mesh.components import (
            MeshEdgeCoords,
            MeshFaceCoords,
            MeshNodeCoords,
            MeshXY,
        )

        # Exclude any mesh coords, which are bundled in with the aux-coords.
        coords_to_add = [
            coord for coord in cube.aux_coords if not hasattr(coord, "mesh")
        ]

        # Include any relevant mesh location coordinates.
        mesh: MeshXY | None = getattr(cube, "mesh")
        mesh_location: str | None = getattr(cube, "location")
        if mesh and mesh_location:
            location_coords: MeshNodeCoords | MeshEdgeCoords | MeshFaceCoords = getattr(
                mesh, f"{mesh_location}_coords"
            )
            coords_to_add.extend(list(location_coords))

        return self._add_inner_related_vars(
            cube,
            cf_var_cube,
            dimension_names,
            coords_to_add,
        )

    def _add_cell_measures(self, cube, cf_var_cube, dimension_names):
        """Add cell measures to the dataset and associate with the data variable.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`netcdf.netcdf_variable`
            A cf variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        return self._add_inner_related_vars(
            cube,
            cf_var_cube,
            dimension_names,
            cube.cell_measures(),
        )

    def _add_ancillary_variables(self, cube, cf_var_cube, dimension_names):
        """Add ancillary variables measures to the dataset and associate with the data variable.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`netcdf.netcdf_variable`
            A cf variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        return self._add_inner_related_vars(
            cube,
            cf_var_cube,
            dimension_names,
            cube.ancillary_variables(),
        )

    def _add_dim_coords(self, cube, dimension_names):
        """Add coordinate variables to NetCDF dataset.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        # Ensure we create the netCDF coordinate variables first.
        for coord in cube.dim_coords:
            # Create the associated coordinate CF-netCDF variable.
            if coord not in self._name_coord_map.coords:
                # Not already present : create it
                cf_name = self._create_generic_cf_array_var(
                    cube, dimension_names, coord
                )
                self._name_coord_map.append(cf_name, coord)

    def _add_aux_factories(self, cube, cf_var_cube, dimension_names):
        """Represent the presence of dimensionless vertical coordinates.

        Modify the variables of the NetCDF dataset to represent
        the presence of dimensionless vertical coordinates based on
        the aux factories of the cube (if any).

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`iris.cube.Cube` to be saved to a netCDF file.
        cf_var_cube : :class:`netcdf.netcdf_variable`
            CF variable cube representation.
        dimension_names : list
            Names associated with the dimensions of the cube.
        """
        primaries = []
        for factory in cube.aux_factories:
            factory_defn = _FACTORY_DEFNS.get(type(factory), None)
            if factory_defn is None:
                msg = "Unable to determine formula terms for AuxFactory: {!r}".format(
                    factory
                )
                warnings.warn(msg, category=iris.warnings.IrisSaveWarning)
            else:
                # Override `standard_name`, `long_name`, and `axis` of the
                # primary coord that signals the presence of a dimensionless
                # vertical coord, then set the `formula_terms` attribute.
                primary_coord = factory.dependencies[factory_defn.primary]
                if primary_coord in primaries:
                    msg = (
                        "Cube {!r} has multiple aux factories that share "
                        "a common primary coordinate {!r}. Unable to save "
                        "to netCDF as having multiple formula terms on a "
                        "single coordinate is not supported."
                    )
                    raise ValueError(msg.format(cube, primary_coord.name()))
                primaries.append(primary_coord)

                cf_name = self._name_coord_map.name(primary_coord)
                cf_var = self._dataset.variables[cf_name]

                names = {
                    key: self._name_coord_map.name(coord)
                    for key, coord in factory.dependencies.items()
                }
                formula_terms = factory_defn.formula_terms_format.format(**names)
                std_name = factory_defn.std_name

                if hasattr(cf_var, "formula_terms"):
                    if (
                        cf_var.formula_terms != formula_terms
                        or cf_var.standard_name != std_name
                    ):
                        # TODO: We need to resolve this corner-case where
                        #  the dimensionless vertical coordinate containing
                        #  the formula_terms is a dimension coordinate of
                        #  the associated cube and a new alternatively named
                        #  dimensionless vertical coordinate is required
                        #  with new formula_terms and a renamed dimension.
                        if cf_name in dimension_names:
                            msg = "Unable to create dimensonless vertical coordinate."
                            raise ValueError(msg)
                        key = (cf_name, std_name, formula_terms)
                        name = self._formula_terms_cache.get(key)
                        if name is None:
                            # Create a new variable
                            name = self._create_generic_cf_array_var(
                                cube, dimension_names, primary_coord
                            )
                            cf_var = self._dataset.variables[name]
                            _setncattr(cf_var, "standard_name", std_name)
                            _setncattr(cf_var, "axis", "Z")
                            # Update the formula terms.
                            ft = formula_terms.split()
                            ft = [name if t == cf_name else t for t in ft]
                            _setncattr(cf_var, "formula_terms", " ".join(ft))
                            # Update the cache.
                            self._formula_terms_cache[key] = name
                        # Update the associated cube variable.
                        coords = cf_var_cube.coordinates.split()
                        coords = [name if c == cf_name else c for c in coords]
                        _setncattr(cf_var_cube, "coordinates", " ".join(coords))
                else:
                    _setncattr(cf_var, "standard_name", std_name)
                    _setncattr(cf_var, "axis", "Z")
                    _setncattr(cf_var, "formula_terms", formula_terms)

    def _get_dim_names(self, cube_or_mesh):
        """Determine suitable CF-netCDF data dimension names.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.

        Returns
        -------
        mesh_dimensions : list of str
            A list of the mesh dimensions of the attached mesh, if any.
        cube_dimensions : list of str
            A lists of dimension names for each dimension of the cube.

        Notes
        -----
        The returned lists are in the preferred file creation order.
        One of the mesh dimensions will typically also appear in the cube
        dimensions.

        """

        def record_dimension(names_list, dim_name, length, matching_coords=None):
            """Record a file dimension, its length and associated "coordinates".

            Record a file dimension, its length and associated "coordinates"
            (which may in fact also be connectivities).

            If the dimension has been seen already, check that it's length
            matches the earlier finding.

            """
            if matching_coords is None:
                matching_coords = []
            if dim_name not in self._existing_dim:
                self._existing_dim[dim_name] = length
            else:
                # Make sure we never re-write one, though it's really not
                # clear how/if this could ever actually happen.
                existing_length = self._existing_dim[dim_name]
                if length != existing_length:
                    msg = (
                        "Netcdf saving error : existing dimension "
                        f'"{dim_name}" has length {existing_length}, '
                        f"but occurrence in cube {cube} has length {length}"
                    )
                    raise ValueError(msg)

            # Record given "coords" (sort-of, maybe connectivities) to be
            # identified with this dimension: add to the already-seen list.
            existing_coords = self._dim_names_and_coords.coords
            for coord in matching_coords:
                if coord not in existing_coords:
                    self._dim_names_and_coords.append(dim_name, coord)

            # Add the latest name to the list passed in.
            names_list.append(dim_name)

        # Choose cube or mesh saving.
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            # there is a mesh, only if the cube has one
            cube = cube_or_mesh
            mesh = cube.mesh
        else:
            # there is no cube, only a mesh
            cube = None
            mesh = cube_or_mesh

        # Get info on mesh, first.
        mesh_dimensions = []
        if mesh is None:
            cube_mesh_dim = None
        else:
            # Identify all the mesh dimensions.
            mesh_location_dimnames = {}
            # NOTE: one of these will be a cube dimension, but that one does not
            # get any special handling.  We *do* want to list/create them in a
            # definite order (node,edge,face), and before non-mesh dimensions.
            for location in MESH_ELEMENTS:
                # Find if this location exists in the mesh, and a characteristic
                # coordinate to identify it with.
                # To use only _required_ UGRID components, we use a location
                # coord for nodes, but a connectivity for faces/edges
                if location == "node":
                    # For nodes, identify the dim with a coordinate variable.
                    # Selecting the X-axis one for definiteness.
                    dim_coords = mesh.coords(location="node", axis="x")
                else:
                    # For face/edge, use the relevant "optionally required"
                    # connectivity variable.
                    cf_role = f"{location}_node_connectivity"
                    dim_coords = mesh.connectivities(cf_role=cf_role)
                if len(dim_coords) > 0:
                    # As the mesh contains this location, we want to include this
                    # dim in our returned mesh dims.
                    # We should have 1 identifying variable (of either type).
                    assert len(dim_coords) == 1
                    dim_element = dim_coords[0]
                    dim_name = self._dim_names_and_coords.name(dim_element)
                    if dim_name is None:
                        # No existing dim matches this, so assign a new name
                        if location == "node":
                            # always 1-d
                            (dim_length,) = dim_element.shape
                        else:
                            # extract source dim, respecting dim-ordering
                            dim_length = dim_element.shape[dim_element.location_axis]
                        # Name it for the relevant mesh dimension
                        location_dim_attr = f"{location}_dimension"
                        dim_name = getattr(mesh, location_dim_attr)
                        # NOTE: This cannot currently be empty, as a MeshXY
                        # "invents" dimension names which were not given.
                        assert dim_name is not None
                        # Ensure it is a valid variable name.
                        dim_name = self.cf_valid_var_name(dim_name)
                        # Disambiguate if it has the same name as an existing
                        # dimension.
                        # NOTE: *OR* if it matches the name of an existing file
                        # variable.  Because there is a bug ...
                        # See https://github.com/Unidata/netcdf-c/issues/1772
                        # N.B. the workarounds here *ONLY* function because the
                        # caller (write) will not create any more variables
                        # in between choosing dim names (here), and creating
                        # the new dims (via '_create_cf_dimensions').
                        while (
                            dim_name in self._existing_dim
                            or dim_name in self._dataset.variables
                        ):
                            dim_name = self._increment_name(dim_name)

                        # Record the new dimension.
                        record_dimension(
                            mesh_dimensions, dim_name, dim_length, dim_coords
                        )

                    # Store the mesh dims indexed by location
                    mesh_location_dimnames[location] = dim_name

            if cube is None:
                cube_mesh_dim = None
            else:
                # Finally, identify the cube dimension which maps to the mesh: this
                # is used below to recognise the mesh among the cube dimensions.
                any_mesh_coord = cube.coords(mesh_coords=True)[0]
                (cube_mesh_dim,) = any_mesh_coord.cube_dims(cube)

            # Record actual file dimension names for each mesh saved.
            self._mesh_dims[mesh] = mesh_location_dimnames

        # Get the cube dimensions, in order.
        cube_dimensions = []
        if cube is not None:
            for dim in range(cube.ndim):
                if dim == cube_mesh_dim:
                    # Handle a mesh dimension: we already named this.
                    dim_coords = []
                    dim_name = self._mesh_dims[mesh][cube.location]
                else:
                    # Get a name from the dim-coord (if any).
                    dim_coords = cube.coords(dimensions=dim, dim_coords=True)
                    if dim_coords:
                        # Derive a dim name from a coord.
                        coord = dim_coords[0]  # always have at least one

                        # Add only dimensions that have not already been added.
                        dim_name = self._dim_names_and_coords.name(coord)
                        if dim_name is None:
                            # Not already present : create  a unique dimension name
                            # from the coord.
                            dim_name = self._get_coord_variable_name(cube, coord)
                            # Disambiguate if it has the same name as an
                            # existing dimension.
                            # OR if it matches an existing file variable name.
                            # NOTE: check against variable names is needed
                            # because of a netcdf bug ... see note in the
                            # mesh dimensions block above.
                            while (
                                dim_name in self._existing_dim
                                or dim_name in self._dataset.variables
                            ):
                                dim_name = self._increment_name(dim_name)

                    else:
                        # No CF-netCDF coordinates describe this data dimension.
                        # Make up a new, distinct dimension name
                        dim_name = f"dim{dim}"
                        # Increment name if conflicted with one already existing
                        # (or planned)
                        # NOTE: check against variable names is needed because
                        # of a netcdf bug ... see note in the mesh dimensions
                        # block above.
                        while (
                            dim_name in self._existing_dim
                            and (self._existing_dim[dim_name] != cube.shape[dim])
                        ) or dim_name in self._dataset.variables:
                            dim_name = self._increment_name(dim_name)

                # Record the dimension.
                record_dimension(cube_dimensions, dim_name, cube.shape[dim], dim_coords)

        return mesh_dimensions, cube_dimensions

    @staticmethod
    def cf_valid_var_name(var_name):
        """Return a valid CF var_name given a potentially invalid name.

        Parameters
        ----------
        var_name : str
            The var_name to normalise.

        Returns
        -------
        str
            The var_name suitable for passing through for variable creation.

        """
        # Replace invalid characters with an underscore ("_").
        var_name = re.sub(r"[^a-zA-Z0-9]", "_", var_name)
        # Ensure the variable name starts with a letter.
        if re.match(r"^[^a-zA-Z]", var_name):
            var_name = "var_{}".format(var_name)
        return var_name

    @staticmethod
    def _cf_coord_standardised_units(coord):
        """Determine a suitable units from a given coordinate.

        Parameters
        ----------
        coord : :class:`iris.coords.Coord`
            A coordinate of a cube.

        Returns
        -------
        units
            The (standard_name, long_name, unit) of the given
            :class:`iris.coords.Coord` instance.
        """
        units = str(coord.units)
        # Set the 'units' of 'latitude' and 'longitude' coordinates specified
        # in 'degrees' to 'degrees_north' and 'degrees_east' respectively,
        # as defined in the CF conventions for netCDF files: sections 4.1 and
        # 4.2.
        if (
            isinstance(coord.coord_system, iris.coord_systems.GeogCS)
            or coord.coord_system is None
        ) and coord.units == "degrees":
            if coord.standard_name == "latitude":
                units = "degrees_north"
            elif coord.standard_name == "longitude":
                units = "degrees_east"

        return units

    def _ensure_valid_dtype(self, values, src_name, src_object):
        # NetCDF3 and NetCDF4 classic do not support int64 or unsigned ints,
        # so we check if we can store them as int32 instead.
        if (
            np.issubdtype(values.dtype, np.int64)
            or np.issubdtype(values.dtype, np.unsignedinteger)
        ) and self._dataset.file_format in (
            "NETCDF3_CLASSIC",
            "NETCDF3_64BIT",
            "NETCDF4_CLASSIC",
        ):
            val_min, val_max = (values.min(), values.max())
            if is_lazy_data(values):
                val_min, val_max = _co_realise_lazy_arrays([val_min, val_max])
            # Cast to an integer type supported by netCDF3.
            can_cast = all([np.can_cast(m, np.int32) for m in (val_min, val_max)])
            if not can_cast:
                msg = (
                    "The data type of {} {!r} is not supported by {} and"
                    " its values cannot be safely cast to a supported"
                    " integer type."
                )
                msg = msg.format(src_name, src_object, self._dataset.file_format)
                raise ValueError(msg)
            values = values.astype(np.int32)
        return values

    def _create_cf_bounds(self, coord, cf_var, cf_name):
        """Create the associated CF-netCDF bounds variable.

        Parameters
        ----------
        coord : :class:`iris.coords.Coord`
            A coordinate of a cube.
        cf_var :
            CF-netCDF variable.
        cf_name : str
            Name of the CF-NetCDF variable.

        Returns
        -------
        None

        """
        if hasattr(coord, "has_bounds") and coord.has_bounds():
            # Get the values in a form which is valid for the file format.
            bounds = self._ensure_valid_dtype(
                coord.core_bounds(), "the bounds of coordinate", coord
            )
            n_bounds = bounds.shape[-1]

            if n_bounds == 2:
                bounds_dimension_name = "bnds"
            else:
                bounds_dimension_name = "bnds_%s" % n_bounds

            if coord.climatological:
                property_name = "climatology"
                varname_extra = "climatology"
            else:
                property_name = "bounds"
                varname_extra = "bnds"

            if bounds_dimension_name not in self._dataset.dimensions:
                # Create the bounds dimension with the appropriate extent.
                while bounds_dimension_name in self._dataset.variables:
                    # Also avoid collision with variable names.
                    # See '_get_dim_names' for reason.
                    bounds_dimension_name = self._increment_name(bounds_dimension_name)
                self._dataset.createDimension(bounds_dimension_name, n_bounds)

            boundsvar_name = "{}_{}".format(cf_name, varname_extra)
            _setncattr(cf_var, property_name, boundsvar_name)
            cf_var_bounds = self._dataset.createVariable(
                boundsvar_name,
                bounds.dtype.newbyteorder("="),
                cf_var.dimensions + (bounds_dimension_name,),
            )
            self._lazy_stream_data(data=bounds, cf_var=cf_var_bounds)

    def _get_cube_variable_name(self, cube):
        """Return a CF-netCDF variable name for the given cube.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            An instance of a cube for which a CF-netCDF variable
            name is required.

        Returns
        -------
        str
            A CF-netCDF variable name as a string.

        """
        if cube.var_name is not None:
            cf_name = cube.var_name
        else:
            # Convert to lower case and replace whitespace by underscores.
            cf_name = "_".join(cube.name().lower().split())

        cf_name = self.cf_valid_var_name(cf_name)
        return cf_name

    def _get_coord_variable_name(self, cube_or_mesh, coord):
        """Return a CF-netCDF variable name for a given coordinate-like element.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.
        coord : :class:`iris.coords._DimensionalMetadata`
            An instance of a coordinate (or similar), for which a CF-netCDF
            variable name is required.

        Returns
        -------
        str
            A CF-netCDF variable name as a string.

        """
        # Support cube or mesh save.
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            cube = cube_or_mesh
            mesh = cube.mesh
        else:
            cube = None
            mesh = cube_or_mesh

        if coord.var_name is not None:
            cf_name = coord.var_name
        else:
            name = coord.standard_name or coord.long_name
            if not name or set(name).intersection(string.whitespace):
                # We need to invent a name, based on its associated dimensions.
                if cube is not None and cube.coords(coord):
                    # It is a regular cube coordinate.
                    # Auto-generate a name based on the dims.
                    name = ""
                    for dim in cube.coord_dims(coord):
                        name += f"dim{dim}"
                    # Handle scalar coordinate (dims == ()).
                    if not name:
                        name = "unknown_scalar"
                else:
                    # Not a cube coord, so must be a connectivity or
                    # element-coordinate of the mesh.
                    # Name it for it's first dim, i.e. mesh-dim of its location.

                    from iris.mesh import Connectivity

                    # At present, a location-coord cannot be nameless, as the
                    # MeshXY code relies on guess_coord_axis.
                    assert isinstance(coord, Connectivity)
                    location = coord.cf_role.split("_")[0]
                    location_dim_attr = f"{location}_dimension"
                    name = getattr(mesh, location_dim_attr)

            # Convert to lower case and replace whitespace by underscores.
            cf_name = "_".join(name.lower().split())

        cf_name = self.cf_valid_var_name(cf_name)
        return cf_name

    def _get_mesh_variable_name(self, mesh):
        """Return a CF-netCDF variable name for the mesh.

        Parameters
        ----------
        mesh : :class:`iris.mesh.MeshXY`
            An instance of a Mesh for which a CF-netCDF variable name is
            required.

        Returns
        -------
        str
            A CF-netCDF variable name as a string.

        """
        cf_name = mesh.var_name or mesh.long_name
        # Prefer a var-name, but accept a long_name as an alternative.
        # N.B. we believe it can't (shouldn't) have a standard name.
        if not cf_name:
            # Auto-generate a name based on mesh properties.
            cf_name = f"Mesh_{mesh.topology_dimension}d"

        # Ensure valid form for var-name.
        cf_name = self.cf_valid_var_name(cf_name)
        return cf_name

    def _create_mesh(self, mesh):
        """Create a mesh variable in the netCDF dataset.

        Parameters
        ----------
        mesh : :class:`iris.mesh.MeshXY`
            The Mesh to be saved to CF-netCDF file.

        Returns
        -------
        str
            The string name of the associated CF-netCDF variable saved.

        """
        # First choose a var-name for the mesh variable itself.
        cf_mesh_name = self._get_mesh_variable_name(mesh)
        # Disambiguate any possible clashes.
        while cf_mesh_name in self._dataset.variables:
            cf_mesh_name = self._increment_name(cf_mesh_name)

        # Create the main variable
        cf_mesh_var = self._dataset.createVariable(
            cf_mesh_name,
            np.dtype(np.int32),
            [],
        )

        # Add the basic essential attributes
        _setncattr(cf_mesh_var, "cf_role", "mesh_topology")
        _setncattr(
            cf_mesh_var,
            "topology_dimension",
            np.int32(mesh.topology_dimension),
        )
        # Add the usual names + units attributes
        self._set_cf_var_attributes(cf_mesh_var, mesh)

        return cf_mesh_name

    def _set_cf_var_attributes(self, cf_var, element):
        # Deal with CF-netCDF units, and add the name+units properties.
        if isinstance(element, iris.coords.Coord):
            # Fix "degree" units if needed.
            units_str = self._cf_coord_standardised_units(element)
        else:
            units_str = str(element.units)

        if cf_units.as_unit(units_str).is_udunits():
            _setncattr(cf_var, "units", units_str)

        standard_name = element.standard_name
        if standard_name is not None:
            _setncattr(cf_var, "standard_name", standard_name)

        long_name = element.long_name
        if long_name is not None:
            _setncattr(cf_var, "long_name", long_name)

        # Add the CF-netCDF calendar attribute.
        if element.units.calendar:
            _setncattr(cf_var, "calendar", str(element.units.calendar))

        # Add any other custom coordinate attributes.
        for name in sorted(element.attributes):
            value = element.attributes[name]

            if name == "STASH":
                # Adopting provisional Metadata Conventions for representing MO
                # Scientific Data encoded in NetCDF Format.
                name = "um_stash_source"
                value = str(value)

            # Don't clobber existing attributes.
            if not hasattr(cf_var, name):
                _setncattr(cf_var, name, value)

    def _create_generic_cf_array_var(
        self,
        cube_or_mesh,
        cube_dim_names,
        element,
        element_dims=None,
        fill_value=None,
    ):
        """Create theCF-netCDF variable given dimensional_metadata.

        Create the associated CF-netCDF variable in the netCDF dataset for the
        given dimensional_metadata.

        .. note::
            If the metadata element is a coord, it may also contain bounds.
            In which case, an additional var is created and linked to it.

        Parameters
        ----------
        cube_or_mesh : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The Cube or Mesh being saved to the netCDF file.
        cube_dim_names : list of str
            The name of each dimension of the cube.
        element : :class:`iris.coords._DimensionalMetadata`
            An Iris :class:`iris.coords._DimensionalMetadata`, belonging to the
            cube.  Provides data, units and standard/long/var names.
            Not used if 'element_dims' is not None.
        element_dims : list of str, optionsl
            If set, contains the variable dimension (names),
            otherwise these are taken from `element.cube_dims[cube]`.
            For Mesh components (element coordinates and connectivities), this
            *must* be passed in, as "element.cube_dims" does not function.
        fill_value : number, optional
            If set, create the variable with this fill-value, and fill any
            masked data points with this value.
            If not set, standard netcdf4-python behaviour : the variable has no
            '_FillValue' property, and uses the "standard" fill-value for its
            type.

        Returns
        -------
        str
            The name of the CF-netCDF variable created.
        """
        # Support cube or mesh save.
        from iris.cube import Cube

        if isinstance(cube_or_mesh, Cube):
            cube = cube_or_mesh
        else:
            cube = None

        # Work out the var-name to use.
        # N.B. the only part of this routine that may use a mesh _or_ a cube.
        cf_name = self._get_coord_variable_name(cube_or_mesh, element)
        while cf_name in self._dataset.variables:
            cf_name = self._increment_name(cf_name)

        if element_dims is None:
            # Get the list of file-dimensions (names), to create the variable.
            element_dims = [
                cube_dim_names[dim] for dim in element.cube_dims(cube)
            ]  # NB using 'cube_dims' as this works for any type of element

        # Get the data values, in a way which works for any element type, as
        # all are subclasses of _DimensionalMetadata.
        # (e.g. =points if a coord, =data if an ancillary, etc)
        data = element._core_values()

        if np.issubdtype(data.dtype, np.str_):
            # Deal with string-type variables.
            # Typically CF label variables, but also possibly ancil-vars ?
            string_dimension_depth = data.dtype.itemsize
            if data.dtype.kind == "U":
                string_dimension_depth //= 4
            string_dimension_name = "string%d" % string_dimension_depth

            # Determine whether to create the string length dimension.
            if string_dimension_name not in self._dataset.dimensions:
                while string_dimension_name in self._dataset.variables:
                    # Also avoid collision with variable names.
                    # See '_get_dim_names' for reason.
                    string_dimension_name = self._increment_name(string_dimension_name)
                self._dataset.createDimension(
                    string_dimension_name, string_dimension_depth
                )

            # Add the string length dimension to the variable dimensions.
            element_dims.append(string_dimension_name)

            # Create the label coordinate variable.
            cf_var = self._dataset.createVariable(cf_name, "|S1", element_dims)

            # Convert data from an array of strings into a character array
            # with an extra string-length dimension.
            if len(element_dims) == 1:
                data_first = data[0]
                if is_lazy_data(data_first):
                    data_first = data_first.compute()
                data = list("%- *s" % (string_dimension_depth, data_first))
            else:
                orig_shape = data.shape
                new_shape = orig_shape + (string_dimension_depth,)
                new_data = np.zeros(new_shape, cf_var.dtype)
                for index in np.ndindex(orig_shape):
                    index_slice = tuple(list(index) + [slice(None, None)])
                    new_data[index_slice] = list(
                        "%- *s" % (string_dimension_depth, data[index])
                    )
                data = new_data
        else:
            # A normal (numeric) variable.
            # ensure a valid datatype for the file format.
            element_type = type(element).__name__
            data = self._ensure_valid_dtype(data, element_type, element)

            # Check if this is a dim-coord.
            is_dimcoord = cube is not None and element in cube.dim_coords

            if is_dimcoord:
                # By definition of a CF-netCDF coordinate variable this
                # coordinate must be 1-D and the name of the CF-netCDF variable
                # must be the same as its dimension name.
                cf_name = element_dims[0]

            # Create the CF-netCDF variable.
            cf_var = self._dataset.createVariable(
                cf_name,
                data.dtype.newbyteorder("="),
                element_dims,
                fill_value=fill_value,
            )

            # Add the axis attribute for spatio-temporal CF-netCDF coordinates.
            if is_dimcoord:
                axis = iris.util.guess_coord_axis(element)
                if axis is not None and axis.lower() in SPATIO_TEMPORAL_AXES:
                    _setncattr(cf_var, "axis", axis.upper())

            # Create the associated CF-netCDF bounds variable, if any.
            self._create_cf_bounds(element, cf_var, cf_name)

        # Add the data to the CF-netCDF variable.
        self._lazy_stream_data(data=data, cf_var=cf_var)

        # Add names + units
        self._set_cf_var_attributes(cf_var, element)

        return cf_name

    def _create_cf_cell_methods(self, cube, dimension_names):
        """Create CF-netCDF string representation of a cube cell methods.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        dimension_names : list
            Names associated with the dimensions of the cube.

        Returns
        -------
        str
            CF-netCDF string representation of a cube cell methods.

        """
        cell_methods = []

        # Identify the collection of coordinates that represent CF-netCDF
        # coordinate variables.
        cf_coordinates = cube.dim_coords

        for cm in cube.cell_methods:
            names = ""

            for name in cm.coord_names:
                coord = cube.coords(name)

                if coord:
                    coord = coord[0]
                    if coord in cf_coordinates:
                        name = dimension_names[cube.coord_dims(coord)[0]]

                names += "%s: " % name

            interval = " ".join(
                ["interval: %s" % interval for interval in cm.intervals or []]
            )
            comment = " ".join(
                ["comment: %s" % comment for comment in cm.comments or []]
            )
            extra = " ".join([interval, comment]).strip()

            if extra:
                extra = " (%s)" % extra

            cell_methods.append(names + cm.method + extra)

        return " ".join(cell_methods)

    def _create_cf_grid_mapping(self, cube, cf_var_cube):
        """Create CF-netCDF grid mapping and associated CF-netCDF variable.

        Create CF-netCDF grid mapping variable and associated CF-netCDF
        data variable grid mapping attribute.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
            A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of
            cubes to be saved to a netCDF file.
        cf_var_cube : :class:`netcdf.netcdf_variable`
            A cf variable cube representation.

        Returns
        -------
        None

        """
        cs = cube.coord_system("CoordSystem")
        if cs is not None:
            # Grid var not yet created?
            if cs not in self._coord_systems:
                while cs.grid_mapping_name in self._dataset.variables:
                    aname = self._increment_name(cs.grid_mapping_name)
                    cs.grid_mapping_name = aname

                cf_var_grid = self._dataset.createVariable(
                    cs.grid_mapping_name, np.int32
                )
                _setncattr(cf_var_grid, "grid_mapping_name", cs.grid_mapping_name)

                def add_ellipsoid(ellipsoid):
                    cf_var_grid.longitude_of_prime_meridian = (
                        ellipsoid.longitude_of_prime_meridian
                    )
                    semi_major = ellipsoid.semi_major_axis
                    semi_minor = ellipsoid.semi_minor_axis
                    if semi_minor == semi_major:
                        cf_var_grid.earth_radius = semi_major
                    else:
                        cf_var_grid.semi_major_axis = semi_major
                        cf_var_grid.semi_minor_axis = semi_minor
                    if ellipsoid.datum is not None:
                        cf_var_grid.horizontal_datum_name = ellipsoid.datum

                # latlon
                if isinstance(cs, iris.coord_systems.GeogCS):
                    add_ellipsoid(cs)

                # rotated latlon
                elif isinstance(cs, iris.coord_systems.RotatedGeogCS):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.grid_north_pole_latitude = cs.grid_north_pole_latitude
                    cf_var_grid.grid_north_pole_longitude = cs.grid_north_pole_longitude
                    cf_var_grid.north_pole_grid_longitude = cs.north_pole_grid_longitude

                # tmerc
                elif isinstance(cs, iris.coord_systems.TransverseMercator):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_central_meridian = (
                        cs.longitude_of_central_meridian
                    )
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    cf_var_grid.scale_factor_at_central_meridian = (
                        cs.scale_factor_at_central_meridian
                    )

                # merc
                elif isinstance(cs, iris.coord_systems.Mercator):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_projection_origin = (
                        cs.longitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    # Only one of these should be set
                    if cs.standard_parallel is not None:
                        cf_var_grid.standard_parallel = cs.standard_parallel
                    elif cs.scale_factor_at_projection_origin is not None:
                        cf_var_grid.scale_factor_at_projection_origin = (
                            cs.scale_factor_at_projection_origin
                        )

                # lcc
                elif isinstance(cs, iris.coord_systems.LambertConformal):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.standard_parallel = cs.secant_latitudes
                    cf_var_grid.latitude_of_projection_origin = cs.central_lat
                    cf_var_grid.longitude_of_central_meridian = cs.central_lon
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing

                # polar stereo (have to do this before Stereographic because it subclasses it)
                elif isinstance(cs, iris.coord_systems.PolarStereographic):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.latitude_of_projection_origin = cs.central_lat
                    cf_var_grid.straight_vertical_longitude_from_pole = cs.central_lon
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    # Only one of these should be set
                    if cs.true_scale_lat is not None:
                        cf_var_grid.true_scale_lat = cs.true_scale_lat
                    elif cs.scale_factor_at_projection_origin is not None:
                        cf_var_grid.scale_factor_at_projection_origin = (
                            cs.scale_factor_at_projection_origin
                        )
                    else:
                        cf_var_grid.scale_factor_at_projection_origin = 1.0

                # stereo
                elif isinstance(cs, iris.coord_systems.Stereographic):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_projection_origin = cs.central_lon
                    cf_var_grid.latitude_of_projection_origin = cs.central_lat
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    # Only one of these should be set
                    if cs.true_scale_lat is not None:
                        msg = (
                            "It is not valid CF to save a true_scale_lat for "
                            "a Stereographic grid mapping."
                        )
                        raise ValueError(msg)
                    elif cs.scale_factor_at_projection_origin is not None:
                        cf_var_grid.scale_factor_at_projection_origin = (
                            cs.scale_factor_at_projection_origin
                        )
                    else:
                        cf_var_grid.scale_factor_at_projection_origin = 1.0

                # osgb (a specific tmerc)
                elif isinstance(cs, iris.coord_systems.OSGB):
                    warnings.warn(
                        "OSGB coordinate system not yet handled",
                        category=iris.warnings.IrisSaveWarning,
                    )

                # lambert azimuthal equal area
                elif isinstance(cs, iris.coord_systems.LambertAzimuthalEqualArea):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_projection_origin = (
                        cs.longitude_of_projection_origin
                    )
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing

                # albers conical equal area
                elif isinstance(cs, iris.coord_systems.AlbersEqualArea):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_central_meridian = (
                        cs.longitude_of_central_meridian
                    )
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    cf_var_grid.standard_parallel = cs.standard_parallels

                # vertical perspective
                elif isinstance(cs, iris.coord_systems.VerticalPerspective):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_projection_origin = (
                        cs.longitude_of_projection_origin
                    )
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    cf_var_grid.perspective_point_height = cs.perspective_point_height

                # geostationary
                elif isinstance(cs, iris.coord_systems.Geostationary):
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.longitude_of_projection_origin = (
                        cs.longitude_of_projection_origin
                    )
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    cf_var_grid.perspective_point_height = cs.perspective_point_height
                    cf_var_grid.sweep_angle_axis = cs.sweep_angle_axis

                # oblique mercator (and rotated variant)
                # Use duck-typing over isinstance() - subclasses (i.e.
                #  RotatedMercator) upset mock tests.
                elif getattr(cs, "grid_mapping_name", None) == "oblique_mercator":
                    # RotatedMercator subclasses ObliqueMercator, and RM
                    #  instances are implicitly saved as OM due to inherited
                    #  properties. This is correct because CF 1.11 is removing
                    #  all mention of RM.
                    if cs.ellipsoid:
                        add_ellipsoid(cs.ellipsoid)
                    cf_var_grid.azimuth_of_central_line = cs.azimuth_of_central_line
                    cf_var_grid.latitude_of_projection_origin = (
                        cs.latitude_of_projection_origin
                    )
                    cf_var_grid.longitude_of_projection_origin = (
                        cs.longitude_of_projection_origin
                    )
                    cf_var_grid.false_easting = cs.false_easting
                    cf_var_grid.false_northing = cs.false_northing
                    cf_var_grid.scale_factor_at_projection_origin = (
                        cs.scale_factor_at_projection_origin
                    )

                # other
                else:
                    warnings.warn(
                        "Unable to represent the horizontal "
                        "coordinate system. The coordinate system "
                        "type %r is not yet implemented." % type(cs),
                        category=iris.warnings.IrisSaveWarning,
                    )

                self._coord_systems.append(cs)

            # Refer to grid var
            _setncattr(cf_var_cube, "grid_mapping", cs.grid_mapping_name)

    def _create_cf_data_variable(
        self,
        cube,
        dimension_names,
        local_keys=None,
        packing=None,
        fill_value=None,
        **kwargs,
    ):
        """Create CF-netCDF data variable for the cube and any associated grid mapping.

        # TODO: when iris.FUTURE.save_split_attrs is removed, the 'local_keys' arg can
        # be removed.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            The associated cube being saved to CF-netCDF file.
        dimension_names : list
            String names for each dimension of the cube.
        local_keys : iterable of str, optional
            See :func:`iris.fileformats.netcdf.Saver.write`.
        packing : type or str or dict or list, optional
            See :func:`iris.fileformats.netcdf.Saver.write`.
        fill_value : optional
            See :func:`iris.fileformats.netcdf.Saver.write`.

        Notes
        -----
        All other keywords are passed through to the dataset's `createVariable`
        method.

        Returns
        -------
        The newly created CF-netCDF data variable.

        """
        # TODO: when iris.FUTURE.save_split_attrs is removed, the 'local_keys' arg can
        # be removed.
        # Get the values in a form which is valid for the file format.
        data = self._ensure_valid_dtype(cube.core_data(), "cube", cube)

        if packing:
            if isinstance(packing, dict):
                if "dtype" not in packing:
                    msg = "The dtype attribute is required for packing."
                    raise ValueError(msg)
                dtype = np.dtype(packing["dtype"])
                scale_factor = packing.get("scale_factor", None)
                add_offset = packing.get("add_offset", None)
                valid_keys = {"dtype", "scale_factor", "add_offset"}
                invalid_keys = set(packing.keys()) - valid_keys
                if invalid_keys:
                    msg = (
                        "Invalid packing key(s) found: '{}'. The valid "
                        "keys are '{}'.".format(
                            "', '".join(invalid_keys), "', '".join(valid_keys)
                        )
                    )
                    raise ValueError(msg)
            else:
                # We compute the scale_factor and add_offset based on the
                # min/max of the data.
                masked = iris.util.is_masked(data)
                dtype = np.dtype(packing)
                cmin, cmax = (data.min(), data.max())
                if is_lazy_data(data):
                    cmin, cmax = _co_realise_lazy_arrays([cmin, cmax])
                n = dtype.itemsize * 8
                if masked:
                    scale_factor = (cmax - cmin) / (2**n - 2)
                else:
                    scale_factor = (cmax - cmin) / (2**n - 1)
                if dtype.kind == "u":
                    add_offset = cmin
                elif dtype.kind == "i":
                    if masked:
                        add_offset = (cmax + cmin) / 2
                    else:
                        add_offset = cmin + 2 ** (n - 1) * scale_factor
        else:
            dtype = data.dtype.newbyteorder("=")

        def set_packing_ncattrs(cfvar):
            """Set netCDF packing attributes.

            NOTE: cfvar needs to be a _thread_safe_nc._ThreadSafeWrapper subclass.

            """
            assert hasattr(cfvar, "THREAD_SAFE_FLAG")
            if packing:
                if scale_factor:
                    _setncattr(cfvar, "scale_factor", scale_factor)
                if add_offset:
                    _setncattr(cfvar, "add_offset", add_offset)

        cf_name = self._get_cube_variable_name(cube)
        while cf_name in self._dataset.variables:
            cf_name = self._increment_name(cf_name)

        # Create the cube CF-netCDF data variable with data payload.
        cf_var = self._dataset.createVariable(
            cf_name, dtype, dimension_names, fill_value=fill_value, **kwargs
        )

        set_packing_ncattrs(cf_var)
        self._lazy_stream_data(data=data, cf_var=cf_var)

        if cube.standard_name:
            _setncattr(cf_var, "standard_name", cube.standard_name)

        if cube.long_name:
            _setncattr(cf_var, "long_name", cube.long_name)

        if cube.units.is_udunits():
            _setncattr(cf_var, "units", str(cube.units))

        # Add the CF-netCDF calendar attribute.
        if cube.units.calendar:
            _setncattr(cf_var, "calendar", cube.units.calendar)

        if iris.FUTURE.save_split_attrs:
            attr_names = cube.attributes.locals.keys()
        else:
            # Add data variable-only attribute names to local_keys.
            if local_keys is None:
                local_keys = set()
            else:
                local_keys = set(local_keys)
            local_keys.update(_CF_DATA_ATTRS, _UKMO_DATA_ATTRS)

            # Add any cube attributes whose keys are in local_keys as
            # CF-netCDF data variable attributes.
            attr_names = set(cube.attributes).intersection(local_keys)

        for attr_name in sorted(attr_names):
            # Do not output 'conventions' attribute.
            if attr_name.lower() == "conventions":
                continue

            value = cube.attributes[attr_name]

            if attr_name == "STASH":
                # Adopting provisional Metadata Conventions for representing MO
                # Scientific Data encoded in NetCDF Format.
                attr_name = "um_stash_source"
                value = str(value)

            if attr_name == "ukmo__process_flags":
                value = " ".join([x.replace(" ", "_") for x in value])

            if attr_name in _CF_GLOBAL_ATTRS:
                msg = (
                    "{attr_name!r} is being added as CF data variable "
                    "attribute, but {attr_name!r} should only be a CF "
                    "global attribute.".format(attr_name=attr_name)
                )
                warnings.warn(msg, category=iris.warnings.IrisCfSaveWarning)

            _setncattr(cf_var, attr_name, value)

        # Create the CF-netCDF data variable cell method attribute.
        cell_methods = self._create_cf_cell_methods(cube, dimension_names)

        if cell_methods:
            _setncattr(cf_var, "cell_methods", cell_methods)

        # Create the CF-netCDF grid mapping.
        self._create_cf_grid_mapping(cube, cf_var)

        return cf_var

    def _increment_name(self, varname):
        """Increment string name or begin increment.

        Avoidance of conflicts between variable names, where the name is
        incremented to distinguish it from others.

        Parameters
        ----------
        varname : str
            Variable name to increment.

        Returns
        -------
        Incremented varname.

        """
        num = 0
        try:
            name, endnum = varname.rsplit("_", 1)
            if endnum.isdigit():
                num = int(endnum) + 1
                varname = name
        except ValueError:
            pass

        return "{}_{}".format(varname, num)

    def _lazy_stream_data(
        self,
        data: np.typing.ArrayLike,
        cf_var: _CFVariable,
    ) -> None:
        if hasattr(data, "shape") and data.shape == (1,) + cf_var.shape:
            # (Don't do this check for string data).
            # Reduce dimensionality where the data array has an extra dimension
            #  versus the cf_var - to avoid a broadcasting ambiguity.
            # Happens when bounds data is for a scalar point - array is 2D but
            #  contains just 1 row, so the cf_var is 1D.
            data = data.squeeze(axis=0)

        if hasattr(cf_var, "_data_array"):
            # The variable is not an actual netCDF4 file variable, but an emulating
            # object with an attached data array (either numpy or dask), which should be
            # copied immediately to the target.  This is used as a hook to translate
            # data to/from netcdf data container objects in other packages, such as
            # xarray.
            # See https://github.com/SciTools/iris/issues/4994 "Xarray bridge".
            cf_var._data_array = data

        else:
            doing_delayed_save = is_lazy_data(data)
            if doing_delayed_save:
                # save lazy data with a delayed operation.  For now, we just record the
                # necessary information -- a single, complete delayed action is constructed
                # later by a call to delayed_completion().
                def store(
                    data: np.typing.ArrayLike,
                    cf_var: _CFVariable,
                ) -> None:
                    # Create a data-writeable object that we can stream into, which
                    # encapsulates the file to be opened + variable to be written.
                    write_wrapper = _thread_safe_nc.NetCDFWriteProxy(
                        self.filepath, cf_var, self.file_write_lock
                    )
                    # Add to the list of delayed writes, used in delayed_completion().
                    self._delayed_writes.append((data, write_wrapper))

            else:
                # Real data is always written directly, i.e. not via lazy save.
                def store(
                    data: np.typing.ArrayLike,
                    cf_var: _CFVariable,
                ) -> None:
                    cf_var[:] = data  # type: ignore[index]

            # Store the data.
            store(data, cf_var)

    def delayed_completion(self) -> Delayed:
        """Perform file completion for delayed saves.

        Create and return a :class:`dask.delayed.Delayed` to perform file
        completion for delayed saves.

        Returns
        -------
        :class:`dask.delayed.Delayed`

        Notes
        -----
        The dataset *must* be closed (saver has exited its context) before the
        result can be computed, otherwise computation will hang (never return).
        """
        if self._delayed_writes:
            # Create a single delayed da.store operation to complete the file.
            sources, targets = zip(*self._delayed_writes)
            result = da.store(sources, targets, compute=False, lock=False)

        else:
            # Return a do-nothing delayed, for usage consistency.
            @dask.delayed
            def no_op():
                return None

            result = no_op()

        return result

    def complete(self) -> None:
        """Complete file by computing any delayed variable saves.

        This requires that the Saver has closed the dataset (exited its context).

        """
        if self._dataset.isopen():
            msg = (
                "Cannot call Saver.complete() until its dataset is closed, "
                "i.e. the saver's context has exited."
            )
            raise ValueError(msg)

        # Complete the saves now
        self.delayed_completion().compute()


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
    r"""Save cube(s) to a netCDF file, given the cube and the filename.

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
        Name of the netCDF file to save the cube(s).
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
        `here <https://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml>`__.
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
        described in
        `here <https://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml>`__
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
            # As done in :meth:`iris.common.mixin.LimitedAttributeDict.__eq__`.
            match = val1 == val2
            try:
                match = bool(match)
            except ValueError:
                match = match.all()
            return match

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
                if np.any(attributes[key] != cube.attributes[key]):
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
                local_keys,
                unlimited_dimensions,
                zlib,
                complevel,
                shuffle,
                fletcher32,
                contiguous,
                chunksizes,
                endian,
                least_significant_digit,
                packing=packspec,
                fill_value=fill_value,
            )

        if iris.config.netcdf.conventions_override:
            # Set to the default if custom conventions are not available.
            conventions = cube.attributes.get("Conventions", CF_CONVENTIONS_VERSION)
        else:
            conventions = CF_CONVENTIONS_VERSION

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


def save_mesh(mesh, filename, netcdf_format="NETCDF4"):
    """Save mesh(es) to a netCDF file.

    Parameters
    ----------
    mesh : :class:`iris.mesh.MeshXY` or iterable
        Mesh(es) to save.
    filename : str
        Name of the netCDF file to create.
    netcdf_format : str, default="NETCDF4"
        Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
        'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.

    """
    if isinstance(mesh, typing.Iterable):
        meshes = mesh
    else:
        meshes = [mesh]

    # Initialise Manager for saving
    with Saver(filename, netcdf_format) as sman:
        # Iterate through the list.
        for mesh in meshes:
            # Get suitable dimension names.
            mesh_dimensions, _ = sman._get_dim_names(mesh)

            # Create dimensions.
            sman._create_cf_dimensions(cube=None, dimension_names=mesh_dimensions)

            # Create the mesh components.
            sman._add_mesh(mesh)

        # Add a conventions attribute.
        # TODO: add 'UGRID' to conventions, when this is agreed with CF ?
        sman.update_global_attributes(Conventions=CF_CONVENTIONS_VERSION)

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Automatic concatenation of multiple cubes over one or more existing dimensions."""

from collections import namedtuple
from collections.abc import Mapping, Sequence
import itertools
from typing import Any
import warnings

import dask
import dask.array as da
import numpy as np
from xxhash import xxh3_64

from iris._lazy_data import concatenate as concatenate_arrays
import iris.coords
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
import iris.cube
import iris.exceptions
from iris.util import array_equal, guess_coord_axis
import iris.warnings

#
# TODO:
#
#   * Cope with auxiliary coordinate factories.
#
#   * Allow concatenation over a user specified dimension.
#


# Restrict the names imported from this namespace.
__all__ = ["concatenate"]

# Direction of dimension coordinate value order.
_CONSTANT = 0
_DECREASING = -1
_INCREASING = 1


class _CoordAndDims(namedtuple("CoordAndDims", ["coord", "dims"])):
    """Container for a coordinate and the associated data dimension(s).

    Container for a coordinate and the associated data dimension(s)
    spanned over a :class:`iris.cube.Cube`.

    Parameters
    ----------
    coord : :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
    dims : tuple
        A tuple of the data dimension(s) spanned by the coordinate.

    """

    __slots__ = ()


class _CoordMetaData(
    namedtuple(
        "CoordMetaData",
        ["defn", "dims", "points_dtype", "bounds_dtype", "kwargs"],
    )
):
    """Container for the metadata that defines a dimension or auxiliary coordinate.

    Parameters
    ----------
    defn : :class:`iris.common.CoordMetadata`
        The :class:`iris.common.CoordMetadata` metadata that represents a
        coordinate.
    dims :
        The dimension(s) associated with the coordinate.
    points_dtype : :class:`np.dtype`
        The points data :class:`np.dtype` of an associated coordinate.
    bounds_dtype : :class:`np.dtype`
        The bounds data :class:`np.dtype` of an associated coordinate.
    **kwargs : dict, optional
        A dictionary of key/value pairs required to define a coordinate.

    """

    def __new__(mcs, coord, dims):
        """Create a new :class:`_CoordMetaData` instance.

        Parameters
        ----------
        coord : :class:`iris.coord.DimCoord` or :class:`iris.coord.AuxCoord`
        dims :
            The dimension(s) associated with the coordinate.

        Returns
        -------
        The new class instance.

        """
        defn = coord.metadata
        points_dtype = coord.core_points().dtype
        bounds_dtype = (
            coord.core_bounds().dtype if coord.core_bounds() is not None else None
        )
        kwargs = {}
        # Add scalar flag metadata.
        kwargs["scalar"] = coord.core_points().size == 1
        # Add circular flag metadata for dimensional coordinates.
        if hasattr(coord, "circular"):
            kwargs["circular"] = coord.circular
        if isinstance(coord, iris.coords.DimCoord):
            # Mix the monotonic ordering into the metadata.
            if coord.points[0] == coord.points[-1]:
                order = _CONSTANT
            elif coord.points[-1] > coord.points[0]:
                order = _INCREASING
            else:
                order = _DECREASING
            kwargs["order"] = order
        metadata = super().__new__(mcs, defn, dims, points_dtype, bounds_dtype, kwargs)
        return metadata

    __slots__ = ()

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, _CoordMetaData):
            sprops, oprops = self._asdict(), other._asdict()
            # Ignore "kwargs" meta-data for the first comparison.
            sprops["kwargs"] = oprops["kwargs"] = None
            result = sprops == oprops
            if result:
                skwargs, okwargs = self.kwargs.copy(), other.kwargs.copy()
                # Monotonic "order" only applies to DimCoord's.
                # The monotonic "order" must be _INCREASING or _DECREASING if
                # the DimCoord is NOT "scalar". Otherwise, if the DimCoord is
                # "scalar" then the "order" must be _CONSTANT.
                if skwargs["scalar"] or okwargs["scalar"]:
                    # We don't care about the monotonic "order" given that
                    # at least one coordinate is a scalar coordinate.
                    skwargs["scalar"] = okwargs["scalar"] = None
                    skwargs["order"] = okwargs["order"] = None
                result = skwargs == okwargs
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def name(self):
        """Get the name from the coordinate definition."""
        return self.defn.name()


class _DerivedCoordAndDims(
    namedtuple("DerivedCoordAndDims", ["coord", "dims", "aux_factory"])
):
    """Container for a derived coordinate and dimensions(s).

    Container for a derived coordinate, the associated AuxCoordFactory, and the
    associated data dimension(s) spanned over a :class:`iris.cube.Cube`.

    Parameters
    ----------
    coord : :class:`iris.coord.DimCoord` or :class:`iris.coord.AuxCoord`
    dims : tuple
        A tuple of the data dimension(s) spanned by the coordinate.
    aux_factory : :class:`iris.aux_factory.AuxCoordFactory`

    """

    __slots__ = ()

    def __eq__(self, other):
        """Do not take aux factories into account for equality."""
        result = NotImplemented
        if isinstance(other, _DerivedCoordAndDims):
            equal_coords = self.coord == other.coord
            equal_dims = self.dims == other.dims
            result = equal_coords and equal_dims
        return result


class _OtherMetaData(namedtuple("OtherMetaData", ["defn", "dims"])):
    """Container for the metadata that defines a cell measure or ancillary variable.

    Parameters
    ----------
    defn : :class:`iris.coords._DMDefn` or :class:`iris.coords._CellMeasureDefn`
        The :class:`iris.coords._DMDefn` or :class:`iris.coords._CellMeasureDefn`
        metadata that represents a coordinate.
    dims :
        The dimension(s) associated with the coordinate.

    """

    def __new__(cls, ancil, dims):
        """Create a new :class:`_OtherMetaData` instance.

        Parameters
        ----------
        ancil : :class:`iris.coord.CellMeasure` or :class:`iris.coord.AncillaryVariable`
        dims :
            The dimension(s) associated with ancil.

        Returns
        -------
        The new class instance.

        """
        defn = ancil.metadata
        metadata = super().__new__(cls, defn, dims)
        return metadata

    __slots__ = ()

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, _OtherMetaData):
            result = self._asdict() == other._asdict()
        return result

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def name(self):
        """Get the name from the coordinate definition."""
        return self.defn.name()


class _SkeletonCube(namedtuple("SkeletonCube", ["signature", "data"])):
    """Basis of a source-cube.

    Basis of a source-cube, containing the associated coordinate metadata,
    coordinates and cube data payload.

    Parameters
    ----------
    signature : :class:`_CoordSignature`
        The :class:`_CoordSignature` of an associated source-cube.
    data :
        The data payload of an associated :class:`iris.cube.Cube` source-cube.

    """

    __slots__ = ()


class _Extent(namedtuple("Extent", ["min", "max"])):
    """Container representing the limits of a one-dimensional extent/range.

    Parameters
    ----------
    min :
        The minimum value of the extent.
    max :
        The maximum value of the extent.

    """

    __slots__ = ()


class _CoordExtent(namedtuple("CoordExtent", ["points", "bounds"])):
    """Container representing the points and bounds extent of a one dimensional coordinate.

    Parameters
    ----------
    points : :class:`_Extent`
        The :class:`_Extent` of the coordinate point values.
    bounds :
        A list containing the :class:`_Extent` of the coordinate lower
        bound and the upper bound. Defaults to None if no associated
        bounds exist for the coordinate.

    """

    __slots__ = ()


def _hash_ndarray(a: np.ndarray) -> np.ndarray:
    """Compute a hash from a numpy array.

    Calculates a 64-bit non-cryptographic hash of the provided array, using
    the fast ``xxhash`` hashing algorithm.

    Parameters
    ----------
    a :
        The array to hash.

    Returns
    -------
    numpy.ndarray :
        An array of shape (1,) containing the hash value.

    """
    # Include the array dtype as it is not preserved by `ndarray.tobytes()`.
    hash = xxh3_64(f"dtype={a.dtype}".encode("utf-8"))

    # Hash the bytes representing the array data.
    hash.update(b"data=")
    if np.ma.is_masked(a):
        # Hash only the unmasked data
        hash.update(a.compressed().tobytes())
        # Hash the mask
        hash.update(b"mask=")
        hash.update(a.mask.tobytes())
    else:
        hash.update(a.tobytes())
    return np.frombuffer(hash.digest(), dtype=np.int64)


def _hash_chunk(
    x_chunk: np.ndarray,
    axis: tuple[int] | None,
    keepdims: bool,
) -> np.ndarray:
    """Compute a hash from a numpy array.

    This function can be applied to each chunk or intermediate chunk in
    :func:`~dask.array.reduction`. It preserves the number of input dimensions
    to facilitate combining intermediate results into intermediate chunks.

    Parameters
    ----------
    x_chunk :
        The array to hash.
    axis :
        Unused but required by :func:`~dask.array.reduction`.
    keepdims :
        Unused but required by :func:`~dask.array.reduction`.

    Returns
    -------
    numpy.ndarray :
        An array containing the hash value.

    """
    return _hash_ndarray(x_chunk).reshape((1,) * x_chunk.ndim)


def _hash_aggregate(
    x_chunk: np.ndarray,
    axis: tuple[int] | None,
    keepdims: bool,
) -> np.int64:
    """Compute a hash from a numpy array.

    This function can be applied as the final step in :func:`~dask.array.reduction`.

    Parameters
    ----------
    x_chunk :
        The array to hash.
    axis :
        Unused but required by :func:`~dask.array.reduction`.
    keepdims :
        Unused but required by :func:`~dask.array.reduction`.

    Returns
    -------
    np.int64 :
        The hash value.

    """
    (result,) = _hash_ndarray(x_chunk)
    return result


def _hash_array(a: da.Array | np.ndarray) -> np.int64:
    """Calculate a hash representation of the provided array.

    Calculates a 64-bit non-cryptographic hash of the provided array, using
    the fast ``xxhash`` hashing algorithm.

    Note that the computed hash depends on how the array is chunked.

    Parameters
    ----------
    a :
        The array that requires to have its hexdigest calculated.

    Returns
    -------
    np.int64
        The array's hash.

    """
    if isinstance(a, da.Array):
        # Use :func:`~dask.array.reduction` to compute a hash from a Dask array.
        #
        # A hash of each input chunk will be computed by the `chunk` function
        # and those hashes will be combined into one or more intermediate chunks.
        # If there are multiple intermediate chunks, a hash for each intermediate
        # chunk will be computed by the `combine` function and the
        # results will be combined into a new layer of intermediate chunks. This
        # will be repeated until only a single intermediate chunk remains.
        # Finally, a single hash value will be computed from the last
        # intermediate chunk by the `aggregate` function.
        result = da.reduction(
            a,
            chunk=_hash_chunk,
            combine=_hash_chunk,
            aggregate=_hash_aggregate,
            keepdims=False,
            meta=np.empty(tuple(), dtype=np.int64),
            dtype=np.int64,
        )
    else:
        result = _hash_aggregate(a, None, False)
    return result


class _ArrayHash(namedtuple("ArrayHash", ["value", "chunks"])):
    """Container for a hash value and the chunks used when computing it.

    Parameters
    ----------
    value : :class:`np.int64`
        The hash value.
    chunks : tuple
        The chunks the array had when the hash was computed.
    """

    __slots__ = ()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Unable to compare {repr(self)} to {repr(other)}")

        def shape(chunks):
            return tuple(sum(c) for c in chunks)

        if shape(self.chunks) == shape(other.chunks):
            if self.chunks != other.chunks:
                raise ValueError(
                    "Unable to compare arrays with different chunks: "
                    f"{self.chunks} != {other.chunks}"
                )
            result = self.value == other.value
        else:
            result = False
        return result


def _array_id(
    coord: DimCoord | AuxCoord | AncillaryVariable | CellMeasure,
    bound: bool,
) -> str:
    """Get a unique key for looking up arrays associated with coordinates."""
    return f"{id(coord)}{bound}"


def _compute_hashes(
    arrays: Mapping[str, np.ndarray | da.Array],
) -> dict[str, _ArrayHash]:
    """Compute hashes for the arrays that will be compared.

    Two arrays are considered equal if each unmasked element compares equal
    and the masks are equal. However, hashes depend on chunking and dtype.
    Therefore, arrays with the same shape are rechunked so they have the same
    chunks and arrays with numerical dtypes are cast up to the same dtype before
    computing the hashes.

    Parameters
    ----------
    arrays :
        A mapping with key-array pairs.

    Returns
    -------
    dict[str, _ArrayHash] :
        An dictionary of hashes.

    """
    hashes = {}

    def is_numerical(dtype):
        return np.issubdtype(dtype, np.bool_) or np.issubdtype(dtype, np.number)

    def group_key(item):
        array_id, a = item
        if is_numerical(a.dtype):
            dtype = "numerical"
        else:
            dtype = str(a.dtype)
        return a.shape, dtype

    sorted_arrays = sorted(arrays.items(), key=group_key)
    for _, group_iter in itertools.groupby(sorted_arrays, key=group_key):
        array_ids, group = zip(*group_iter)
        # Unify dtype for numerical arrays, as the hash depends on it
        if is_numerical(group[0].dtype):
            dtype = np.result_type(*group)
            same_dtype_arrays = tuple(a.astype(dtype) for a in group)
        else:
            same_dtype_arrays = group
        if any(isinstance(a, da.Array) for a in same_dtype_arrays):
            # Unify chunks as the hash depends on the chunks.
            indices = tuple(range(group[0].ndim))
            # Because all arrays in a group have the same shape, `indices`
            # are the same for all of them. Providing `indices` as a tuple
            # instead of letters is easier to do programmatically.
            argpairs = [(a, indices) for a in same_dtype_arrays]
            __, rechunked_arrays = da.core.unify_chunks(*itertools.chain(*argpairs))
        else:
            rechunked_arrays = same_dtype_arrays
        for array_id, rechunked in zip(array_ids, rechunked_arrays):
            if isinstance(rechunked, da.Array):
                chunks = rechunked.chunks
            else:
                chunks = tuple((i,) for i in rechunked.shape)
            hashes[array_id] = (_hash_array(rechunked), chunks)

    (hashes,) = dask.compute(hashes)
    return {k: _ArrayHash(*v) for k, v in hashes.items()}


def concatenate(
    cubes: Sequence[iris.cube.Cube],
    error_on_mismatch: bool = False,
    check_aux_coords: bool = True,
    check_cell_measures: bool = True,
    check_ancils: bool = True,
    check_derived_coords: bool = True,
) -> iris.cube.CubeList:
    """Concatenate the provided cubes over common existing dimensions.

    Parameters
    ----------
    cubes : iterable of :class:`iris.cube.Cube`
        An iterable containing one or more :class:`iris.cube.Cube` instances
        to be concatenated together.
    error_on_mismatch : bool, default=False
        If True, raise an informative
        :class:`~iris.exceptions.ContatenateError` if registration fails.
    check_aux_coords : bool, default=True
        Checks if the points and bounds of auxiliary coordinates of the cubes
        match. This check is not applied to auxiliary coordinates that span the
        dimension the concatenation is occurring along.  Defaults to True.
    check_cell_measures : bool, default=True
        Checks if the data of cell measures of the cubes match. This check is
        not applied to cell measures that span the dimension the concatenation
        is occurring along. Defaults to True.
    check_ancils : bool, default=True
        Checks if the data of ancillary variables of the cubes match. This
        check is not applied to ancillary variables that span the dimension the
        concatenation is occurring along. Defaults to True.
    check_derived_coords : bool, default=True
        Checks if the points and bounds of derived coordinates of the cubes
        match. This check is not applied to derived coordinates that span the
        dimension the concatenation is occurring along. Note that differences
        in scalar coordinates and dimensional coordinates used to derive the
        coordinate are still checked. Checks for auxiliary coordinates used to
        derive the coordinates can be ignored with `check_aux_coords`. Defaults
        to True.

    Returns
    -------
     :class:`iris.cube.CubeList`
        A :class:`iris.cube.CubeList` of concatenated :class:`iris.cube.Cube` instances.

    """
    cube_signatures = [_CubeSignature(cube) for cube in cubes]

    proto_cubes: list[_ProtoCube] = []
    # Initialise the nominated axis (dimension) of concatenation
    # which requires to be negotiated.
    axis = None

    # Compute hashes for parallel array comparison.
    arrays = {}

    def add_coords(cube_signature: _CubeSignature, coord_type: str) -> None:
        for coord_and_dims in getattr(cube_signature, coord_type):
            coord = coord_and_dims.coord
            array_id = _array_id(coord, bound=False)
            if isinstance(coord, (DimCoord, AuxCoord)):
                arrays[array_id] = coord.core_points()
                if coord.has_bounds():
                    bound_array_id = _array_id(coord, bound=True)
                    arrays[bound_array_id] = coord.core_bounds()
            else:
                arrays[array_id] = coord.core_data()

    for cube_signature in cube_signatures:
        if check_aux_coords:
            add_coords(cube_signature, "aux_coords_and_dims")
        if check_derived_coords:
            add_coords(cube_signature, "derived_coords_and_dims")
        if check_cell_measures:
            add_coords(cube_signature, "cell_measures_and_dims")
        if check_ancils:
            add_coords(cube_signature, "ancillary_variables_and_dims")

    hashes = _compute_hashes(arrays)

    # Register each cube with its appropriate proto-cube.
    for cube_signature in cube_signatures:
        registered = False

        # Register cube with an existing proto-cube.
        for proto_cube in proto_cubes:
            registered = proto_cube.register(
                cube_signature,
                hashes,
                axis,
                error_on_mismatch,
                check_aux_coords,
                check_cell_measures,
                check_ancils,
                check_derived_coords,
            )
            if registered:
                axis = proto_cube.axis
                break

        # Create a new proto-cube for an unregistered cube.
        if not registered:
            proto_cubes.append(_ProtoCube(cube_signature))

    # Construct a concatenated cube from each of the proto-cubes.
    concatenated_cubes = iris.cube.CubeList()

    # Emulate Python 2 behaviour.
    def _none_sort(proto_cube):
        return (proto_cube.name is not None, proto_cube.name)

    for proto_cube in sorted(proto_cubes, key=_none_sort):
        concatenated_cubes.append(proto_cube.concatenate())

    # Perform concatenation until we've reached an equilibrium.
    count = len(concatenated_cubes)
    if count != 1 and count != len(cubes):
        concatenated_cubes = concatenate(concatenated_cubes)

    return concatenated_cubes


class _CubeSignature:
    """Template for identifying a specific type of :class:`iris.cube.Cube`.

    Template for identifying a specific type of :class:`iris.cube.Cube` based
    on its metadata, coordinates and cell_measures.

    """

    def __init__(self, cube: iris.cube.Cube) -> None:
        """Represent the cube metadata and associated coordinate metadata.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            The :class:`iris.cube.Cube` source-cube.

        """
        self.aux_coords_and_dims = []
        self.aux_metadata = []
        self.dim_coords = cube.dim_coords
        self.dim_metadata = []
        self.ndim = cube.ndim
        self.scalar_coords = []
        self.cell_measures_and_dims = []
        self.cm_metadata = []
        self.ancillary_variables_and_dims = []
        self.av_metadata = []
        self.derived_coords_and_dims = []
        self.derived_metadata = []
        self.dim_mapping = []

        # Determine whether there are any anonymous cube dimensions.
        covered = set(cube.coord_dims(coord)[0] for coord in self.dim_coords)
        self.anonymous = covered != set(range(self.ndim))

        self.defn = cube.metadata
        self.data_type = cube.dtype
        self.src_cube = cube

        #
        # Collate the dimension coordinate metadata.
        #
        for dim_coord in self.dim_coords:
            dims = cube.coord_dims(dim_coord)
            self.dim_metadata.append(_CoordMetaData(dim_coord, dims))
            self.dim_mapping.append(dims[0])

        #
        # Collate the auxiliary coordinate metadata and scalar coordinates.
        #
        axes = dict(T=0, Z=1, Y=2, X=3)

        # Coordinate sort function - by guessed coordinate axis, then
        # by coordinate name, then by dimensions, in ascending order.
        def key_func(coord):
            return (
                axes.get(guess_coord_axis(coord), len(axes) + 1),
                coord.name(),
                cube.coord_dims(coord),
            )

        for aux_coord in sorted(cube.aux_coords, key=key_func):
            dims = cube.coord_dims(aux_coord)
            if dims:
                self.aux_metadata.append(_CoordMetaData(aux_coord, dims))
                self.aux_coords_and_dims.append(_CoordAndDims(aux_coord, tuple(dims)))
            else:
                self.scalar_coords.append(aux_coord)

        def meta_key_func(dm):
            return (dm.metadata, dm.cube_dims(cube))

        for cm in sorted(cube.cell_measures(), key=meta_key_func):
            dims = cube.cell_measure_dims(cm)
            self.cm_metadata.append(_OtherMetaData(cm, dims))
            self.cell_measures_and_dims.append(_CoordAndDims(cm, tuple(dims)))

        for av in sorted(cube.ancillary_variables(), key=meta_key_func):
            dims = cube.ancillary_variable_dims(av)
            self.av_metadata.append(_OtherMetaData(av, dims))
            self.ancillary_variables_and_dims.append(_CoordAndDims(av, tuple(dims)))

        def name_key_func(factory):
            return factory.name()

        for factory in sorted(cube.aux_factories, key=name_key_func):
            coord = factory.make_coord(cube.coord_dims)
            dims = factory.derived_dims(cube.coord_dims)
            self.derived_metadata.append(_CoordMetaData(coord, dims))
            self.derived_coords_and_dims.append(
                _DerivedCoordAndDims(coord, tuple(dims), factory)
            )

    def _coordinate_differences(self, other, attr, reason="metadata"):
        """Determine the names of the coordinates that differ.

        Determine the names of the coordinates that differ between `self` and
        `other` for a coordinate attribute on a _CubeSignature.

        Parameters
        ----------
        other : _CubeSignature
            The _CubeSignature to compare against.
        attr : str
            The _CubeSignature attribute within which differences exist
            between `self` and `other`.
        reason : str, default="metadata"
            The reason to give for mismatch (function is normally, but not
            always, testing metadata).

        Returns
        -------
        tuple
            Tuple of a descriptive error message and the names of attributes
            that differ between `self` and `other`.

        """
        # Set up {name: attribute} dictionaries.
        self_dict = {x.name(): x for x in getattr(self, attr)}
        other_dict = {x.name(): x for x in getattr(other, attr)}
        if len(self_dict) == 0:
            self_dict = {"< None >": None}
        if len(other_dict) == 0:
            other_dict = {"< None >": None}
        self_names = sorted(self_dict.keys())
        other_names = sorted(other_dict.keys())

        # Compare coord attributes.
        if len(self_names) != len(other_names) or self_names != other_names:
            result = ("", ", ".join(self_names), ", ".join(other_names))
        else:
            diff_names = []
            for self_key, self_value in self_dict.items():
                other_value = other_dict[self_key]
                if self_value != other_value:
                    diff_names.append(self_key)
            result = (
                " " + reason,
                ", ".join(diff_names),
                ", ".join(diff_names),
            )
        return result

    def match(self, other, error_on_mismatch):
        """Return whether this _CubeSignature equals another.

        This is the first step to determine if two "cubes" (either a
        real Cube or a ProtoCube) can be concatenated, by considering:

        * data dimensions
        * aux coords metadata
        * scalar coords
        * attributes
        * dtype

        Parameters
        ----------
        other : _CubeSignature
            The _CubeSignature to compare against.
        error_on_mismatch : bool
            If True, raise a :class:`~iris.exceptions.MergeException`
            with a detailed explanation if the two do not match.

        Returns
        -------
        bool
            True if and only if this _CubeSignature matches the other.

        """
        msg_template = "{}{} differ: {} != {}"
        msgs = []

        # Check cube definitions.
        if self.defn != other.defn:
            # Note that the case of different phenomenon names is dealt
            # with in :meth:`iris.cube.CubeList.concatenate_cube()`.
            msg = "Cube metadata differs for phenomenon: {}"
            msgs.append(msg.format(self.defn.name()))

        # Check dim coordinates.
        if self.dim_metadata != other.dim_metadata:
            differences = self._coordinate_differences(other, "dim_metadata")
            msgs.append(msg_template.format("Dimension coordinates", *differences))
        # Check aux coordinates.
        if self.aux_metadata != other.aux_metadata:
            differences = self._coordinate_differences(other, "aux_metadata")
            msgs.append(msg_template.format("Auxiliary coordinates", *differences))
        # Check cell measures.
        if self.cm_metadata != other.cm_metadata:
            differences = self._coordinate_differences(other, "cm_metadata")
            msgs.append(msg_template.format("Cell measures", *differences))
        # Check ancillary variables.
        if self.av_metadata != other.av_metadata:
            differences = self._coordinate_differences(other, "av_metadata")
            msgs.append(msg_template.format("Ancillary variables", *differences))
        # Check derived coordinates.
        if self.derived_metadata != other.derived_metadata:
            differences = self._coordinate_differences(other, "derived_metadata")
            msgs.append(msg_template.format("Derived coordinates", *differences))
        # Check scalar coordinates.
        if self.scalar_coords != other.scalar_coords:
            differences = self._coordinate_differences(
                other, "scalar_coords", reason="values or metadata"
            )
            msgs.append(msg_template.format("Scalar coordinates", *differences))
        # Check ndim.
        if self.ndim != other.ndim:
            msgs.append(
                msg_template.format("Data dimensions", "", self.ndim, other.ndim)
            )
        # Check data type.
        if self.data_type != other.data_type:
            msgs.append(
                msg_template.format("Data types", "", self.data_type, other.data_type)
            )

        match = not bool(msgs)
        if error_on_mismatch and not match:
            raise iris.exceptions.ConcatenateError(msgs)
        return match


class _CoordSignature:
    """Template for identifying a specific type of :class:`iris.cube.Cube` based on its coordinates."""

    def __init__(self, cube_signature: _CubeSignature) -> None:
        """Represent the coordinate metadata.

        Represent the coordinate metadata required to identify suitable
        non-overlapping :class:`iris.cube.Cube` source-cubes for
        concatenation over a common single dimension.

        Parameters
        ----------
        cube_signature : :class:`_CubeSignature`
            The :class:`_CubeSignature` that defines the source-cube.

        """
        self.aux_coords_and_dims = cube_signature.aux_coords_and_dims
        self.cell_measures_and_dims = cube_signature.cell_measures_and_dims
        self.ancillary_variables_and_dims = cube_signature.ancillary_variables_and_dims
        self.derived_coords_and_dims = cube_signature.derived_coords_and_dims
        self.dim_coords = cube_signature.dim_coords
        self.dim_mapping = cube_signature.dim_mapping
        self.dim_extents: list[_CoordExtent] = []
        self.dim_order = [
            metadata.kwargs["order"] for metadata in cube_signature.dim_metadata
        ]

        # Calculate the extents for each dimensional coordinate.
        self._calculate_extents()

    @staticmethod
    def _cmp(
        coord: iris.coords.DimCoord,
        other: iris.coords.DimCoord,
    ) -> tuple[bool, bool]:
        """Compare the coordinates for concatenation compatibility.

        Returns
        -------
        bool tuple
            A boolean tuple pair of whether the coordinates are compatible,
            and whether they represent a candidate axis of concatenation.

        """
        # A candidate axis must have non-identical coordinate points.
        candidate_axis = not array_equal(coord.points, other.points)

        # Ensure both have equal availability of bounds.
        result = coord.has_bounds() == other.has_bounds()
        if result and not candidate_axis:
            # Ensure equality of bounds.
            result = array_equal(coord.bounds, other.bounds)

        return result, candidate_axis

    def candidate_axis(self, other: "_CoordSignature") -> int | None:
        """Determine the candidate axis of concatenation with the given coordinate signature.

        If a candidate axis is found, then the coordinate
        signatures are compatible.

        Parameters
        ----------
        other : :class:`_CoordSignature`

        Returns
        -------
        result :
            None if no single candidate axis exists, otherwise the candidate
            axis of concatenation.

        """
        result = False
        candidate_axes = []

        # Compare dimension coordinates.
        for ind, coord in enumerate(self.dim_coords):
            result, candidate_axis = self._cmp(coord, other.dim_coords[ind])
            if not result:
                break
            if candidate_axis:
                dim = self.dim_mapping[ind]
                candidate_axes.append(dim)

        # Only permit one degree of dimensional freedom when
        # determining the candidate axis of concatenation.
        if result and len(candidate_axes) == 1:
            axis = candidate_axes[0]
        else:
            axis = None

        return axis

    def _calculate_extents(self) -> None:
        """Calculate the extent over each dimension coordinates points and bounds."""
        self.dim_extents = []
        for coord, order in zip(self.dim_coords, self.dim_order):
            if order == _CONSTANT or order == _INCREASING:
                points = _Extent(coord.points[0], coord.points[-1])
                if coord.bounds is not None:
                    bounds = (
                        _Extent(coord.bounds[0, 0], coord.bounds[-1, 0]),
                        _Extent(coord.bounds[0, 1], coord.bounds[-1, 1]),
                    )
                else:
                    bounds = None
            else:
                # The order must be decreasing ...
                points = _Extent(coord.points[-1], coord.points[0])
                if coord.bounds is not None:
                    bounds = (
                        _Extent(coord.bounds[-1, 0], coord.bounds[0, 0]),
                        _Extent(coord.bounds[-1, 1], coord.bounds[0, 1]),
                    )
                else:
                    bounds = None

            self.dim_extents.append(_CoordExtent(points, bounds))


class _ProtoCube:
    """Framework for concatenating multiple source-cubes over one common dimension."""

    def __init__(self, cube_signature):
        """Create a new _ProtoCube from the given cube and record the cube as a source-cube.

        Parameters
        ----------
        cube_signature :
            Source :class:`_CubeSignature` of the :class:`_ProtoCube`.

        """
        # Cache the source-cube of this proto-cube.
        self._cube = cube_signature.src_cube

        # The cube signature is a combination of cube and coordinate
        # metadata that defines this proto-cube.
        self._cube_signature = cube_signature

        # The coordinate signature allows suitable non-overlapping
        # source-cubes to be identified.
        self._coord_signature = _CoordSignature(self._cube_signature)

        # The list of source-cubes relevant to this proto-cube.
        self._skeletons = []
        self._add_skeleton(self._coord_signature, self._cube.lazy_data())

        # The nominated axis of concatenation.
        self._axis = None

    @property
    def axis(self):
        """Return the nominated dimension of concatenation."""
        return self._axis

    @property
    def name(self) -> str | None:
        """Return the standard_name or long name."""
        metadata = self._cube_signature.defn
        return metadata.standard_name or metadata.long_name

    def concatenate(self):
        """Concatenate all the source-cubes registered with the :class:`_ProtoCube`.

        Concatenate all the source-cubes registered with the
        :class:`_ProtoCube` over the nominated common dimension.

        Returns
        -------
        :class:`iris.cube.Cube`
            The concatenated :class:`iris.cube.Cube`.

        """
        if len(self._skeletons) > 1:
            skeletons = self._skeletons
            dim_ind = self._coord_signature.dim_mapping.index(self.axis)
            order = self._coord_signature.dim_order[dim_ind]
            cube_signature = self._cube_signature

            # Sequence the skeleton segments into the correct order
            # pending concatenation.
            skeletons.sort(
                key=lambda skeleton: skeleton.signature.dim_extents,
                reverse=(order == _DECREASING),
            )

            # Concatenate the new dimension coordinate.
            dim_coords_and_dims = self._build_dim_coordinates()

            # Concatenate the new auxiliary coordinates (does NOT include
            # scalar coordinates!).
            aux_coords_and_dims = self._build_aux_coordinates()

            # Concatenate the new scalar coordinates.
            scalar_coords = self._build_scalar_coordinates()

            # Concatenate the new cell measures
            cell_measures_and_dims = self._build_cell_measures()

            # Concatenate the new ancillary variables
            ancillary_variables_and_dims = self._build_ancillary_variables()

            # Concatenate the new aux factories
            aux_factories = self._build_aux_factories(
                dim_coords_and_dims, aux_coords_and_dims, scalar_coords
            )

            # Concatenate the new data payload.
            data = self._build_data()

            # Build the new cube.
            all_aux_coords_and_dims = aux_coords_and_dims + [
                (scalar_coord, ()) for scalar_coord in scalar_coords
            ]
            kwargs = cube_signature.defn._asdict()
            cube = iris.cube.Cube(
                data,
                dim_coords_and_dims=dim_coords_and_dims,
                aux_coords_and_dims=all_aux_coords_and_dims,
                cell_measures_and_dims=cell_measures_and_dims,
                ancillary_variables_and_dims=ancillary_variables_and_dims,
                aux_factories=aux_factories,
                **kwargs,
            )
        else:
            # There are no other source-cubes to concatenate
            # with this proto-cube.
            cube = self._cube

        return cube

    def register(
        self,
        cube_signature: _CubeSignature,
        hashes: Mapping[str, _ArrayHash],
        axis: int | None = None,
        error_on_mismatch: bool = False,
        check_aux_coords: bool = False,
        check_cell_measures: bool = False,
        check_ancils: bool = False,
        check_derived_coords: bool = False,
    ) -> bool:
        """Determine if  the given source-cube is suitable for concatenation.

        Determine if  the given source-cube is suitable for concatenation
        with this :class:`_ProtoCube`.

        Parameters
        ----------
        cube_signature : :class:`_CubeSignature`
            The :class:`_CubeSignature` of the source-cube candidate for
            concatenation.
        hashes :
            A mapping containing hash values for checking coordinate, ancillary
            variable, and cell measure equality.
        axis : optional
            Seed the dimension of concatenation for the :class:`_ProtoCube`
            rather than rely on negotiation with source-cubes.
        error_on_mismatch : bool, default=False
            If True, raise an informative error if registration fails.
        check_aux_coords : bool, default=False
            Checks if the points and bounds of auxiliary coordinates of the
            cubes match. This check is not applied to auxiliary coordinates
            that span the dimension the concatenation is occurring along.
            Defaults to False.
        check_cell_measures : bool, default=False
            Checks if the data of cell measures of the cubes match. This check
            is not applied to cell measures that span the dimension the
            concatenation is occurring along. Defaults to False.
        check_ancils : bool, default=False
            Checks if the data of ancillary variables of the cubes match. This
            check is not applied to ancillary variables that span the dimension
            the concatenation is occurring along. Defaults to False.
        check_derived_coords : bool, default=False
            Checks if the points and bounds of derived coordinates of the cubes
            match. This check is not applied to derived coordinates that span
            the dimension the concatenation is occurring along. Note that
            differences in scalar coordinates and dimensional coordinates used
            to derive the coordinate are still checked. Checks for auxiliary
            coordinates used to derive the coordinates can be ignored with
            `check_aux_coords`. Defaults to False.

        Returns
        -------
        bool

        """
        # Verify and assert the nominated axis.
        if axis is not None and self.axis is not None and self.axis != axis:
            msg = "Nominated axis [{}] is not equal to negotiated axis [{}]".format(
                axis, self.axis
            )
            raise ValueError(msg)

        # Check for compatible cube signatures.
        match = self._cube_signature.match(cube_signature, error_on_mismatch)
        mismatch_error_msg = None

        # Check for compatible coordinate signatures.
        if match:
            coord_signature = _CoordSignature(cube_signature)
            candidate_axis = self._coord_signature.candidate_axis(coord_signature)
            match = candidate_axis is not None and (
                candidate_axis == axis or axis is None
            )
            if not match:
                mismatch_error_msg = (
                    f"Cannot find an axis to concatenate over for phenomenon "
                    f"`{self._cube.name()}`"
                )

        # Check for compatible coordinate extents.
        if match:
            dim_ind = self._coord_signature.dim_mapping.index(candidate_axis)
            match = self._sequence(coord_signature.dim_extents[dim_ind], candidate_axis)
            if error_on_mismatch and not match:
                mismatch_error_msg = f"Found cubes with overlap on concatenate axis {candidate_axis}, cannot concatenate overlapping cubes"
            elif not match:
                mismatch_error_msg = f"Found cubes with overlap on concatenate axis {candidate_axis}, skipping concatenation for these cubes"

        def get_hashes(
            coord: DimCoord | AuxCoord | AncillaryVariable | CellMeasure,
        ) -> tuple[_ArrayHash, ...]:
            array_id = _array_id(coord, bound=False)
            result = [hashes[array_id]]
            if isinstance(coord, (DimCoord, AuxCoord)) and coord.has_bounds():
                bound_array_id = _array_id(coord, bound=True)
                result.append(hashes[bound_array_id])
            return tuple(result)

        # Mapping from `_CubeSignature` attributes to human readable names.
        coord_type_names = {
            "aux_coords_and_dims": "Auxiliary coordinates",
            "cell_measures_and_dims": "Cell measures",
            "ancillary_variables_and_dims": "Ancillary variables",
            "derived_coords_and_dims": "Derived coordinates",
        }

        def check_coord_match(coord_type: str) -> tuple[bool, str]:
            result = (True, "")
            for coord_a, coord_b in zip(
                getattr(self._cube_signature, coord_type),
                getattr(cube_signature, coord_type),
            ):
                # Coordinates that span the candidate axis can differ
                if (
                    candidate_axis not in coord_a.dims
                    or candidate_axis not in coord_b.dims
                ):
                    if not get_hashes(coord_a.coord) == get_hashes(coord_b.coord):
                        mismatch_error_msg = (
                            f"{coord_type_names[coord_type]} are unequal for phenomenon"
                            f" `{self._cube.name()}`:\n"
                            f"a: {coord_a}\n"
                            f"b: {coord_b}"
                        )
                        result = (False, mismatch_error_msg)
                        break

            return result

        # Check for compatible AuxCoords.
        if match and check_aux_coords:
            match, msg = check_coord_match("aux_coords_and_dims")
            if not match:
                mismatch_error_msg = msg

        # Check for compatible CellMeasures.
        if match and check_cell_measures:
            match, msg = check_coord_match("cell_measures_and_dims")
            if not match:
                mismatch_error_msg = msg

        # Check for compatible AncillaryVariables.
        if match and check_ancils:
            match, msg = check_coord_match("ancillary_variables_and_dims")
            if not match:
                mismatch_error_msg = msg

        # Check for compatible derived coordinates.
        if match and check_derived_coords:
            match, msg = check_coord_match("derived_coords_and_dims")
            if not match:
                mismatch_error_msg = msg

        if match:
            # Register the cube as a source-cube for this proto-cube.
            self._add_skeleton(coord_signature, cube_signature.src_cube.lazy_data())
            # Declare the nominated axis of concatenation.
            self._axis = candidate_axis
            # If the protocube dimension order is constant (indicating it was
            # created from a cube with a length 1 dimension coordinate) but
            # a subsequently registered cube has a non-constant dimension
            # order we should use that instead of _CONSTANT to make sure all
            # the ordering checks and sorts work as expected.
            dim_ind = self._coord_signature.dim_mapping.index(candidate_axis)
            existing_order = self._coord_signature.dim_order[dim_ind]
            this_order = coord_signature.dim_order[dim_ind]
            if existing_order == _CONSTANT and this_order != _CONSTANT:
                self._coord_signature.dim_order[dim_ind] = this_order

        if mismatch_error_msg and not match:
            if error_on_mismatch:
                raise iris.exceptions.ConcatenateError([mismatch_error_msg])
            else:
                warnings.warn(
                    mismatch_error_msg, category=iris.warnings.IrisUserWarning
                )

        return match

    def _add_skeleton(self, coord_signature, data):
        """Create and add the source-cube skeleton to the :class:`_ProtoCube`.

        Parameters
        ----------
        coord_signature : :`_CoordSignature`
            The :class:`_CoordSignature` of the associated
            given source-cube.

        data : :class:`iris.cube.Cube`
            The data payload of an associated :class:`iris.cube.Cube`
            source-cube.

        """
        skeleton = _SkeletonCube(coord_signature, data)
        self._skeletons.append(skeleton)

    def _build_aux_coordinates(self):
        """Generate the auxiliary coordinates with associated dimension(s) mapping.

        Generate the auxiliary coordinates with associated dimension(s)
        mapping for the new concatenated cube.

        Returns
        -------
        A list of auxiliary coordinates and dimension(s) tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        cube_signature = self._cube_signature

        aux_coords_and_dims = []

        # Generate all the auxiliary coordinates for the new concatenated cube.
        for i, (coord, dims) in enumerate(cube_signature.aux_coords_and_dims):
            # Check whether the coordinate spans the nominated
            # dimension of concatenation.
            if self.axis in dims:
                # Concatenate the points together.
                dim = dims.index(self.axis)
                points = [
                    skton.signature.aux_coords_and_dims[i].coord.core_points()
                    for skton in skeletons
                ]
                points = np.concatenate(tuple(points), axis=dim)

                # Concatenate the bounds together.
                bnds = None
                if coord.has_bounds():
                    bnds = [
                        skton.signature.aux_coords_and_dims[i].coord.core_bounds()
                        for skton in skeletons
                    ]
                    bnds = np.concatenate(tuple(bnds), axis=dim)

                # Generate the associated coordinate metadata.
                kwargs = cube_signature.aux_metadata[i].defn._asdict()

                # Build the concatenated coordinate.
                if isinstance(coord, iris.coords.AuxCoord):
                    coord = iris.coords.AuxCoord(points, bounds=bnds, **kwargs)
                else:
                    # Attempt to create a DimCoord, otherwise default to
                    # an AuxCoord on failure.
                    try:
                        coord = iris.coords.DimCoord(points, bounds=bnds, **kwargs)
                    except ValueError:
                        # Ensure to remove the "circular" kwarg, which may be
                        # present in the defn of a DimCoord being demoted.
                        _ = kwargs.pop("circular", None)
                        coord = iris.coords.AuxCoord(points, bounds=bnds, **kwargs)

            aux_coords_and_dims.append((coord.copy(), dims))

        return aux_coords_and_dims

    def _build_scalar_coordinates(self):
        """Generate the scalar coordinates for the new concatenated cube.

        Returns
        -------
        A list of scalar coordinates.

        """
        scalar_coords = []
        for coord in self._cube_signature.scalar_coords:
            scalar_coords.append(coord.copy())

        return scalar_coords

    def _build_cell_measures(self):
        """Generate the cell measures with associated dimension(s) mapping.

        Generate the cell measures with associated dimension(s)
        mapping for the new concatenated cube.

        Returns
        -------
        A list of cell measures and dimension(s) tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        cube_signature = self._cube_signature

        cell_measures_and_dims = []

        # Generate all the cell measures for the new concatenated cube.
        for i, (cm, dims) in enumerate(cube_signature.cell_measures_and_dims):
            # Check whether the cell measure spans the nominated
            # dimension of concatenation.
            if self.axis in dims:
                # Concatenate the data together.
                dim = dims.index(self.axis)
                data = [
                    skton.signature.cell_measures_and_dims[i].coord.core_data()
                    for skton in skeletons
                ]
                data = concatenate_arrays(tuple(data), axis=dim)

                # Generate the associated metadata.
                kwargs = cube_signature.cm_metadata[i].defn._asdict()

                # Build the concatenated coordinate.
                cm = iris.coords.CellMeasure(data, **kwargs)

            cell_measures_and_dims.append((cm.copy(), dims))

        return cell_measures_and_dims

    def _build_ancillary_variables(self):
        """Generate the ancillary variables with associated dimension(s) mapping.

        Generate the ancillary variables with associated dimension(s)
        mapping for the new concatenated cube.

        Returns
        -------
        A list of ancillary variables and dimension(s) tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        cube_signature = self._cube_signature

        ancillary_variables_and_dims = []

        # Generate all the ancillary variables for the new concatenated cube.
        for i, (av, dims) in enumerate(cube_signature.ancillary_variables_and_dims):
            # Check whether the ancillary variable spans the nominated
            # dimension of concatenation.
            if self.axis in dims:
                # Concatenate the data together.
                dim = dims.index(self.axis)
                data = [
                    skton.signature.ancillary_variables_and_dims[i].coord.core_data()
                    for skton in skeletons
                ]
                data = concatenate_arrays(tuple(data), axis=dim)

                # Generate the associated metadata.
                kwargs = cube_signature.av_metadata[i].defn._asdict()

                # Build the concatenated coordinate.
                av = iris.coords.AncillaryVariable(data, **kwargs)

            ancillary_variables_and_dims.append((av.copy(), dims))

        return ancillary_variables_and_dims

    def _build_aux_factories(
        self, dim_coords_and_dims, aux_coords_and_dims, scalar_coords
    ):
        """Generate the aux factories for the new concatenated cube.

        Parameters
        ----------
        dim_coords_and_dims :
            A list of dimension coordinate and dimension tuple pairs from the
            concatenated cube.
        aux_coords_and_dims :
            A list of auxiliary coordinates and dimension(s) tuple pairs from
            the concatenated cube.
         scalar_coords :
            A list of scalar coordinates from the concatenated cube.

        Returns
        -------
        list of :class:`iris.aux_factory.AuxCoordFactory`

        """
        # Setup convenience hooks.
        cube_signature = self._cube_signature
        old_dim_coords = cube_signature.dim_coords
        old_aux_coords = [a[0] for a in cube_signature.aux_coords_and_dims]
        new_dim_coords = [d[0] for d in dim_coords_and_dims]
        new_aux_coords = [a[0] for a in aux_coords_and_dims]
        old_scalar_coords = cube_signature.scalar_coords
        new_scalar_coords = scalar_coords

        aux_factories = []

        # Generate all the factories for the new concatenated cube.
        for _, _, factory in cube_signature.derived_coords_and_dims:
            # Update the dependencies of the factory with coordinates of
            # the concatenated cube. We need to check all coordinate types
            # here (dim coords, aux coords, and scalar coords).

            # Note: in contrast to other _build_... methods of this class, we
            # do NOT need to distinguish between aux factories that span the
            # nominated concatenation axis and aux factories that do not. The
            # reason is that ALL aux factories need to be updated with the new
            # coordinates of the concatenated cube (passed to this function via
            # dim_coords_and_dims, aux_coords_and_dims, scalar_coords [these
            # contain ALL new coordinates, not only the ones spanning the
            # concatenation dimension]), so no special treatment for the aux
            # factories that span the concatenation dimension is necessary. If
            # not all aux factories are properly updated with references to the
            # new coordinates, this may lead to KeyErrors (see
            # https://github.com/SciTools/iris/issues/5339).
            new_dependencies = {}
            for old_dependency in factory.dependencies.values():
                if old_dependency in old_dim_coords:
                    dep_idx = old_dim_coords.index(old_dependency)
                    new_dependency = new_dim_coords[dep_idx]
                elif old_dependency in old_aux_coords:
                    dep_idx = old_aux_coords.index(old_dependency)
                    new_dependency = new_aux_coords[dep_idx]
                else:
                    dep_idx = old_scalar_coords.index(old_dependency)
                    new_dependency = new_scalar_coords[dep_idx]
                new_dependencies[id(old_dependency)] = new_dependency

            # Create new factory with the updated dependencies.
            factory = factory.updated(new_dependencies)

            aux_factories.append(factory)

        return aux_factories

    def _build_data(self):
        """Generate the data payload for the new concatenated cube.

        Returns
        -------
        The concatenated :class:`iris.cube.Cube` data payload.

        """
        skeletons = self._skeletons
        data = [skeleton.data for skeleton in skeletons]

        data = concatenate_arrays(data, self.axis)

        return data

    def _build_dim_coordinates(self):
        """Generate the dimension coordinates.

        Generate the dimension coordinates with associated dimension
        mapping for the new concatenated cube.

        Returns
        -------
        A list of dimension coordinate and dimension tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        axis = self.axis
        dim_ind = self._cube_signature.dim_mapping.index(axis)
        metadata = self._cube_signature.dim_metadata[dim_ind]
        defn, circular = metadata.defn, metadata.kwargs["circular"]

        # Concatenate the points together for the nominated dimension.
        points = [
            skeleton.signature.dim_coords[dim_ind].core_points()
            for skeleton in skeletons
        ]
        points = np.concatenate(tuple(points))

        # Concatenate the bounds together for the nominated dimension.
        bounds = None
        if self._cube_signature.dim_coords[dim_ind].has_bounds():
            bounds = [
                skeleton.signature.dim_coords[dim_ind].core_bounds()
                for skeleton in skeletons
            ]
            bounds = np.concatenate(tuple(bounds))

        # Populate the new dimension coordinate with the concatenated
        # points, bounds and associated metadata.
        kwargs = defn._asdict()
        kwargs["circular"] = circular
        dim_coord = iris.coords.DimCoord(points, bounds=bounds, **kwargs)

        # Generate all the dimension coordinates for the new concatenated cube.
        dim_coords_and_dims = []
        for ind, coord in enumerate(self._cube_signature.dim_coords):
            dim = self._cube_signature.dim_mapping[ind]
            if dim == axis:
                dim_coords_and_dims.append((dim_coord, dim))
            else:
                dim_coords_and_dims.append((coord.copy(), dim))

        return dim_coords_and_dims

    def _sequence(self, extent, axis):
        """Determine whether the extent can be sequenced.

        Determine whether the given extent can be sequenced along with
        all the extents of the source-cubes already registered with
        this :class:`_ProtoCube` into non-overlapping segments for the
        given axis.

        Parameters
        ----------
        extent : :class:`_CoordExtent`
            The :class:`_CoordExtent` of the candidate source-cube.
        axis :
            The candidate axis of concatenation.

        Returns
        -------
        bool

        """
        result = True

        # Add the new extent to the current extents collection.
        dim_ind = self._coord_signature.dim_mapping.index(axis)
        dim_extents = [
            skeleton.signature.dim_extents[dim_ind] for skeleton in self._skeletons
        ]
        dim_extents.append(extent)

        # Sort into the appropriate dimension order.
        order = self._coord_signature.dim_order[dim_ind]
        dim_extents.sort(reverse=(order == _DECREASING))

        # Ensure that the extents don't overlap.
        if len(dim_extents) > 1:
            for i, extent in enumerate(dim_extents[1:]):
                # Check the points - must be strictly monotonic.
                if order == _DECREASING:
                    big = dim_extents[i].points.min
                    small = extent.points.max
                else:
                    small = dim_extents[i].points.max
                    big = extent.points.min

                if small >= big:
                    result = False
                    break

                # Check the bounds - must be strictly monotonic.
                if extent.bounds is not None:
                    if order == _DECREASING:
                        big_0 = dim_extents[i].bounds[0].min
                        big_1 = dim_extents[i].bounds[1].min
                        small_0 = extent.bounds[0].max
                        small_1 = extent.bounds[1].max
                    else:
                        small_0 = dim_extents[i].bounds[0].max
                        small_1 = dim_extents[i].bounds[1].max
                        big_0 = extent.bounds[0].min
                        big_1 = extent.bounds[1].min

                    lower_bound_fail = small_0 >= big_0
                    upper_bound_fail = small_1 >= big_1

                    if lower_bound_fail or upper_bound_fail:
                        result = False
                        break

        return result

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Classes for representing multi-dimensional data with metadata."""

from __future__ import annotations

from collections.abc import (
    Container,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
import copy
from copy import deepcopy
from functools import partial, reduce
import itertools
import operator
from typing import TYPE_CHECKING, Any, Optional, TypeGuard
import warnings
from xml.dom.minidom import Document
import zlib

from cf_units import Unit
import dask.array as da
import numpy as np
import numpy.ma as ma

import iris._constraints
from iris._data_manager import DataManager
import iris._lazy_data as _lazy
import iris._merge
import iris.analysis
from iris.analysis import _Weights
from iris.analysis.cartography import wrap_lons
import iris.analysis.maths
import iris.aux_factory
from iris.aux_factory import AuxCoordFactory
from iris.common import CFVariableMixin, CubeMetadata, metadata_manager_factory
from iris.common.metadata import CoordMetadata, metadata_filter
from iris.common.mixin import LimitedAttributeDict
import iris.coord_systems
import iris.coords
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, CellMethod, DimCoord

if TYPE_CHECKING:
    import iris.mesh
    from iris.mesh import MeshCoord
import iris.exceptions
import iris.util
import iris.warnings

__all__ = ["Cube", "CubeAttrsDict", "CubeList"]


# The XML namespace to use for CubeML documents
XML_NAMESPACE_URI = "urn:x-iris:cubeml-0.2"


class _CubeFilter:
    """A constraint, paired with a list of cubes matching that constraint."""

    def __init__(self, constraint, cubes=None):
        self.constraint = constraint
        if cubes is None:
            cubes = CubeList()
        self.cubes = cubes

    def __len__(self):
        return len(self.cubes)

    def add(self, cube):
        """Add the appropriate (sub)cube to the list of cubes where it matches the constraint."""
        sub_cube = self.constraint.extract(cube)
        if sub_cube is not None:
            self.cubes.append(sub_cube)

    def combined(self, unique=False):
        """Return a new :class:`_CubeFilter` by combining the list of cubes.

        Combines the list of cubes with :func:`~iris._combine_load_cubes`.

        Parameters
        ----------
        unique : bool, default=False
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        from iris import _combine_load_cubes

        return _CubeFilter(
            self.constraint,
            _combine_load_cubes(self.cubes, merge_require_unique=unique),
        )


class _CubeFilterCollection:
    """A list of _CubeFilter instances."""

    @staticmethod
    def from_cubes(cubes, constraints=None):
        """Create a new collection from an iterable of cubes, and some optional constraints."""
        constraints = iris._constraints.list_of_constraints(constraints)
        pairs = [_CubeFilter(constraint) for constraint in constraints]
        collection = _CubeFilterCollection(pairs)
        for c in cubes:
            collection.add_cube(c)
        return collection

    def __init__(self, pairs):
        self.pairs = pairs

    def add_cube(self, cube):
        """Add the given :class:`~iris.cube.Cube` to all of the relevant constraint pairs."""
        for pair in self.pairs:
            pair.add(cube)

    def cubes(self):
        """Return all the cubes in this collection in a single :class:`CubeList`."""
        from iris.cube import CubeList

        result = CubeList()
        for pair in self.pairs:
            result.extend(pair.cubes)
        return result

    def combined(self, unique=False):
        """Return a new :class:`_CubeFilterCollection` by combining all the cube lists of this collection.

        Combines each list of cubes using :func:`~iris._combine_load_cubes`.

        Parameters
        ----------
        unique : bool, default=False
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        return _CubeFilterCollection([pair.combined(unique) for pair in self.pairs])


class CubeList(list):
    """All the functionality of a standard :class:`list` with added "Cube" context."""

    def __init__(self, *args, **kwargs):
        """Given an iterable of cubes, return a CubeList instance."""
        # Do whatever a list does, to initialise ourself "as a list"
        super().__init__(*args, **kwargs)
        # Check that all items in the list are cubes.
        for c in self:
            self._assert_is_cube(c)

    def __str__(self):
        """Run short :meth:`Cube.summary` on every cube."""
        result = [
            "%s: %s" % (i, cube.summary(shorten=True)) for i, cube in enumerate(self)
        ]
        if result:
            result = "\n".join(result)
        else:
            result = "< No cubes >"
        return result

    def __repr__(self):
        """Run repr on every cube."""
        return "[%s]" % ",\n".join([repr(cube) for cube in self])

    @staticmethod
    def _assert_is_cube(obj):
        if not hasattr(obj, "add_aux_coord"):
            msg = r"Object {obj} cannot be put in a cubelist, as it is not a Cube."
            raise ValueError(msg)

    def _repr_html_(self):
        from iris.experimental.representation import CubeListRepresentation

        representer = CubeListRepresentation(self)
        return representer.repr_html()

    # TODO #370 Which operators need overloads?

    def __add__(self, other):
        return CubeList(list.__add__(self, other))

    def __getitem__(self, keys):  # numpydoc ignore=SS02
        """x.__getitem__(y) <==> x[y]."""
        result = super().__getitem__(keys)
        if isinstance(result, list):
            result = CubeList(result)
        return result

    def __getslice__(self, start, stop):  # numpydoc ignore=SS02
        """x.__getslice__(i, j) <==> x[i:j].

        Use of negative indices is not supported.

        """
        result = super().__getslice__(start, stop)
        result = CubeList(result)
        return result

    def __iadd__(self, other_cubes):
        """Add a sequence of cubes to the cubelist in place."""
        return super(CubeList, self).__iadd__(CubeList(other_cubes))

    def __setitem__(self, key, cube_or_sequence):
        """Set self[key] to cube or sequence of cubes."""
        if isinstance(key, int):
            # should have single cube.
            self._assert_is_cube(cube_or_sequence)
        else:
            # key is a slice (or exception will come from list method).
            cube_or_sequence = CubeList(cube_or_sequence)

        super(CubeList, self).__setitem__(key, cube_or_sequence)

    def append(self, cube):
        """Append a cube."""
        self._assert_is_cube(cube)
        super(CubeList, self).append(cube)

    def extend(self, other_cubes):
        """Extend cubelist by appending the cubes contained in other_cubes.

        Parameters
        ----------
        other_cubes :
           A cubelist or other sequence of cubes.

        """
        super(CubeList, self).extend(CubeList(other_cubes))

    def insert(self, index, cube):
        """Insert a cube before index."""
        self._assert_is_cube(cube)
        super(CubeList, self).insert(index, cube)

    def xml(self, checksum=False, order=True, byteorder=True):
        """Return a string of the XML that this list of cubes represents."""
        doc = Document()
        cubes_xml_element = doc.createElement("cubes")
        cubes_xml_element.setAttribute("xmlns", XML_NAMESPACE_URI)

        for cube_obj in self:
            cubes_xml_element.appendChild(
                cube_obj._xml_element(
                    doc, checksum=checksum, order=order, byteorder=byteorder
                )
            )

        doc.appendChild(cubes_xml_element)

        # return our newly created XML string
        doc = Cube._sort_xml_attrs(doc)
        return doc.toprettyxml(indent="  ")

    def extract(self, constraints):
        """Filter each of the cubes which can be filtered by the given constraints.

        This method iterates over each constraint given, and subsets each of
        the cubes in this CubeList where possible. Thus, a CubeList of length
        **n** when filtered with **m** constraints can generate a maximum of
        **m * n** cubes.

        Parameters
        ----------
        constraints : :class:`~iris.Constraint` or iterable of constraints
            A single constraint or an iterable.

        """
        return self._extract_and_merge(self, constraints, strict=False)

    def extract_cube(self, constraint):
        """Extract a single cube from a CubeList, and return it.

        Extract a single cube from a CubeList, and return it.
        Raise an error if the extract produces no cubes, or more than one.

        Parameters
        ----------
        constraint : :class:`~iris.Constraint`
            The constraint to extract with.

        See Also
        --------
        iris.cube.CubeList.extract :
            Filter each of the cubes which can be filtered by the given constraints.

        """
        # Just validate this, so we can accept strings etc, but not multiples.
        constraint = iris._constraints.as_constraint(constraint)
        return self._extract_and_merge(
            self, constraint, strict=True, return_single_cube=True
        )

    def extract_cubes(self, constraints):
        """Extract specific cubes from a CubeList, one for each given constraint.

        Extract specific cubes from a CubeList, one for each given constraint.
        Each constraint must produce exactly one cube, otherwise an error is
        raised.

        Parameters
        ----------
        constraints : iter of, or single, :class:`~iris.Constraint`
            The constraints to extract with.

        See Also
        --------
        iris.cube.CubeList.extract :
            Filter each of the cubes which can be filtered by the given constraints.

        """
        return self._extract_and_merge(
            self, constraints, strict=True, return_single_cube=False
        )

    @staticmethod
    def _extract_and_merge(cubes, constraints, strict=False, return_single_cube=False):
        constraints = iris._constraints.list_of_constraints(constraints)

        # group the resultant cubes by constraints in a dictionary
        constraint_groups = dict(
            [(constraint, CubeList()) for constraint in constraints]
        )
        for c in cubes:
            for constraint, cube_list in constraint_groups.items():
                sub_cube = constraint.extract(c)
                if sub_cube is not None:
                    cube_list.append(sub_cube)

        result = CubeList()
        for constraint in constraints:
            constraint_cubes = constraint_groups[constraint]
            if strict and len(constraint_cubes) != 1:
                msg = "Got %s cubes for constraint %r, expecting 1." % (
                    len(constraint_cubes),
                    constraint,
                )
                raise iris.exceptions.ConstraintMismatchError(msg)
            result.extend(constraint_cubes)

        if return_single_cube:
            if len(result) != 1:
                # Practically this should never occur, as we now *only* request
                # single cube result for 'extract_cube'.
                msg = "Got {!s} cubes for constraints {!r}, expecting 1."
                raise iris.exceptions.ConstraintMismatchError(
                    msg.format(len(result), constraints)
                )
            result = result[0]

        return result

    def extract_overlapping(self, coord_names):
        """Return a :class:`CubeList` of cubes extracted over regions.

        Return a :class:`CubeList` of cubes extracted over regions
        where the coordinates overlap, for the coordinates
        in coord_names.

        Parameters
        ----------
        coord_names : str or list of str
           A string or list of strings of the names of the coordinates
           over which to perform the extraction.

        """
        if isinstance(coord_names, str):
            coord_names = [coord_names]

        def make_overlap_fn(coord_name):
            def overlap_fn(cell):
                return all(cell in cube.coord(coord_name).cells() for cube in self)

            return overlap_fn

        coord_values = {
            coord_name: make_overlap_fn(coord_name) for coord_name in coord_names
        }

        return self.extract(iris.Constraint(coord_values=coord_values))

    def merge_cube(self):
        """Return the merged contents of the :class:`CubeList` as a single :class:`Cube`.

        If it is not possible to merge the `CubeList` into a single
        `Cube`, a :class:`~iris.exceptions.MergeError` will be raised
        describing the reason for the failure.

        For example:

            >>> cube_1 = iris.cube.Cube([1, 2])
            >>> cube_1.add_aux_coord(iris.coords.AuxCoord(0, long_name='x'))
            >>> cube_2 = iris.cube.Cube([3, 4])
            >>> cube_2.add_aux_coord(iris.coords.AuxCoord(1, long_name='x'))
            >>> cube_2.add_dim_coord(
            ...     iris.coords.DimCoord([0, 1], long_name='z'), 0)
            >>> single_cube = iris.cube.CubeList([cube_1, cube_2]).merge_cube()
            Traceback (most recent call last):
            ...
            iris.exceptions.MergeError: failed to merge into a single cube.
              Coordinates in cube.dim_coords differ: z.
              Coordinate-to-dimension mapping differs for cube.dim_coords.

        """
        if not self:
            raise ValueError("can't merge an empty CubeList")

        # Register each of our cubes with a single ProtoCube.
        proto_cube = iris._merge.ProtoCube(self[0])
        for c in self[1:]:
            proto_cube.register(c, error_on_mismatch=True)

        # Extract the merged cube from the ProtoCube.
        (merged_cube,) = proto_cube.merge()
        return merged_cube

    def merge(self, unique=True):
        """Return the :class:`CubeList` resulting from merging this :class:`CubeList`.

        Parameters
        ----------
        unique : bool, default=True
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        Examples
        --------
        This combines cubes with different values of an auxiliary scalar
        coordinate, by constructing a new dimension.

        .. testsetup::

            import iris
            c1 = iris.cube.Cube([0,1,2], long_name='some_parameter')
            xco = iris.coords.DimCoord([11, 12, 13], long_name='x_vals')
            c1.add_dim_coord(xco, 0)
            c1.add_aux_coord(iris.coords.AuxCoord([100], long_name='y_vals'))
            c2 = c1.copy()
            c2.coord('y_vals').points = [200]

        ::

            >>> print(c1)
            some_parameter / (unknown)          (x_vals: 3)
                 Dimension coordinates:
                      x_vals                           x
                 Scalar coordinates:
                      y_vals: 100
            >>> print(c2)
            some_parameter / (unknown)          (x_vals: 3)
                 Dimension coordinates:
                      x_vals                           x
                 Scalar coordinates:
                      y_vals: 200
            >>> cube_list = iris.cube.CubeList([c1, c2])
            >>> new_cube = cube_list.merge()[0]
            >>> print(new_cube)
            some_parameter / (unknown)          (y_vals: 2; x_vals: 3)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(new_cube.coord('y_vals').points)
            [100 200]
            >>>

        Contrast this with :meth:`iris.cube.CubeList.concatenate`, which joins
        cubes along an existing dimension.

        .. note::

            Cubes may contain additional dimensional elements such as auxiliary
            coordinates, cell measures or ancillary variables.
            A group of similar cubes can only merge to a single result if all such
            elements are identical in every input cube : they are then present,
            unchanged, in the merged output cube.

        .. note::

            If time coordinates in the list of cubes have differing epochs then
            the cubes will not be able to be merged. If this occurs, use
            :func:`iris.util.unify_time_units` to normalise the epochs of the
            time coordinates so that the cubes can be merged.

        """
        # Register each of our cubes with its appropriate ProtoCube.
        proto_cubes_by_name = {}
        for c in self:
            name = c.standard_name
            proto_cubes = proto_cubes_by_name.setdefault(name, [])
            proto_cube = None

            for target_proto_cube in proto_cubes:
                if target_proto_cube.register(c):
                    proto_cube = target_proto_cube
                    break

            if proto_cube is None:
                proto_cube = iris._merge.ProtoCube(c)
                proto_cubes.append(proto_cube)

        # Emulate Python 2 behaviour.
        def _none_sort(item):
            return (item is not None, item)

        # Extract all the merged cubes from the ProtoCubes.
        merged_cubes = CubeList()
        for name in sorted(proto_cubes_by_name, key=_none_sort):
            for proto_cube in proto_cubes_by_name[name]:
                merged_cubes.extend(proto_cube.merge(unique=unique))

        return merged_cubes

    def concatenate_cube(
        self,
        check_aux_coords=True,
        check_cell_measures=True,
        check_ancils=True,
        check_derived_coords=True,
    ):
        """Return the concatenated contents of the :class:`CubeList` as a single :class:`Cube`.

        If it is not possible to concatenate the `CubeList` into a single
        `Cube`, a :class:`~iris.exceptions.ConcatenateError` will be raised
        describing the reason for the failure.

        Parameters
        ----------
        check_aux_coords : bool, default=True
            Checks if the points and bounds of auxiliary coordinates of the
            cubes match. This check is not applied to auxiliary coordinates
            that span the dimension the concatenation is occurring along.
            Defaults to True.
        check_cell_measures : bool, default=True
            Checks if the data of cell measures of the cubes match. This check
            is not applied to cell measures that span the dimension the
            concatenation is occurring along. Defaults to True.
        check_ancils : bool, default=True
            Checks if the data of ancillary variables of the cubes match. This
            check is not applied to ancillary variables that span the dimension
            the concatenation is occurring along. Defaults to True.
        check_derived_coords : bool, default=True
            Checks if the points and bounds of derived coordinates of the cubes
            match. This check is not applied to derived coordinates that span
            the dimension the concatenation is occurring along. Note that
            differences in scalar coordinates and dimensional coordinates used
            to derive the coordinate are still checked. Checks for auxiliary
            coordinates used to derive the coordinates can be ignored with
            `check_aux_coords`. Defaults to True.

        Notes
        -----
        .. note::

            Concatenation cannot occur along an anonymous dimension.

        """
        from iris._concatenate import concatenate

        if not self:
            raise ValueError("can't concatenate an empty CubeList")

        names = [cube.metadata.name() for cube in self]
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) == 1:
            res = concatenate(
                self,
                error_on_mismatch=True,
                check_aux_coords=check_aux_coords,
                check_cell_measures=check_cell_measures,
                check_ancils=check_ancils,
                check_derived_coords=check_derived_coords,
            )
            n_res_cubes = len(res)
            if n_res_cubes == 1:
                return res[0]
            else:
                msgs = []
                msgs.append("An unexpected problem prevented concatenation.")
                msgs.append(
                    "Expected only a single cube, found {}.".format(n_res_cubes)
                )
                raise iris.exceptions.ConcatenateError(msgs)
        else:
            msgs = []
            msgs.append(
                "Cube names differ: {} != {}".format(unique_names[0], unique_names[1])
            )
            raise iris.exceptions.ConcatenateError(msgs)

    def concatenate(
        self,
        check_aux_coords=True,
        check_cell_measures=True,
        check_ancils=True,
        check_derived_coords=True,
    ):
        """Concatenate the cubes over their common dimensions.

        Parameters
        ----------
        check_aux_coords : bool, default=True
            Checks if the points and bounds of auxiliary coordinates of the
            cubes match. This check is not applied to auxiliary coordinates
            that span the dimension the concatenation is occurring along.
            Defaults to True.
        check_cell_measures : bool, default=True
            Checks if the data of cell measures of the cubes match. This check
            is not applied to cell measures that span the dimension the
            concatenation is occurring along. Defaults to True.
        check_ancils : bool, default=True
            Checks if the data of ancillary variables of the cubes match. This
            check is not applied to ancillary variables that span the dimension
            the concatenation is occurring along. Defaults to True.
        check_derived_coords : bool, default=True
            Checks if the points and bounds of derived coordinates of the cubes
            match. This check is not applied to derived coordinates that span
            the dimension the concatenation is occurring along. Note that
            differences in scalar coordinates and dimensional coordinates used
            to derive the coordinate are still checked. Checks for auxiliary
            coordinates used to derive the coordinates can be ignored with
            `check_aux_coords`. Defaults to True.

        Returns
        -------
        :class:`iris.cube.CubeList`
            A new :class:`iris.cube.CubeList` of concatenated
            :class:`iris.cube.Cube` instances.

        Notes
        -----
        This combines cubes with a common dimension coordinate, but occupying
        different regions of the coordinate value.  The cubes are joined across
        that dimension.

        .. testsetup::

            import iris
            import numpy as np
            xco = iris.coords.DimCoord([11, 12, 13, 14], long_name='x_vals')
            yco1 = iris.coords.DimCoord([4, 5], long_name='y_vals')
            yco2 = iris.coords.DimCoord([7, 9, 10], long_name='y_vals')
            c1 = iris.cube.Cube(np.zeros((2,4)), long_name='some_parameter')
            c1.add_dim_coord(xco, 1)
            c1.add_dim_coord(yco1, 0)
            c2 = iris.cube.Cube(np.zeros((3,4)), long_name='some_parameter')
            c2.add_dim_coord(xco, 1)
            c2.add_dim_coord(yco2, 0)

        For example::

            >>> print(c1)
            some_parameter / (unknown)          (y_vals: 2; x_vals: 4)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(c1.coord('y_vals').points)
            [4 5]
            >>> print(c2)
            some_parameter / (unknown)          (y_vals: 3; x_vals: 4)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(c2.coord('y_vals').points)
            [ 7  9 10]
            >>> cube_list = iris.cube.CubeList([c1, c2])
            >>> new_cube = cube_list.concatenate()[0]
            >>> print(new_cube)
            some_parameter / (unknown)          (y_vals: 5; x_vals: 4)
                 Dimension coordinates:
                      y_vals                           x          -
                      x_vals                           -          x
            >>> print(new_cube.coord('y_vals').points)
            [ 4  5  7  9 10]
            >>>

        Contrast this with :meth:`iris.cube.CubeList.merge`, which makes a new
        dimension from values of an auxiliary scalar coordinate.

        .. note::

            Cubes may contain 'extra' dimensional elements such as auxiliary
            coordinates, cell measures or ancillary variables.
            For a group of similar cubes to concatenate together into one output, all
            such elements which do not map to the concatenation axis must be identical
            in every input cube : these then appear unchanged in the output.
            Similarly, those elements which *do* map to the concatenation axis must
            have matching properties, but may have different data values : these then
            appear, concatenated, in the output cube.
            If any cubes in a group have dimensional elements which do not match
            correctly, the group will not concatenate to a single output cube.

        .. note::

            If time coordinates in the list of cubes have differing epochs then
            the cubes will not be able to be concatenated. If this occurs, use
            :func:`iris.util.unify_time_units` to normalise the epochs of the
            time coordinates so that the cubes can be concatenated.

        .. note::

            Concatenation cannot occur along an anonymous dimension.

        """
        from iris._concatenate import concatenate

        return concatenate(
            self,
            check_aux_coords=check_aux_coords,
            check_cell_measures=check_cell_measures,
            check_ancils=check_ancils,
            check_derived_coords=check_derived_coords,
        )

    def realise_data(self):
        """Fetch 'real' data for all cubes, in a shared calculation.

        This computes any lazy data, equivalent to accessing each `cube.data`.
        However, lazy calculations and data fetches can be shared between the
        computations, improving performance.

        For example::

            # Form stats.
            a_std = cube_a.collapsed(['x', 'y'], iris.analysis.STD_DEV)
            b_std = cube_b.collapsed(['x', 'y'], iris.analysis.STD_DEV)
            ab_mean_diff = (cube_b - cube_a).collapsed(['x', 'y'],
                                                       iris.analysis.MEAN)
            std_err = (a_std * a_std + b_std * b_std) ** 0.5

            # Compute these stats together (avoiding multiple data passes).
            CubeList([a_std, b_std, ab_mean_diff, std_err]).realise_data()

        .. note::

            Cubes with non-lazy data are not affected.

        """
        _lazy.co_realise_cubes(*self)

    def copy(self):
        """Return a CubeList when CubeList.copy() is called."""
        if isinstance(self, CubeList):
            return deepcopy(self)


def _is_single_item(testee) -> TypeGuard[str | AuxCoord | DimCoord | int]:
    """Return whether this is a single item, rather than an iterable.

    We count string types as 'single', also.

    """
    return isinstance(testee, str) or not isinstance(testee, Iterable)


class CubeAttrsDict(MutableMapping):
    """A :class:`dict`-like object for :attr:`iris.cube.Cube.attributes`.

    A :class:`dict`-like object for :attr:`iris.cube.Cube.attributes`,
    providing unified user access to combined cube "local" and "global" attributes
    dictionaries, with the access behaviour of an ordinary (single) dictionary.

    Properties :attr:`globals` and :attr:`locals` are regular
    :class:`~iris.common.mixin.LimitedAttributeDict`, which can be accessed and
    modified separately.  The :class:`CubeAttrsDict` itself contains *no* additional
    state, but simply provides a 'combined' view of both global + local attributes.

    All the read- and write-type methods, such as ``get()``, ``update()``, ``values()``,
    behave according to the logic documented for : :meth:`__getitem__`,
    :meth:`__setitem__` and :meth:`__iter__`.

    Notes
    -----
    For type testing, ``issubclass(CubeAttrsDict, Mapping)`` is ``True``, but
    ``issubclass(CubeAttrsDict, dict)`` is ``False``.

    Examples
    --------
        >>> from iris.cube import Cube
        >>> cube = Cube([0])
        >>> # CF defines 'history' as global by default.
        >>> cube.attributes.update({"history": "from test-123", "mycode": 3})
        >>> print(cube.attributes)
        {'history': 'from test-123', 'mycode': 3}
        >>> print(repr(cube.attributes))
        CubeAttrsDict(globals={'history': 'from test-123'}, locals={'mycode': 3})

        >>> cube.attributes['history'] += ' +added'
        >>> print(repr(cube.attributes))
        CubeAttrsDict(globals={'history': 'from test-123 +added'}, locals={'mycode': 3})

        >>> cube.attributes.locals['history'] = 'per-variable'
        >>> print(cube.attributes)
        {'history': 'per-variable', 'mycode': 3}
        >>> print(repr(cube.attributes))
        CubeAttrsDict(globals={'history': 'from test-123 +added'}, locals={'mycode': 3, 'history': 'per-variable'})

    """

    # TODO: Create a 'further topic' / 'tech paper' on NetCDF I/O, including
    #  discussion of attribute handling.

    def __init__(
        self,
        combined: Optional[Mapping] = None,
        locals: Optional[Mapping] = None,
        globals: Optional[Mapping] = None,
    ):
        """Create a cube attributes dictionary.

        We support initialisation from a single generic mapping input, using the default
        global/local assignment rules explained at :meth:`__setattr__`, or from
        two separate mappings.  Two separate dicts can be passed in the ``locals``
        and ``globals`` args, **or** via a ``combined`` arg which has its own
        ``.globals`` and ``.locals`` properties -- so this allows passing an existing
        :class:`CubeAttrsDict`, which will be copied.

        Parameters
        ----------
        combined : dict
            Values to init both 'self.globals' and 'self.locals'.  If 'combined' itself
            has attributes named 'locals' and 'globals', these are used to update the
            respective content (after initially setting the individual ones).
            Otherwise, 'combined' is treated as a generic mapping, applied as
            ``self.update(combined)``,
            i.e. it will set locals and/or globals with the same logic as
            :meth:`~iris.cube.CubeAttrsDict.__setitem__` .
        locals : dict
            Initial content for 'self.locals'.
        globals : dict
            Initial content for 'self.globals'.

        Examples
        --------
            >>> from iris.cube import CubeAttrsDict
            >>> # CF defines 'history' as global by default.
            >>> CubeAttrsDict({'history': 'data-story', 'comment': 'this-cube'})
            CubeAttrsDict(globals={'history': 'data-story'}, locals={'comment': 'this-cube'})

            >>> CubeAttrsDict(locals={'history': 'local-history'})
            CubeAttrsDict(globals={}, locals={'history': 'local-history'})

            >>> CubeAttrsDict(globals={'x': 'global'}, locals={'x': 'local'})
            CubeAttrsDict(globals={'x': 'global'}, locals={'x': 'local'})

            >>> x1 = CubeAttrsDict(globals={'x': 1}, locals={'y': 2})
            >>> x2 = CubeAttrsDict(x1)
            >>> x2
            CubeAttrsDict(globals={'x': 1}, locals={'y': 2})

        """
        # First initialise locals + globals, defaulting to empty.
        # See https://github.com/python/mypy/issues/3004
        self.locals = locals  # type: ignore[assignment]
        self.globals = globals  # type: ignore[assignment]
        # Update with combined, if present.
        if combined is not None:
            # Treat a single input with 'locals' and 'globals' properties as an
            # existing CubeAttrsDict, and update from its content.
            # N.B. enforce deep copying, consistent with general Iris usage.
            if hasattr(combined, "globals") and hasattr(combined, "locals"):
                # Copy a mapping with globals/locals, like another 'CubeAttrsDict'
                self.globals.update(deepcopy(combined.globals))
                self.locals.update(deepcopy(combined.locals))
            else:
                # Treat any arbitrary single input value as a mapping (dict), and
                # update from it.
                self.update(dict(deepcopy(combined)))

    #
    # Ensure that the stored local/global dictionaries are "LimitedAttributeDicts".
    #
    @staticmethod
    def _normalise_attrs(
        attributes: Optional[Mapping],
    ) -> LimitedAttributeDict:
        # Convert an input attributes arg into a standard form.
        # N.B. content is always a LimitedAttributeDict, and a deep copy of input.
        # Allow arg of None, etc.
        if not attributes:
            attributes = {}
        else:
            attributes = deepcopy(attributes)

        # Ensure the expected mapping type.
        attributes = LimitedAttributeDict(attributes)
        return attributes

    @property
    def locals(self) -> LimitedAttributeDict:
        return self._locals

    @locals.setter
    def locals(self, attributes: Optional[Mapping]):
        self._locals = self._normalise_attrs(attributes)

    @property
    def globals(self) -> LimitedAttributeDict:
        return self._globals

    @globals.setter
    def globals(self, attributes: Optional[Mapping]):
        self._globals = self._normalise_attrs(attributes)

    #
    # Provide a serialisation interface
    #
    def __getstate__(self):
        return (self.locals, self.globals)

    def __setstate__(self, state):
        self.locals, self.globals = state

    #
    # Support comparison -- required because default operation only compares a single
    # value at each key.
    #
    def __eq__(self, other):
        # For equality, require both globals + locals to match exactly.
        # NOTE: array content works correctly, since 'locals' and 'globals' are always
        # iris.common.mixin.LimitedAttributeDict, which gets this right.
        if not isinstance(other, CubeAttrsDict):
            other = CubeAttrsDict(other)
        result = self.locals == other.locals and self.globals == other.globals
        return result

    #
    # Provide methods duplicating those for a 'dict', but which are *not* provided by
    # MutableMapping, for compatibility with code which expected a cube.attributes to be
    # a :class:`~iris.common.mixin.LimitedAttributeDict`.
    # The extra required methods are :
    #   'copy', 'update', '__ior__', '__or__', '__ror__' and 'fromkeys'.
    #
    def copy(self):
        """Return a copy.

        Implemented with deep copying, consistent with general Iris usage.

        """
        return CubeAttrsDict(self)

    def update(self, *args, **kwargs):
        """Update by adding items from a mapping arg, or keyword-values.

        If the argument is a split dictionary, preserve the local/global nature of its
        keys.
        """
        if args and hasattr(args[0], "globals") and hasattr(args[0], "locals"):
            dic = args[0]
            self.globals.update(dic.globals)
            self.locals.update(dic.locals)
        else:
            super().update(*args)
        super().update(**kwargs)

    def __or__(self, arg):
        """Implement 'or' via 'update'."""
        if not isinstance(arg, Mapping):
            return NotImplemented
        new_dict = self.copy()
        new_dict.update(arg)
        return new_dict

    def __ior__(self, arg):
        """Implement 'ior' via 'update'."""
        self.update(arg)
        return self

    def __ror__(self, arg):
        """Implement 'ror' via 'update'.

        This needs to promote, such that the result is a CubeAttrsDict.
        """
        if not isinstance(arg, Mapping):
            return NotImplemented
        result = CubeAttrsDict(arg)
        result.update(self)
        return result

    @classmethod
    def fromkeys(cls, iterable, value=None):
        """Create a new object with keys taken from an argument, all set to one value.

        If the argument is a split dictionary, preserve the local/global nature of its
        keys.
        """
        if hasattr(iterable, "globals") and hasattr(iterable, "locals"):
            # When main input is a split-attrs dict, create global/local parts from its
            # global/local keys
            result = cls(
                globals=dict.fromkeys(iterable.globals, value),
                locals=dict.fromkeys(iterable.locals, value),
            )
        else:
            # Create from a dict.fromkeys, using default classification of the keys.
            result = cls(dict.fromkeys(iterable, value))
        return result

    #
    # The remaining methods are sufficient to generate a complete standard Mapping
    # API.  See -
    #  https://docs.python.org/3/reference/datamodel.html#emulating-container-types.
    #

    def __iter__(self):
        """Define the combined iteration order.

        Result is: all global keys, then all local ones, but omitting duplicates.

        """
        # NOTE: this means that in the "summary" view, attributes present in both
        # locals+globals are listed first, amongst the globals, even though they appear
        # with the *value* from locals.
        # Otherwise follows order of insertion, as is normal for dicts.
        return itertools.chain(
            self.globals.keys(),
            (x for x in self.locals.keys() if x not in self.globals),
        )

    def __len__(self):
        # Return the number of keys in the 'combined' view.
        return len(list(iter(self)))

    def __getitem__(self, key):
        """Fetch an item from the "combined attributes".

        If the name is present in *both* ``self.locals`` and ``self.globals``, then
        the local value is returned.

        """
        if key in self.locals:
            store = self.locals
        else:
            store = self.globals
        return store[key]

    def __setitem__(self, key, value):
        """Assign an attribute value.

        This may be assigned in either ``self.locals`` or ``self.globals``, chosen as
        follows:

        * If there is an existing setting in either ``.locals`` or ``.globals``, then
          that is updated (i.e. overwritten).

        * If it is present in *both*, only
          ``.locals`` is updated.

        * If there is *no* existing attribute, it is usually created in ``.locals``.
          **However** a handful of "known normally global" cases, as defined by CF,
          go into ``.globals`` instead.
          At present these are : ('conventions', 'featureType', 'history', 'title').
          See `CF Conventions, Appendix A: <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#attribute-appendix>`_ .

        """
        # If an attribute of this name is already present, update that
        # (the local one having priority).
        if key in self.locals:
            store = self.locals
        elif key in self.globals:
            store = self.globals
        else:
            # If NO existing attribute, create local unless it is a "known global" one.
            from iris.fileformats.netcdf.saver import _CF_GLOBAL_ATTRS

            if key in _CF_GLOBAL_ATTRS:
                store = self.globals
            else:
                store = self.locals

        store[key] = value

    def __delitem__(self, key):
        """Remove an attribute.

        Delete from both local + global.

        """
        if key in self.locals:
            del self.locals[key]
        if key in self.globals:
            del self.globals[key]

    def __str__(self):
        # Print it just like a "normal" dictionary.
        # Convert to a normal dict to do that.
        return str(dict(self))

    def __repr__(self):
        # Special repr form, showing "real" contents.
        return f"CubeAttrsDict(globals={self.globals}, locals={self.locals})"


class Cube(CFVariableMixin):
    """A single Iris cube of data and metadata.

    Typically obtained from :func:`iris.load`, :func:`iris.load_cube`,
    :func:`iris.load_cubes`, or from the manipulation of existing cubes.

    For example:

        >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
        >>> print(cube)
        air_temperature / (K)               (latitude: 73; longitude: 96)
            Dimension coordinates:
                latitude                             x              -
                longitude                            -              x
            Scalar coordinates:
                forecast_period             \
6477 hours, bound=(-28083.0, 6477.0) hours
                forecast_reference_time     1998-03-01 03:00:00
                pressure                    1000.0 hPa
                time                        \
1998-12-01 00:00:00, bound=(1994-12-01 00:00:00, 1998-12-01 00:00:00)
            Cell methods:
                0                           time: mean within years
                1                           time: mean over years
            Attributes:
                STASH                       m01s16i203
                source                      'Data from Met Office Unified Model'


    See the :doc:`user guide</userguide/index>` for more information.

    """

    #: Indicates to client code that the object supports
    #: "orthogonal indexing", which means that slices that are 1d arrays
    #: or lists slice along each dimension independently. This behavior
    #: is similar to Fortran or Matlab, but different than numpy.
    __orthogonal_indexing__ = True

    @classmethod
    def _sort_xml_attrs(cls, doc):
        """Return a copy with all element attributes sorted in alphabetical order.

        Take an xml document and returns a copy with all element
        attributes sorted in alphabetical order.

        This is a private utility method required by iris to maintain
        legacy xml behaviour beyond python 3.7.

        Parameters
        ----------
        doc : :class:`xml.dom.minidom.Document`

        Returns
        -------
        :class:`xml.dom.minidom.Document`
            The :class:`xml.dom.minidom.Document` with sorted element
            attributes.

        """
        from xml.dom.minidom import Document

        def _walk_nodes(node):
            """Note: _walk_nodes is called recursively on child elements."""
            # we don't want to copy the children here, so take a shallow copy
            new_node = node.cloneNode(deep=False)

            # Versions of python <3.8 order attributes in alphabetical order.
            # Python >=3.8 order attributes in insert order.  For consistent behaviour
            # across both, we'll go with alphabetical order always.
            # Remove all the attribute nodes, then add back in alphabetical order.
            attrs = [
                new_node.getAttributeNode(attr_name).cloneNode(deep=True)
                for attr_name in sorted(node.attributes.keys())
            ]
            for attr in attrs:
                new_node.removeAttributeNode(attr)
            for attr in attrs:
                new_node.setAttributeNode(attr)

            if node.childNodes:
                children = [_walk_nodes(x) for x in node.childNodes]
                for c in children:
                    new_node.appendChild(c)

            return new_node

        nodes = _walk_nodes(doc.documentElement)
        new_doc = Document()
        new_doc.appendChild(nodes)

        return new_doc

    def __init__(
        self,
        data: np.typing.ArrayLike,
        standard_name: str | None = None,
        long_name: str | None = None,
        var_name: str | None = None,
        units: Unit | str | None = None,
        attributes: Mapping | None = None,
        cell_methods: Iterable[CellMethod] | None = None,
        dim_coords_and_dims: Iterable[tuple[DimCoord, int]] | None = None,
        aux_coords_and_dims: Iterable[tuple[AuxCoord, int | Iterable[int]]]
        | None = None,
        aux_factories: Iterable[AuxCoordFactory] | None = None,
        cell_measures_and_dims: Iterable[tuple[CellMeasure, int]] | None = None,
        ancillary_variables_and_dims: Iterable[tuple[AncillaryVariable, int]]
        | None = None,
    ):
        """Create a cube with data and optional metadata.

        Not typically used - normally cubes are obtained by loading data
        (e.g. :func:`iris.load`) or from manipulating existing cubes.

        Parameters
        ----------
        data :
            This object defines the shape of the cube and the phenomenon
            value in each cell.

            ``data`` can be a :class:`dask.array.Array`, a
            :class:`numpy.ndarray`, a NumPy array
            subclass (such as :class:`numpy.ma.MaskedArray`), or
            array_like (as described in :func:`numpy.asarray`).

            See :attr:`Cube.data<iris.cube.Cube.data>`.
        standard_name :
            The standard name for the Cube's data.
        long_name :
            An unconstrained description of the cube.
        var_name :
            The NetCDF variable name for the cube.
        units :
            The unit of the cube, e.g. ``"m s-1"`` or ``"kelvin"``.
        attributes :
            A dictionary of cube attributes.
        cell_methods :
            A tuple of CellMethod objects, generally set by Iris, e.g.
            ``(CellMethod("mean", coords='latitude'), )``.
        dim_coords_and_dims :
            A list of coordinates with scalar dimension mappings, e.g
            ``[(lat_coord, 0), (lon_coord, 1)]``.
        aux_coords_and_dims :
            A list of coordinates with dimension mappings,
            e.g ``[(lat_coord, 0), (lon_coord, (0, 1))]``.
            See also :meth:`Cube.add_dim_coord()<iris.cube.Cube.add_dim_coord>`
            and :meth:`Cube.add_aux_coord()<iris.cube.Cube.add_aux_coord>`.
        aux_factories :
            A list of auxiliary coordinate factories. See
            :mod:`iris.aux_factory`.
        cell_measures_and_dims :
            A list of CellMeasures with dimension mappings.
        ancillary_variables_and_dims :
            A list of AncillaryVariables with dimension mappings.

        Examples
        --------
        ::

            >>> from iris.coords import DimCoord
            >>> from iris.cube import Cube
            >>> latitude = DimCoord(np.linspace(-90, 90, 4),
            ...                     standard_name='latitude',
            ...                     units='degrees')
            >>> longitude = DimCoord(np.linspace(45, 360, 8),
            ...                      standard_name='longitude',
            ...                      units='degrees')
            >>> cube = Cube(np.zeros((4, 8), np.float32),
            ...             dim_coords_and_dims=[(latitude, 0),
            ...                                  (longitude, 1)])

        """
        # Temporary error while we transition the API.
        if isinstance(data, str):
            raise TypeError("Invalid data type: {!r}.".format(data))

        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CubeMetadata)

        # Initialise the cube data manager.
        self._data_manager = DataManager(data)

        #: The "standard name" for the Cube's phenomenon.
        self.standard_name = standard_name

        #: An instance of :class:`cf_units.Unit` describing the Cube's data.
        self.units = units

        #: The "long name" for the Cube's phenomenon.
        self.long_name = long_name

        #: The NetCDF variable name for the Cube.
        self.var_name = var_name

        # See https://github.com/python/mypy/issues/3004.
        self.cell_methods = cell_methods  # type: ignore[assignment]

        #: A dictionary for arbitrary Cube metadata.
        #: A few keys are restricted - see :class:`CubeAttrsDict`.
        # See https://github.com/python/mypy/issues/3004.
        self.attributes = attributes  # type: ignore[assignment]

        # Coords
        self._dim_coords_and_dims: list[tuple[DimCoord, int]] = []
        self._aux_coords_and_dims: list[
            tuple[AuxCoord | DimCoord, tuple[int, ...]]
        ] = []
        self._aux_factories: list[AuxCoordFactory] = []

        # Cell Measures
        self._cell_measures_and_dims: list[tuple[CellMeasure, tuple[int, ...]]] = []

        # Ancillary Variables
        self._ancillary_variables_and_dims: list[
            tuple[AncillaryVariable, tuple[int, ...]]
        ] = []

        identities = set()
        if dim_coords_and_dims:
            dims = set()
            for coord, dim in dim_coords_and_dims:
                identity = coord.standard_name, coord.long_name
                if identity not in identities and dim not in dims:
                    self._add_unique_dim_coord(coord, dim)
                else:
                    self.add_dim_coord(coord, dim)
                identities.add(identity)
                dims.add(dim)

        if aux_coords_and_dims:
            for auxcoord, auxdims in aux_coords_and_dims:
                identity = auxcoord.standard_name, auxcoord.long_name
                if identity not in identities:
                    self._add_unique_aux_coord(auxcoord, auxdims)
                else:
                    self.add_aux_coord(auxcoord, auxdims)
                identities.add(identity)

        if aux_factories:
            for factory in aux_factories:
                self.add_aux_factory(factory)

        if cell_measures_and_dims:
            for cell_measure, cmdims in cell_measures_and_dims:
                self.add_cell_measure(cell_measure, cmdims)

        if ancillary_variables_and_dims:
            for ancillary_variable, avdims in ancillary_variables_and_dims:
                self.add_ancillary_variable(ancillary_variable, avdims)

    @property
    def _names(self) -> tuple[str | None, str | None, str | None, str | None]:
        """Tuple containing the value of each name participating in the identity of a :class:`iris.cube.Cube`.

        A tuple containing the value of each name participating in the identity
        of a :class:`iris.cube.Cube`. This includes the standard name,
        long name, NetCDF variable name, and the STASH from the attributes
        dictionary.

        """
        return self._metadata_manager._names

    #
    # Ensure that .attributes is always a :class:`CubeAttrsDict`.
    #
    @property  # type: ignore[override]
    def attributes(self) -> CubeAttrsDict:
        return super().attributes  # type: ignore[return-value]

    @attributes.setter
    def attributes(self, attributes: Mapping | None) -> None:
        """Override to CfVariableMixin.attributes.setter.

        An override to CfVariableMixin.attributes.setter, which ensures that Cube
        attributes are stored in a way which distinguishes global + local ones.

        """
        self._metadata_manager.attributes = CubeAttrsDict(attributes or {})

    def _dimensional_metadata(self, name_or_dimensional_metadata):
        """Return a single _DimensionalMetadata instance that matches.

        Return a single _DimensionalMetadata instance that matches the given
        name_or_dimensional_metadata. If one is not found, raise an error.

        """
        found_item = None
        for cube_method in [
            self.coord,
            self.cell_measure,
            self.ancillary_variable,
        ]:
            try:
                found_item = cube_method(name_or_dimensional_metadata)
                if found_item:
                    break
            except KeyError:
                pass
        if not found_item:
            raise KeyError(f"{name_or_dimensional_metadata} was not found in {self}.")
        return found_item

    def is_compatible(
        self,
        other: Cube | CubeMetadata,
        ignore: Iterable[str] | str | None = None,
    ) -> bool:
        """Return whether the cube is compatible with another.

        Compatibility is determined by comparing :meth:`iris.cube.Cube.name()`,
        :attr:`iris.cube.Cube.units`, :attr:`iris.cube.Cube.cell_methods` and
        :attr:`iris.cube.Cube.attributes` that are present in both objects.

        Parameters
        ----------
        other :
            An instance of :class:`iris.cube.Cube` or
            :class:`iris.cube.CubeMetadata`.
        ignore :
           A single attribute key or iterable of attribute keys to ignore when
           comparing the cubes. Default is None. To ignore all attributes set
           this to other.attributes.

        Returns
        -------
        bool

        Notes
        -----
        .. seealso::

            :meth:`iris.util.describe_diff()`

        .. note::

            This function does not indicate whether the two cubes can be
            merged, instead it checks only the four items quoted above for
            equality. Determining whether two cubes will merge requires
            additional logic that is beyond the scope of this method.

        """
        compatible = (
            self.name() == other.name()
            and self.units == other.units
            and self.cell_methods == other.cell_methods
        )

        if compatible:
            common_keys = set(self.attributes).intersection(other.attributes)
            if ignore is not None:
                if isinstance(ignore, str):
                    ignore = (ignore,)
                common_keys = common_keys.difference(ignore)
            for key in common_keys:
                if np.any(self.attributes[key] != other.attributes[key]):
                    compatible = False
                    break

        return compatible

    def convert_units(self, unit: str | Unit) -> None:
        """Change the cube's units, converting the values in the data array.

        For example, if a cube's :attr:`~iris.cube.Cube.units` are
        kelvin then::

            cube.convert_units('celsius')

        will change the cube's :attr:`~iris.cube.Cube.units` attribute to
        celsius and subtract 273.15 from each value in
        :attr:`~iris.cube.Cube.data`.

        Full list of supported units can be found in the UDUNITS-2 documentation
        https://docs.unidata.ucar.edu/udunits/current/#Database

        This operation preserves lazy data.

        """
        # If the cube has units convert the data.
        if self.units.is_unknown():
            raise iris.exceptions.UnitConversionError(
                "Cannot convert from unknown units. "
                'The "cube.units" attribute may be set directly.'
            )
        if self.has_lazy_data():
            # Make fixed copies of old + new units for a delayed conversion.
            old_unit = Unit(self.units)
            new_unit = unit

            pointwise_convert = partial(old_unit.convert, other=new_unit)

            new_data = _lazy.lazy_elementwise(self.lazy_data(), pointwise_convert)
        else:
            new_data = self.units.convert(self.data, unit)
        self.data = new_data
        self.units = unit

    def add_cell_method(self, cell_method: CellMethod) -> None:
        """Add a :class:`~iris.coords.CellMethod` to the Cube."""
        self.cell_methods += (cell_method,)

    def add_aux_coord(
        self,
        coord: AuxCoord | DimCoord,
        data_dims: Iterable[int] | int | None = None,
    ) -> None:
        """Add a CF auxiliary coordinate to the cube.

        Parameters
        ----------
        coord :
            The :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
            instance to add to the cube.
        data_dims :
            Integer or iterable of integers giving the data dimensions spanned
            by the coordinate.

        Raises
        ------
        ValueError
            Raises a ValueError if a coordinate with identical metadata already
            exists on the cube.

        See Also
        --------
        remove_coord :
            Remove a coordinate from the cube.

        """
        if self.coords(coord):  # TODO: just fail on duplicate object
            raise iris.exceptions.CannotAddError(
                "Duplicate coordinates are not permitted."
            )
        self._add_unique_aux_coord(coord, data_dims)

    def _check_multi_dim_metadata(
        self,
        metadata: iris.coords._DimensionalMetadata,
        data_dims: Iterable[int] | int | None,
    ) -> tuple[int, ...]:
        # Convert to a tuple of integers
        if data_dims is None:
            data_dims = tuple()
        elif isinstance(data_dims, Iterable):
            data_dims = tuple(int(d) for d in data_dims)
        else:
            data_dims = (int(data_dims),)

        if data_dims:
            if len(data_dims) != metadata.ndim:
                msg = "Invalid data dimensions: {} given, {} expected for {!r}.".format(
                    len(data_dims), metadata.ndim, metadata.name()
                )
                raise iris.exceptions.CannotAddError(msg)
            # Check compatibility with the shape of the data
            for i, dim in enumerate(data_dims):
                if metadata.shape[i] != self.shape[dim]:
                    msg = (
                        "Unequal lengths. Cube dimension {} => {};"
                        " metadata {!r} dimension {} => {}."
                    )
                    raise iris.exceptions.CannotAddError(
                        msg.format(
                            dim,
                            self.shape[dim],
                            metadata.name(),
                            i,
                            metadata.shape[i],
                        )
                    )
        elif metadata.shape != (1,):
            msg = "Missing data dimensions for multi-valued {} {!r}"
            msg = msg.format(metadata.__class__.__name__, metadata.name())
            raise iris.exceptions.CannotAddError(msg)
        return data_dims

    def _add_unique_aux_coord(
        self,
        coord: AuxCoord | DimCoord,
        data_dims: Iterable[int] | int | None,
    ) -> None:
        data_dims = self._check_multi_dim_metadata(coord, data_dims)

        def is_mesh_coord(anycoord: iris.coords.Coord) -> TypeGuard[MeshCoord]:
            return hasattr(anycoord, "mesh")

        if is_mesh_coord(coord):
            mesh = self.mesh
            if mesh:
                msg = (
                    "{item} of Meshcoord {coord!r} is "
                    "{thisval!r}, which does not match existing "
                    "cube {item} of {ownval!r}."
                )
                if coord.mesh != mesh:
                    raise iris.exceptions.CannotAddError(
                        msg.format(
                            item="mesh",
                            coord=coord,
                            thisval=coord.mesh,
                            ownval=mesh,
                        )
                    )
                location = self.location
                if coord.location != location:
                    raise iris.exceptions.CannotAddError(
                        msg.format(
                            item="location",
                            coord=coord,
                            thisval=coord.location,
                            ownval=location,
                        )
                    )
                mesh_dims = (self.mesh_dim(),)
                if data_dims != mesh_dims:
                    raise iris.exceptions.CannotAddError(
                        msg.format(
                            item="mesh dimension",
                            coord=coord,
                            thisval=data_dims,
                            ownval=mesh_dims,
                        )
                    )

        self._aux_coords_and_dims.append((coord, data_dims))

    def add_aux_factory(self, aux_factory: AuxCoordFactory) -> None:
        """Add an auxiliary coordinate factory to the cube.

        Parameters
        ----------
        aux_factory :
            The :class:`iris.aux_factory.AuxCoordFactory` instance to add.

        """
        if not isinstance(aux_factory, iris.aux_factory.AuxCoordFactory):
            raise TypeError(
                "Factory must be a subclass of iris.aux_factory.AuxCoordFactory."
            )

        # Get all 'real' coords (i.e. not derived ones) : use private data
        # rather than cube.coords(), as that is quite slow.
        def coordsonly(coords_and_dims):
            return [coord for coord, dims in coords_and_dims]

        cube_coords = coordsonly(self._dim_coords_and_dims) + coordsonly(
            self._aux_coords_and_dims
        )

        for dependency in aux_factory.dependencies:
            ref_coord = aux_factory.dependencies[dependency]
            if ref_coord is not None and ref_coord not in cube_coords:
                msg = "{} coordinate for factory is not present on cube {}"
                raise iris.exceptions.CannotAddError(
                    msg.format(ref_coord.name(), self.name())
                )
        self._aux_factories.append(aux_factory)

    def add_cell_measure(
        self,
        cell_measure: CellMeasure,
        data_dims: Iterable[int] | int | None = None,
    ) -> None:
        """Add a CF cell measure to the cube.

        Parameters
        ----------
        cell_measure :
            The :class:`iris.coords.CellMeasure`
            instance to add to the cube.
        data_dims :
            Integer or iterable of integers giving the data dimensions spanned
            by the coordinate.

        Raises
        ------
        ValueError
            Raises a ValueError if a cell_measure with identical metadata already
            exists on the cube.

        See Also
        --------
        remove_cell_measure :
            Remove a cell measure from the cube.

        """
        if self.cell_measures(cell_measure):
            raise iris.exceptions.CannotAddError(
                "Duplicate cell_measures are not permitted."
            )
        data_dims = self._check_multi_dim_metadata(cell_measure, data_dims)
        self._cell_measures_and_dims.append((cell_measure, data_dims))
        self._cell_measures_and_dims.sort(
            key=lambda cm_dims: (cm_dims[0].metadata, cm_dims[1])
        )

    def add_ancillary_variable(
        self,
        ancillary_variable: AncillaryVariable,
        data_dims: Iterable[int] | int | None = None,
    ) -> None:
        """Add a CF ancillary variable to the cube.

        Parameters
        ----------
        ancillary_variable :
            The :class:`iris.coords.AncillaryVariable` instance to be added to
            the cube.
        data_dims :
            Integer or iterable of integers giving the data dimensions spanned
            by the ancillary variable.

        Raises
        ------
        ValueError
            Raises a ValueError if an ancillary variable with identical metadata
            already exists on the cube.

        """
        if self.ancillary_variables(ancillary_variable):
            raise iris.exceptions.CannotAddError(
                "Duplicate ancillary variables not permitted"
            )

        data_dims = self._check_multi_dim_metadata(ancillary_variable, data_dims)
        self._ancillary_variables_and_dims.append((ancillary_variable, data_dims))
        self._ancillary_variables_and_dims.sort(
            key=lambda av_dims: (av_dims[0].metadata, av_dims[1])
        )

    def add_dim_coord(self, dim_coord: DimCoord, data_dim: int | tuple[int]) -> None:
        """Add a CF coordinate to the cube.

        Parameters
        ----------
        dim_coord :
            The :class:`iris.coords.DimCoord` instance to add to the cube.
        data_dim :
            Integer giving the data dimension spanned by the coordinate.

        Raises
        ------
        ValueError
            Raises a ValueError if a coordinate with identical metadata already
            exists on the cube or if a coord already exists for the
            given dimension.

        See Also
        --------
        remove_coord :
            Remove a coordinate from the cube.

        """
        if self.coords(dim_coord):
            raise iris.exceptions.CannotAddError(
                "The coordinate already exists on the cube. "
                "Duplicate coordinates are not permitted."
            )
        # Check dimension is available
        if self.coords(dimensions=data_dim, dim_coords=True):
            raise iris.exceptions.CannotAddError(
                "A dim_coord is already associated with dimension %d." % data_dim
            )
        self._add_unique_dim_coord(dim_coord, data_dim)

    def _add_unique_dim_coord(
        self,
        dim_coord: DimCoord,
        data_dim: int | tuple[int],
    ) -> None:
        if isinstance(dim_coord, iris.coords.AuxCoord):
            raise iris.exceptions.CannotAddError(
                "The dim_coord may not be an AuxCoord instance."
            )

        # Convert data_dim to a single integer
        if isinstance(data_dim, Container):
            if len(data_dim) != 1:
                raise iris.exceptions.CannotAddError(
                    "The supplied data dimension must be a single number."
                )
            data_dim = int(list(data_dim)[0])
        else:
            data_dim = int(data_dim)

        # Check data_dim value is valid
        if data_dim < 0 or data_dim >= self.ndim:
            raise iris.exceptions.CannotAddError(
                "The cube does not have the specified dimension (%d)" % data_dim
            )

        # Check compatibility with the shape of the data
        if dim_coord.shape[0] != self.shape[data_dim]:
            msg = "Unequal lengths. Cube dimension {} => {}; coord {!r} => {}."
            raise iris.exceptions.CannotAddError(
                msg.format(
                    data_dim,
                    self.shape[data_dim],
                    dim_coord.name(),
                    len(dim_coord.points),
                )
            )

        self._dim_coords_and_dims.append((dim_coord, int(data_dim)))

    def remove_aux_factory(self, aux_factory: AuxCoordFactory) -> None:
        """Remove the given auxiliary coordinate factory from the cube."""
        self._aux_factories.remove(aux_factory)

    def _remove_coord(self, coord: DimCoord | AuxCoord) -> None:
        self._dim_coords_and_dims = [
            (coord_, dim)
            for coord_, dim in self._dim_coords_and_dims
            if coord_ is not coord
        ]
        self._aux_coords_and_dims = [
            (coord_, dims)
            for coord_, dims in self._aux_coords_and_dims
            if coord_ is not coord
        ]
        for aux_factory in self.aux_factories:
            if coord.metadata == aux_factory.metadata:
                self.remove_aux_factory(aux_factory)

    def remove_coord(self, coord: str | DimCoord | AuxCoord | AuxCoordFactory) -> None:
        """Remove a coordinate from the cube.

        Parameters
        ----------
        coord :
            The (name of the) coordinate to remove from the cube.

        See Also
        --------
        add_dim_coord :
            Add a CF coordinate to the cube.
        add_aux_coord :
            Add a CF auxiliary coordinate to the cube.

        """
        coord = self.coord(coord)
        self._remove_coord(coord)

        for factory in self.aux_factories:
            factory.update(coord)

    def remove_cell_measure(self, cell_measure: str | CellMeasure) -> None:
        """Remove a cell measure from the cube.

        Parameters
        ----------
        cell_measure :
            The (name of the) cell measure to remove from the cube. As either

            * (a) a :attr:`standard_name`, :attr:`long_name`, or
              :attr:`var_name`. Defaults to value of `default`
              (which itself defaults to `unknown`) as defined in
              :class:`iris.common.CFVariableMixin`.

            * (b) a cell_measure instance with metadata equal to that of
              the desired cell_measures.

        Notes
        -----
        If the argument given does not represent a valid cell_measure on
        the cube, an :class:`iris.exceptions.CellMeasureNotFoundError`
        is raised.

        See Also
        --------
        add_cell_measure :
            Add a CF cell measure to the cube.

        """
        cell_measure = self.cell_measure(cell_measure)

        self._cell_measures_and_dims = [
            (cell_measure_, dim)
            for cell_measure_, dim in self._cell_measures_and_dims
            if cell_measure_ is not cell_measure
        ]

    def remove_ancillary_variable(
        self,
        ancillary_variable: str | AncillaryVariable,
    ) -> None:
        """Remove an ancillary variable from the cube.

        Parameters
        ----------
        ancillary_variable :
            The (name of the) AncillaryVariable to remove from the cube.

        """
        ancillary_variable = self.ancillary_variable(ancillary_variable)

        self._ancillary_variables_and_dims = [
            (ancillary_variable_, dim)
            for ancillary_variable_, dim in self._ancillary_variables_and_dims
            if ancillary_variable_ is not ancillary_variable
        ]

    def replace_coord(self, new_coord: DimCoord | AuxCoord) -> None:
        """Replace the coordinate whose metadata matches the given coordinate."""
        old_coord = self.coord(new_coord)
        dims = self.coord_dims(old_coord)
        was_dimensioned = old_coord in self.dim_coords
        self._remove_coord(old_coord)
        if was_dimensioned and isinstance(new_coord, iris.coords.DimCoord):
            self.add_dim_coord(new_coord, dims[0])
        else:
            self.add_aux_coord(new_coord, dims)

        for factory in self.aux_factories:
            factory.update(old_coord, new_coord)

    def coord_dims(
        self, coord: str | DimCoord | AuxCoord | AuxCoordFactory
    ) -> tuple[int, ...]:
        """Return a tuple of the data dimensions relevant to the given coordinate.

        When searching for the given coordinate in the cube the comparison is
        made using coordinate metadata equality. Hence the given coordinate
        instance need not exist on the cube, and may contain different
        coordinate values.

        Parameters
        ----------
        coord :
            The (name of the) coord to look for.

        Returns
        -------
        tuple:
             A tuple of the data dimensions relevant to the given coordinate.
        """
        name_provided = False
        if isinstance(coord, str):
            # Forced to look-up the coordinate if we only have the name.
            coord = self.coord(coord)
            name_provided = True

        coord_id = id(coord)

        # Dimension of dimension coordinate by object id
        dims_by_id: dict[int, tuple[int, ...]] = {
            id(c): (d,) for c, d in self._dim_coords_and_dims
        }
        # Check for id match - faster than equality check
        match = dims_by_id.get(coord_id)

        if match is None:
            # Dimension/s of auxiliary coordinate by object id
            aux_dims_by_id = {id(c): d for c, d in self._aux_coords_and_dims}
            # Check for id match - faster than equality
            match = aux_dims_by_id.get(coord_id)
            if match is None:
                dims_by_id.update(aux_dims_by_id)

        if match is None and not name_provided:
            # We may have an equivalent coordinate but not the actual
            # cube coordinate instance - so forced to perform coordinate
            # lookup to attempt to retrieve it
            coord = self.coord(coord)
            # Check for id match - faster than equality
            match = dims_by_id.get(id(coord))

        # Search derived aux coordinates
        if match is None:
            target_metadata = coord.metadata

            def matcher(factory):
                return factory.metadata == target_metadata

            factories = filter(matcher, self._aux_factories)
            matches = [factory.derived_dims(self.coord_dims) for factory in factories]
            if matches:
                match = matches[0]

        if match is None:
            raise iris.exceptions.CoordinateNotFoundError(coord.name())

        return match

    def cell_measure_dims(self, cell_measure: str | CellMeasure) -> tuple[int, ...]:
        """Return a tuple of the data dimensions relevant to the given CellMeasure.

        Parameters
        ----------
        cell_measure :
            The (name of the) cell measure to look for.

        Returns
        -------
        tuple:
             A tuple of the data dimensions relevant to the given cell measure.
        """
        cell_measure = self.cell_measure(cell_measure)

        # Search for existing cell measure (object) on the cube, faster lookup
        # than equality - makes no functional difference.
        matches = [
            dims for cm_, dims in self._cell_measures_and_dims if cm_ is cell_measure
        ]

        if not matches:
            raise iris.exceptions.CellMeasureNotFoundError(cell_measure.name())

        return matches[0]

    def ancillary_variable_dims(
        self,
        ancillary_variable: str | AncillaryVariable,
    ) -> tuple[int, ...]:
        """Return a tuple of the data dimensions relevant to the given AncillaryVariable.

        Parameters
        ----------
        ancillary_variable : str or AncillaryVariable
            The (name of the) AncillaryVariable to look for.

        Returns
        -------
        tuple:
             A tuple of the data dimensions relevant to the given ancillary variable.
        """
        ancillary_variable = self.ancillary_variable(ancillary_variable)

        # Search for existing ancillary variable (object) on the cube, faster
        # lookup than equality - makes no functional difference.
        matches = [
            dims
            for av, dims in self._ancillary_variables_and_dims
            if av is ancillary_variable
        ]

        if not matches:
            raise iris.exceptions.AncillaryVariableNotFoundError(
                ancillary_variable.name()
            )

        return matches[0]

    def aux_factory(
        self,
        name: str | None = None,
        standard_name: str | None = None,
        long_name: str | None = None,
        var_name: str | None = None,
    ) -> AuxCoordFactory:
        """Return the single coordinate factory that matches the criteria.

        Return the single coordinate factory that matches the criteria,
        or raises an error if not found.

        Parameters
        ----------
        name :
            If not None, matches against factory.name().
        standard_name :
            The CF standard name of the desired coordinate factory.
            If None, does not check for standard name.
        long_name :
            An unconstrained description of the coordinate factory.
            If None, does not check for long_name.
        var_name :
            The NetCDF variable name of the desired coordinate factory.
            If None, does not check for var_name.

        Returns
        -------
        AuxCoordFactory:
             The single coordinate factory that matches the criteria.

        Notes
        -----
        .. note::

            If the arguments given do not result in precisely 1 coordinate
            factory being matched, an
            :class:`iris.exceptions.CoordinateNotFoundError` is raised.

        """
        factories = list(self.aux_factories)

        if name is not None:
            factories = [factory for factory in factories if factory.name() == name]

        if standard_name is not None:
            factories = [
                factory
                for factory in factories
                if factory.standard_name == standard_name
            ]

        if long_name is not None:
            factories = [
                factory for factory in factories if factory.long_name == long_name
            ]

        if var_name is not None:
            factories = [
                factory for factory in factories if factory.var_name == var_name
            ]

        if len(factories) > 1:
            factory_names = (factory.name() for factory in factories)
            msg = (
                "Expected to find exactly one coordinate factory, but "
                "found {}. They were: {}.".format(
                    len(factories), ", ".join(factory_names)
                )
            )
            raise iris.exceptions.CoordinateNotFoundError(msg)
        elif len(factories) == 0:
            msg = "Expected to find exactly one coordinate factory, but found none."
            raise iris.exceptions.CoordinateNotFoundError(msg)

        return factories[0]

    def coords(
        self,
        name_or_coord: str
        | DimCoord
        | AuxCoord
        | AuxCoordFactory
        | CoordMetadata
        | None = None,
        standard_name: str | None = None,
        long_name: str | None = None,
        var_name: str | None = None,
        attributes: Mapping | None = None,
        axis: iris.util.Axis | None = None,
        contains_dimension=None,
        dimensions: Iterable[int] | int | None = None,
        coord_system=None,
        dim_coords: bool | None = None,
        mesh_coords: bool | None = None,
    ) -> list[DimCoord | AuxCoord]:
        r"""Return a list of coordinates from the :class:`Cube` that match the provided criteria.

        Parameters
        ----------
        name_or_coord :
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of the desired
              coordinate e.g., :class:`~iris.coords.DimCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.
        standard_name :
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard name``.
        long_name :
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.
        var_name :
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.
        attributes :
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.
        axis :
            The desired coordinate axis, see :func:`iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).
        contains_dimension :
            The desired coordinate contains the data dimension. If ``None``, does
            not check for the dimension.
        dimensions :
            The exact data dimensions of the desired coordinate. Coordinates
            with no data dimension can be found with an empty ``tuple`` or
            ``list`` i.e., ``()`` or ``[]``. If ``None``, does not check for
            dimensions.
        coord_system :
            Whether the desired coordinates have a coordinate system equal to
            the given coordinate system. If ``None``, no check is done.
        dim_coords :
            Set to ``True`` to only return coordinates that are the cube's
            dimension coordinates. Set to ``False`` to only return coordinates
            that are the cube's auxiliary, mesh and derived coordinates.
            If ``None``, returns all coordinates.
        mesh_coords :
            Set to ``True`` to return only coordinates which are
            :class:`~iris.mesh.MeshCoord`\'s.
            Set to ``False`` to return only non-mesh coordinates.
            If ``None``, returns all coordinates.

        Returns
        -------
        A list containing zero or more coordinates matching the provided criteria.

        See Also
        --------
        coord :
            For matching exactly one coordinate.


        """
        coords_and_factories: list[DimCoord | AuxCoord | AuxCoordFactory] = []

        if dim_coords in [True, None]:
            coords_and_factories += list(self.dim_coords)

        if dim_coords in [False, None]:
            coords_and_factories += list(self.aux_coords)
            coords_and_factories += list(self.aux_factories)

        if mesh_coords is not None:
            # Select on mesh or non-mesh.
            mesh_coords = bool(mesh_coords)
            # Use duck typing to avoid importing from iris.mesh,
            # which could be a circular import.
            if mesh_coords:
                # *only* MeshCoords
                coords_and_factories = [
                    item for item in coords_and_factories if hasattr(item, "mesh")
                ]
            else:
                # *not* MeshCoords
                coords_and_factories = [
                    item for item in coords_and_factories if not hasattr(item, "mesh")
                ]

        coords_and_factories = metadata_filter(
            coords_and_factories,
            item=name_or_coord,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
        )

        if coord_system is not None:
            coords_and_factories = [
                coord_
                for coord_ in coords_and_factories
                if coord_.coord_system == coord_system
            ]

        if contains_dimension is not None:
            coords_and_factories = [
                coord_
                for coord_ in coords_and_factories
                if contains_dimension in self.coord_dims(coord_)
            ]

        if dimensions is not None:
            if not isinstance(dimensions, Iterable):
                dimensions = [dimensions]
            dimensions = tuple(dimensions)
            coords_and_factories = [
                coord_
                for coord_ in coords_and_factories
                if self.coord_dims(coord_) == dimensions
            ]

        # If any factories remain after the above filters we have to make the
        # coords so they can be returned
        def extract_coord(coord_or_factory):
            if isinstance(coord_or_factory, iris.aux_factory.AuxCoordFactory):
                coord = coord_or_factory.make_coord(self.coord_dims)
            elif isinstance(coord_or_factory, iris.coords.Coord):
                coord = coord_or_factory
            else:
                msg = "Expected Coord or AuxCoordFactory, got {!r}.".format(
                    type(coord_or_factory)
                )
                raise ValueError(msg)
            return coord

        coords = [
            extract_coord(coord_or_factory) for coord_or_factory in coords_and_factories
        ]

        return coords

    def coord(
        self,
        name_or_coord: str
        | DimCoord
        | AuxCoord
        | AuxCoordFactory
        | CoordMetadata
        | None = None,
        standard_name: str | None = None,
        long_name: str | None = None,
        var_name: str | None = None,
        attributes: Mapping | None = None,
        axis: iris.util.Axis | None = None,
        contains_dimension=None,
        dimensions: Iterable[int] | int | None = None,
        coord_system=None,
        dim_coords: bool | None = None,
        mesh_coords: bool | None = None,
    ) -> DimCoord | AuxCoord:
        r"""Return a single coordinate from the :class:`Cube` that matches the provided criteria.

        Parameters
        ----------
        name_or_coord :
            Either,

            * a :attr:`~iris.common.mixin.CFVariableMixin.standard_name`,
              :attr:`~iris.common.mixin.CFVariableMixin.long_name`, or
              :attr:`~iris.common.mixin.CFVariableMixin.var_name` which is
              compared against the :meth:`~iris.common.mixin.CFVariableMixin.name`.

            * a coordinate or metadata instance equal to that of the desired
              coordinate e.g., :class:`~iris.coords.DimCoord` or
              :class:`~iris.common.metadata.CoordMetadata`.
        standard_name :
            The CF standard name of the desired coordinate. If ``None``, does not
            check for ``standard name``.
        long_name :
            An unconstrained description of the coordinate. If ``None``, does not
            check for ``long_name``.
        var_name :
            The NetCDF variable name of the desired coordinate. If ``None``, does
            not check for ``var_name``.
        attributes :
            A dictionary of attributes desired on the coordinates. If ``None``,
            does not check for ``attributes``.
        axis :
            The desired coordinate axis, see :func:`iris.util.guess_coord_axis`.
            If ``None``, does not check for ``axis``. Accepts the values ``X``,
            ``Y``, ``Z`` and ``T`` (case-insensitive).
        contains_dimension :
            The desired coordinate contains the data dimension. If ``None``, does
            not check for the dimension.
        dimensions :
            The exact data dimensions of the desired coordinate. Coordinates
            with no data dimension can be found with an empty ``tuple`` or
            ``list`` i.e., ``()`` or ``[]``. If ``None``, does not check for
            dimensions.
        coord_system :
            Whether the desired coordinates have a coordinate system equal to
            the given coordinate system. If ``None``, no check is done.
        dim_coords :
            Set to ``True`` to only return coordinates that are the cube's
            dimension coordinates. Set to ``False`` to only return coordinates
            that are the cube's auxiliary, mesh and derived coordinates.
            If ``None``, returns all coordinates.
        mesh_coords :
            Set to ``True`` to return only coordinates which are
            :class:`~iris.mesh.MeshCoord`\'s.
            Set to ``False`` to return only non-mesh coordinates.
            If ``None``, returns all coordinates.

        Returns
        -------
        The coordinate that matches the provided criteria.

        Notes
        -----
         .. note::

            If the arguments given do not result in **precisely one** coordinate,
            then a :class:`~iris.exceptions.CoordinateNotFoundError` is raised.

        See Also
        --------
        coords :
            For matching zero or more coordinates.

        """
        coords = self.coords(
            name_or_coord=name_or_coord,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            attributes=attributes,
            axis=axis,
            contains_dimension=contains_dimension,
            dimensions=dimensions,
            coord_system=coord_system,
            dim_coords=dim_coords,
            mesh_coords=mesh_coords,
        )

        if len(coords) > 1:
            emsg = (
                f"Expected to find exactly 1 coordinate, but found {len(coords)}. "
                f"They were: {', '.join(coord.name() for coord in coords)}."
            )
            raise iris.exceptions.CoordinateNotFoundError(emsg)
        elif len(coords) == 0:
            _name = name_or_coord
            if name_or_coord is not None:
                if not isinstance(name_or_coord, str):
                    _name = name_or_coord.name()
                    emsg = (
                        "Expected to find exactly 1 coordinate matching the given "
                        f"{_name!r} coordinate's metadata, but found none."
                    )
                    raise iris.exceptions.CoordinateNotFoundError(emsg)

            bad_name = _name or standard_name or long_name or ""
            emsg = (
                f"Expected to find exactly 1 {bad_name!r} coordinate, but found none."
            )
            raise iris.exceptions.CoordinateNotFoundError(emsg)

        return coords[0]

    def coord_system(
        self,
        spec: str | type[iris.coord_systems.CoordSystem] | None = None,
    ) -> iris.coord_systems.CoordSystem | None:
        """Find the coordinate system of the given type.

        If no target coordinate system is provided then find
        any available coordinate system.

        Parameters
        ----------
        spec :
            The the name or type of a coordinate system subclass.
            E.g. ::

                cube.coord_system("GeogCS")
                cube.coord_system(iris.coord_systems.GeogCS)

            If spec is provided as a type it can be a superclass of
            any coordinate system found.

            If spec is None, then find any available coordinate
            systems within the :class:`iris.cube.Cube`.

        Returns
        -------
        :class:`iris.coord_systems.CoordSystem` or None.

        """
        if isinstance(spec, str) or spec is None:
            spec_name = spec
        else:
            msg = "type %s is not a subclass of CoordSystem" % spec
            assert issubclass(spec, iris.coord_systems.CoordSystem), msg
            spec_name = spec.__name__

        # Gather a temporary list of our unique CoordSystems.
        coord_systems = ClassDict(iris.coord_systems.CoordSystem)
        for coord in self.coords():
            if coord.coord_system:
                coord_systems.add(coord.coord_system, replace=True)

        result = None
        if spec_name is None:
            for key in sorted(coord_systems.keys(), key=lambda class_: class_.__name__):
                result = coord_systems[key]
                break
        else:
            result = coord_systems.get(spec_name)

        return result

    def _any_meshcoord(self) -> MeshCoord | None:
        """Return a MeshCoord if there are any, else None."""
        mesh_coords = self.coords(mesh_coords=True)
        if mesh_coords:
            result = mesh_coords[0]
        else:
            result = None
        return result  # type: ignore[return-value]

    @property
    def mesh(self) -> iris.mesh.MeshXY | None:
        r"""Return the unstructured :class:`~iris.mesh.MeshXY` associated with the cube.

        Return the unstructured :class:`~iris.mesh.MeshXY`
        associated with the cube, if the cube has any
        :class:`~iris.mesh.MeshCoord`,
        or ``None`` if it has none.

        Returns
        -------
        :class:`iris.mesh.MeshXY` or None
            The mesh of the cube
            :class:`~iris.mesh.MeshCoord`'s,
            or ``None``.

        """
        coord = self._any_meshcoord()
        if coord is None:
            result = None
        else:
            result = coord.mesh
        return result

    @property
    def location(self) -> iris.mesh.components.Location | None:
        r"""Return the mesh "location" of the cube data.

        Return the mesh "location" of the cube data, if the cube has any
        :class:`~iris.mesh.MeshCoord`,
        or ``None`` if it has none.

        Returns
        -------
        str or None
            The mesh location of the cube
            :class:`~iris.mesh.MeshCoords`
            (i.e. one of 'face' / 'edge' / 'node'), or ``None``.

        """
        coord = self._any_meshcoord()
        if coord is None:
            result = None
        else:
            result = coord.location
        return result

    def mesh_dim(self) -> int | None:
        r"""Return the cube dimension of the mesh.

        Return the cube dimension of the mesh, if the cube has any
        :class:`~iris.mesh.MeshCoord`,
        or ``None`` if it has none.

        Returns
        -------
        int or None
            The cube dimension which the cube
            :class:`~iris.mesh.MeshCoord` map to,
            or ``None``.

        """
        coord = self._any_meshcoord()
        if coord is None:
            result = None
        else:
            (result,) = self.coord_dims(coord)  # result is a 1-tuple
        return result

    def cell_measures(
        self,
        name_or_cell_measure: str | CellMeasure | None = None,
    ) -> list[CellMeasure]:
        """Return a list of cell measures in this cube fitting the given criteria.

        Parameters
        ----------
        name_or_cell_measure :
            Either

            * (a) a :attr:`standard_name`, :attr:`long_name`, or
              :attr:`var_name`. Defaults to value of `default`
              (which itself defaults to `unknown`) as defined in
              :class:`iris.common.CFVariableMixin`.

            * (b) a cell_measure instance with metadata equal to that of
              the desired cell_measures.

        Returns
        -------
        list
            List of cell measures in this cube fitting the given criteria.

        See Also
        --------
        cell_measure :
            Return a single cell_measure.

        """
        name = None

        if isinstance(name_or_cell_measure, str):
            name = name_or_cell_measure
        else:
            cell_measure = name_or_cell_measure
        cell_measures = []
        for cm, _ in self._cell_measures_and_dims:
            if name is not None:
                if cm.name() == name:
                    cell_measures.append(cm)
            elif cell_measure is not None:
                if cm == cell_measure:
                    cell_measures.append(cm)
            else:
                cell_measures.append(cm)
        return cell_measures

    def cell_measure(
        self,
        name_or_cell_measure: str | CellMeasure | None = None,
    ) -> CellMeasure:
        """Return a single cell_measure given the same arguments as :meth:`Cube.cell_measures`.

        Notes
        -----
        .. note::

            If the arguments given do not result in precisely 1 cell_measure
            being matched, an :class:`iris.exceptions.CellMeasureNotFoundError`
            is raised.

        Returns
        -------
        CellMeasure
            A single cell measure in this cube fitting the given criteria.

        See Also
        --------
        cell_measures :
            For full keyword documentation.

        """
        cell_measures = self.cell_measures(name_or_cell_measure)

        if len(cell_measures) > 1:
            msg = (
                "Expected to find exactly 1 cell_measure, but found {}. They were: {}."
            )
            msg = msg.format(
                len(cell_measures),
                ", ".join(cm.name() for cm in cell_measures),
            )
            raise iris.exceptions.CellMeasureNotFoundError(msg)
        elif len(cell_measures) == 0:
            if isinstance(name_or_cell_measure, str):
                bad_name = name_or_cell_measure
            else:
                bad_name = (name_or_cell_measure and name_or_cell_measure.name()) or ""
                if name_or_cell_measure is not None:
                    emsg = (
                        "Expected to find exactly 1 cell measure matching the given "
                        f"{bad_name!r} cell measure's metadata, but found none."
                    )
                    raise iris.exceptions.CellMeasureNotFoundError(emsg)
            msg = (
                f"Expected to find exactly 1 {bad_name!r} cell measure, but found none."
            )
            raise iris.exceptions.CellMeasureNotFoundError(msg)

        return cell_measures[0]

    def ancillary_variables(
        self,
        name_or_ancillary_variable: str | AncillaryVariable | None = None,
    ) -> list[AncillaryVariable]:
        """Return a list of ancillary variable in this cube fitting the given criteria.

        Parameters
        ----------
        name_or_ancillary_variable :
            Either

            * (a) a :attr:`standard_name`, :attr:`long_name`, or
              :attr:`var_name`. Defaults to value of `default`
              (which itself defaults to `unknown`) as defined in
              :class:`iris.common.CFVariableMixin`.

            * (b) a ancillary_variable instance with metadata equal to that of
              the desired ancillary_variables.

        Returns
        -------
        list
            List of ancillary variables in this cube fitting the given criteria.

        See Also
        --------
        ancillary_variable :
            Return a  ancillary_variable.

        """
        name = None

        if isinstance(name_or_ancillary_variable, str):
            name = name_or_ancillary_variable
        else:
            ancillary_variable = name_or_ancillary_variable
        ancillary_variables = []
        for av, _ in self._ancillary_variables_and_dims:
            if name is not None:
                if av.name() == name:
                    ancillary_variables.append(av)
            elif ancillary_variable is not None:
                if av == ancillary_variable:
                    ancillary_variables.append(av)
            else:
                ancillary_variables.append(av)
        return ancillary_variables

    def ancillary_variable(
        self,
        name_or_ancillary_variable: str | AncillaryVariable | None = None,
    ) -> AncillaryVariable:
        """Return a single ancillary_variable given the same arguments as :meth:`Cube.ancillary_variables`.

        Notes
        -----
        .. note::

            If the arguments given do not result in precisely 1
            ancillary_variable being matched, an
            :class:`iris.exceptions.AncillaryVariableNotFoundError` is raised.

        Returns
        -------
        AncillaryVariable
            A single ancillary variable in this cube fitting the given criteria.

        See Also
        --------
        ancillary_variables :
            For full keyword documentation.

        """
        ancillary_variables = self.ancillary_variables(name_or_ancillary_variable)

        if len(ancillary_variables) > 1:
            msg = (
                "Expected to find exactly 1 ancillary_variable, but found "
                "{}. They were: {}."
            )
            msg = msg.format(
                len(ancillary_variables),
                ", ".join(anc_var.name() for anc_var in ancillary_variables),
            )
            raise iris.exceptions.AncillaryVariableNotFoundError(msg)
        elif len(ancillary_variables) == 0:
            if isinstance(name_or_ancillary_variable, str):
                bad_name = name_or_ancillary_variable
            else:
                bad_name = (
                    name_or_ancillary_variable and name_or_ancillary_variable.name()
                ) or ""
                if name_or_ancillary_variable is not None:
                    emsg = (
                        "Expected to find exactly 1 ancillary_variable matching the "
                        f"given {bad_name!r} ancillary_variable's metadata, but found "
                        "none."
                    )
                    raise iris.exceptions.AncillaryVariableNotFoundError(emsg)
            msg = (
                f"Expected to find exactly 1 {bad_name!r} ancillary_variable, "
                "but found none."
            )
            raise iris.exceptions.AncillaryVariableNotFoundError(msg)

        return ancillary_variables[0]

    @property
    def cell_methods(self) -> tuple[CellMethod, ...]:
        """Tuple of :class:`iris.coords.CellMethod`.

        Tuple of :class:`iris.coords.CellMethod` representing the processing
        done on the phenomenon.

        """
        return self._metadata_manager.cell_methods

    @cell_methods.setter
    def cell_methods(
        self,
        cell_methods: Iterable[CellMethod] | None,
    ) -> None:
        if not cell_methods:
            # For backwards compatibility: Empty or null value is equivalent to ().
            cell_methods = ()
        else:
            # Can supply any iterable, which is converted (copied) to a tuple.
            cell_methods = tuple(cell_methods)
            for cell_method in cell_methods:
                # All contents should be CellMethods.  Requiring class membership is
                # somewhat non-Pythonic, but simple, and not a problem for now.
                if not isinstance(cell_method, iris.coords.CellMethod):
                    msg = (  # type: ignore[unreachable]
                        f"Cube.cell_methods assigned value includes {cell_method}, "
                        "which is not an iris.coords.CellMethod."
                    )
                    raise ValueError(msg)
        self._metadata_manager.cell_methods = cell_methods

    def core_data(self) -> np.ndarray | da.Array:
        """Retrieve the data array of this :class:`~iris.cube.Cube`.

        Retrieve the data array of this :class:`~iris.cube.Cube` in its
        current state, which will either be real or lazy.

        If this :class:`~iris.cube.Cube` has lazy data, accessing its data
        array via this method **will not** realise the data array. This means
        you can perform operations using this method that work equivalently
        on real or lazy data, and will maintain lazy data if present.

        """
        return self._data_manager.core_data()

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the data of this cube."""
        return self._data_manager.shape

    @property
    def dtype(self):
        """The data type of the values in the data array of this :class:`~iris.cube.Cube`."""
        return self._data_manager.dtype

    @property
    def ndim(self) -> int:
        """The number of dimensions in the data of this cube."""
        return self._data_manager.ndim

    def lazy_data(self) -> da.Array:
        """Return a "lazy array" representing the Cube data.

        Return a "lazy array" representing the Cube data. A lazy array
        describes an array whose data values have not been loaded into memory
        from disk.

        Accessing this method will never cause the Cube data to be loaded.
        Similarly, calling methods on, or indexing, the returned Array
        will not cause the Cube data to be loaded.

        If the Cube data have already been loaded (for example by calling
        :meth:`~iris.cube.Cube.data`), the returned Array will be a view of the
        loaded cube data represented as a lazy array object. Note that this
        does _not_ make the Cube data lazy again; the Cube data remains loaded
        in memory.

        Returns
        -------
        A lazy array, representing the Cube data.

        """
        return self._data_manager.lazy_data()

    @property
    def data(self) -> np.ndarray:
        """The :class:`numpy.ndarray` representing the multi-dimensional data of the cube.

        Notes
        -----
        .. note::

            Cubes obtained from NetCDF, PP, and FieldsFile files will only
            populate this attribute on its first use.

            To obtain the shape of the data without causing it to be loaded,
            use the Cube.shape attribute.

        Example::
            >>> fname = iris.sample_data_path('air_temp.pp')
            >>> cube = iris.load_cube(fname, 'air_temperature')
            >>> # cube.data does not yet have a value.
            ...
            >>> print(cube.shape)
            (73, 96)
            >>> # cube.data still does not have a value.
            ...
            >>> cube = cube[:10, :20]
            >>> # cube.data still does not have a value.
            ...
            >>> data = cube.data
            >>> # Only now is the data loaded.
            ...
            >>> print(data.shape)
            (10, 20)

        """
        return self._data_manager.data

    @data.setter
    def data(self, data: np.typing.ArrayLike) -> None:
        self._data_manager.data = data

    def has_lazy_data(self) -> bool:
        """Detail whether this :class:`~iris.cube.Cube` has lazy data.

        Returns
        -------
        bool

        """
        return self._data_manager.has_lazy_data()

    @property
    def dim_coords(self) -> tuple[DimCoord, ...]:
        """Return a tuple of all the dimension coordinates, ordered by dimension.

        .. note::

            The length of the returned tuple is not necessarily the same as
            :attr:`Cube.ndim` as there may be dimensions on the cube without
            dimension coordinates. It is therefore unreliable to use the
            resulting tuple to identify the dimension coordinates for a given
            dimension - instead use the :meth:`Cube.coord` method with the
            ``dimensions`` and ``dim_coords`` keyword arguments.

        """
        return tuple(
            (
                coord
                for coord, dim in sorted(
                    self._dim_coords_and_dims,
                    key=lambda co_di: (co_di[1], co_di[0].name()),
                )
            )
        )

    @property
    def aux_coords(self) -> tuple[AuxCoord | DimCoord, ...]:
        """Return a tuple of all the auxiliary coordinates, ordered by dimension(s)."""
        return tuple(
            (
                coord
                for coord, dims in sorted(
                    self._aux_coords_and_dims,
                    key=lambda co_di: (co_di[1], co_di[0].name()),
                )
            )
        )

    @property
    def derived_coords(self) -> tuple[AuxCoord, ...]:
        """Return a tuple of all the coordinates generated by the coordinate factories."""
        return tuple(
            factory.make_coord(self.coord_dims)
            for factory in sorted(
                self.aux_factories, key=lambda factory: factory.name()
            )
        )

    @property
    def aux_factories(self) -> tuple[AuxCoordFactory, ...]:
        """Return a tuple of all the coordinate factories."""
        return tuple(self._aux_factories)

    def summary(self, shorten: bool = False, name_padding: int = 35) -> str:
        """Summary of the Cube.

        String summary of the Cube with name+units, a list of dim coord names
        versus length and, optionally, a summary of all other components.

        Parameters
        ----------
        shorten :
            If set, produce a one-line summary of minimal width, showing only
            the cube name, units and dimensions.
            When not set (default), produces a full multi-line summary string.
        name_padding :
            Control the *minimum* width of the cube name + units,
            i.e. the indent of the dimension map section.

        """
        from iris._representation.cube_printout import CubePrinter

        printer = CubePrinter(self)
        summary = printer.to_string(oneline=shorten, name_padding=name_padding)
        return summary

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return "<iris 'Cube' of %s>" % self.summary(shorten=True, name_padding=1)

    def _repr_html_(self) -> str:
        from iris.experimental.representation import CubeRepresentation

        representer = CubeRepresentation(self)
        return representer.repr_html()

    # Indicate that the iter option is not available. Python will raise
    # TypeError with a useful message if a Cube is iterated over.
    __iter__ = None

    def __getitem__(self, keys) -> Cube:
        """Cube indexing has been implemented at the data level.

        Cube indexing (through use of square bracket notation) has been
        implemented at the data level. That is, the indices provided to this
        method should be aligned to the data of the cube, and thus the indices
        requested must be applicable directly to the cube.data attribute. All
        metadata will be subsequently indexed appropriately.

        """
        # turn the keys into a full slice spec (all dims)
        full_slice = iris.util._build_full_slice_given_keys(keys, self.ndim)

        def new_coord_dims(coord_):
            return [
                dimension_mapping[d]
                for d in self.coord_dims(coord_)
                if dimension_mapping[d] is not None
            ]

        def new_cell_measure_dims(cm_):
            return [
                dimension_mapping[d]
                for d in self.cell_measure_dims(cm_)
                if dimension_mapping[d] is not None
            ]

        def new_ancillary_variable_dims(av_):
            return [
                dimension_mapping[d]
                for d in self.ancillary_variable_dims(av_)
                if dimension_mapping[d] is not None
            ]

        # Fetch the data as a generic array-like object.
        cube_data = self._data_manager.core_data()

        # Index with the keys, using orthogonal slicing.
        dimension_mapping, data = iris.util._slice_data_with_keys(cube_data, keys)

        # We don't want a view of the data, so take a copy of it.
        data = deepcopy(data)

        # XXX: Slicing a single item from a masked array that is masked,
        #      results in numpy (v1.11.1) *always* returning a MaskedConstant
        #      with a dtype of float64, regardless of the original masked
        #      array dtype!
        if isinstance(data, ma.core.MaskedConstant) and data.dtype != cube_data.dtype:
            data = ma.array(data.data, mask=data.mask, dtype=cube_data.dtype)

        # Make the new cube slice
        cube = self.__class__(data)
        cube.metadata = deepcopy(self.metadata)

        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        # Slice the coords
        for coord in self.aux_coords:
            coord_keys = tuple([full_slice[dim] for dim in self.coord_dims(coord)])
            try:
                new_coord = coord[coord_keys]
            except ValueError:
                # TODO make this except more specific to catch monotonic error
                # Attempt to slice it by converting to AuxCoord first
                new_coord = iris.coords.AuxCoord.from_coord(coord)[coord_keys]
            cube.add_aux_coord(new_coord, new_coord_dims(coord))
            coord_mapping[id(coord)] = new_coord

        for coord in self.dim_coords:
            coord_keys = tuple([full_slice[dim] for dim in self.coord_dims(coord)])
            new_dims = new_coord_dims(coord)
            # Try/Catch to handle slicing that makes the points/bounds
            # non-monotonic
            try:
                new_coord = coord[coord_keys]
                if not new_dims:
                    # If the associated dimension has been sliced so the coord
                    # is a scalar move the coord to the aux_coords container
                    cube.add_aux_coord(new_coord, new_dims)
                else:
                    cube.add_dim_coord(new_coord, new_dims)
            except ValueError:
                # TODO make this except more specific to catch monotonic error
                # Attempt to slice it by converting to AuxCoord first
                new_coord = iris.coords.AuxCoord.from_coord(coord)[coord_keys]
                cube.add_aux_coord(new_coord, new_dims)
            coord_mapping[id(coord)] = new_coord

        for factory in self.aux_factories:
            cube.add_aux_factory(factory.updated(coord_mapping))

        # slice the cell measures and add them to the cube
        for cellmeasure in self.cell_measures():
            dims = self.cell_measure_dims(cellmeasure)
            cm_keys = tuple([full_slice[dim] for dim in dims])
            new_cm = cellmeasure[cm_keys]
            cube.add_cell_measure(new_cm, new_cell_measure_dims(cellmeasure))

        # slice the ancillary variables and add them to the cube
        for ancvar in self.ancillary_variables():
            dims = self.ancillary_variable_dims(ancvar)
            av_keys = tuple([full_slice[dim] for dim in dims])
            new_av = ancvar[av_keys]
            cube.add_ancillary_variable(new_av, new_ancillary_variable_dims(ancvar))

        return cube

    def subset(self, coord: AuxCoord | DimCoord) -> Cube | None:
        """Get a subset of the cube by providing the desired resultant coordinate.

        Get a subset of the cube by providing the desired resultant
        coordinate. If the coordinate provided applies to the whole cube; the
        whole cube is returned. As such, the operation is not strict.

        """
        if not isinstance(coord, iris.coords.Coord):
            raise ValueError("coord_to_extract must be a valid Coord.")

        # Get the coord to extract from the cube
        coord_to_extract = self.coord(coord)

        # If scalar, return the whole cube. Not possible to subset 1 point.
        if coord_to_extract in self.aux_coords and len(coord_to_extract.points) == 1:
            # Default to returning None
            result = None

            indices = coord_to_extract.intersect(coord, return_indices=True)

            # If there is an intersect between the two scalar coordinates;
            # return the whole cube. Else, return None.
            if len(indices):
                result = self

        else:
            if len(self.coord_dims(coord_to_extract)) > 1:
                msg = "Currently, only 1D coords can be used to subset a cube"
                raise iris.exceptions.CoordinateMultiDimError(msg)
            # Identify the dimension of the cube which this coordinate
            # references
            coord_to_extract_dim = self.coord_dims(coord_to_extract)[0]

            # Identify the indices which intersect the requested coord and
            # coord_to_extract
            coord_indices = coord_to_extract.intersect(coord, return_indices=True)

            if coord_indices.size == 0:
                # No matches found.
                return None

            # Build up a slice which spans the whole of the cube
            full_slice = [slice(None, None)] * len(self.shape)
            # Update the full slice to only extract specific indices which
            # were identified above
            full_slice[coord_to_extract_dim] = coord_indices
            result = self[tuple(full_slice)]
        return result

    def extract(self, constraint: iris.Constraint | str | None) -> Cube:
        """Filter cube by the given constraint using :meth:`iris.Constraint.extract`."""
        # Cast the constraint into a proper constraint if it is not so already
        constraint = iris._constraints.as_constraint(constraint)
        return constraint.extract(self)

    def intersection(self, *args, **kwargs) -> Cube:
        """Return the intersection of the cube with specified coordinate ranges.

        Coordinate ranges can be specified as:

        * (a) positional arguments: instances of :class:`iris.coords.CoordExtent`,
          or equivalent tuples of 3-5 items:

        * (b) keyword arguments, where the keyword name specifies the name
          of the coordinate, and the value defines the corresponding range of
          coordinate values as a tuple. The tuple must contain two, three, or
          four items, corresponding to `(minimum, maximum, min_inclusive,
          max_inclusive)` as defined above.

        Parameters
        ----------
        coord :
            Either a :class:`iris.coords.Coord`, or coordinate name
            (as defined in :meth:`iris.cube.Cube.coords()`).
        minimum :
            The minimum value of the range to select.
        maximum :
            The maximum value of the range to select.
        min_inclusive :
            If True, coordinate values equal to `minimum` will be included
            in the selection. Default is True.
        max_inclusive :
            If True, coordinate values equal to `maximum` will be included
            in the selection. Default is True.
        ignore_bounds : optional
            Intersect based on points only. Default False.
        threshold : optional
            Minimum proportion of a bounded cell that must overlap with the
            specified range. Default 0.

        Notes
        -----
        .. note::

            For ranges defined over "circular" coordinates (i.e. those
            where the `units` attribute has a modulus defined) the cube
            will be "rolled" to fit where necessary.  When requesting a
            range that covers the entire modulus, a split cell will
            preferentially be placed at the ``minimum`` end.

        Warnings
        --------
        Currently this routine only works with "circular"
        coordinates (as defined in the previous note.)

        For example::

            >>> import iris
            >>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
            >>> print(cube.coord('longitude').points[::10])
            [   0.           37.49999237   74.99998474  112.49996948  \
149.99996948
            187.49995422  224.99993896  262.49993896  299.99993896  \
337.49990845]
            >>> subset = cube.intersection(longitude=(30, 50))
            >>> print(subset.coord('longitude').points)
            [ 33.74999237  37.49999237  41.24998856  44.99998856  48.74998856]
            >>> subset = cube.intersection(longitude=(-10, 10))
            >>> print(subset.coord('longitude').points)
            [-7.50012207 -3.75012207  0.          3.75        7.5       ]

        Returns
        -------
        :class:`~iris.cube.Cube`
            A new :class:`~iris.cube.Cube` giving the subset of the cube
            which intersects with the requested coordinate intervals.

        """
        result = self
        ignore_bounds = kwargs.pop("ignore_bounds", False)
        threshold = kwargs.pop("threshold", 0)
        for arg in args:
            result = result._intersect(
                *arg, ignore_bounds=ignore_bounds, threshold=threshold
            )  # type: ignore[misc]
        for name, value in kwargs.items():
            result = result._intersect(
                name, *value, ignore_bounds=ignore_bounds, threshold=threshold
            )  # type: ignore[misc]
        return result

    def _intersect(
        self,
        name_or_coord: str
        | DimCoord
        | AuxCoord
        | AuxCoordFactory
        | CoordMetadata
        | None,
        minimum: float | int,
        maximum: float | int,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        ignore_bounds: bool = False,
        threshold=0,
    ) -> Cube:
        coord = self.coord(name_or_coord)
        if coord.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)
        if coord.nbounds not in (0, 2):
            raise ValueError("expected 0 or 2 bound values per cell")
        if minimum > maximum:
            raise ValueError("minimum greater than maximum")
        modulus = coord.units.modulus
        if modulus is None:
            raise ValueError("coordinate units with no modulus are not yet supported")
        subsets, points, bounds = self._intersect_modulus(
            coord,
            minimum,
            maximum,
            min_inclusive,
            max_inclusive,
            ignore_bounds,
            threshold,
        )

        # By this point we have either one or two subsets along the relevant
        # dimension. If it's just one subset (which might be a slice or an
        # unordered collection of indices) we can simply index the cube
        # and we're done. If it's two subsets we need to stitch the two
        # pieces together.
        # subsets provides a way of slicing the coordinates to ensure that
        # they remain contiguous.  In doing so, this can mean
        # transforming the data (this stitching together of two separate
        # pieces).
        def make_chunk(key):
            chunk = self[key_tuple_prefix + (key,)]
            chunk_coord = chunk.coord(coord)
            chunk_coord.points = points[(key,)]
            if chunk_coord.has_bounds():
                chunk_coord.bounds = bounds[(key,)]
            return chunk

        (dim,) = self.coord_dims(coord)
        key_tuple_prefix = (slice(None),) * dim
        chunks = [make_chunk(key) for key in subsets]
        if len(chunks) == 1:
            result = chunks[0]
        else:
            chunk_data = [chunk.core_data() for chunk in chunks]
            data = _lazy.concatenate(chunk_data, axis=dim)
            result = iris.cube.Cube(data)
            result.metadata = deepcopy(self.metadata)

            # Record a mapping from old coordinate IDs to new coordinates,
            # for subsequent use in creating updated aux_factories.
            coord_mapping = {}

            def create_coords(src_coords, add_coord):
                # Add copies of the source coordinates, selecting
                # the appropriate subsets out of coordinates which
                # share the intersection dimension.
                preserve_circular = (
                    min_inclusive
                    and max_inclusive
                    and abs(maximum - minimum) == modulus
                )
                for src_coord in src_coords:
                    dims = self.coord_dims(src_coord)
                    if dim in dims:
                        dim_within_coord = dims.index(dim)
                        points = np.concatenate(
                            [chunk.coord(src_coord).points for chunk in chunks],
                            dim_within_coord,
                        )
                        if src_coord.has_bounds():
                            bounds = np.concatenate(
                                [chunk.coord(src_coord).bounds for chunk in chunks],
                                dim_within_coord,
                            )
                        else:
                            bounds = None
                        result_coord = src_coord.copy(points=points, bounds=bounds)

                        circular = getattr(result_coord, "circular", False)
                        if circular and not preserve_circular:
                            result_coord.circular = False
                    else:
                        result_coord = src_coord.copy()
                    add_coord(result_coord, dims)
                    coord_mapping[id(src_coord)] = result_coord

            def create_metadata(src_metadatas, add_metadata, get_metadata):
                for src_metadata in src_metadatas:
                    dims = src_metadata.cube_dims(self)
                    if dim in dims:
                        dim_within_coord = dims.index(dim)
                        data = np.concatenate(
                            [
                                get_metadata(chunk, src_metadata.name()).core_data()
                                for chunk in chunks
                            ],
                            dim_within_coord,
                        )
                        result_coord = src_metadata.copy(values=data)
                    else:
                        result_coord = src_metadata.copy()
                    add_metadata(result_coord, dims)

            create_coords(self.dim_coords, result.add_dim_coord)
            create_coords(self.aux_coords, result.add_aux_coord)
            create_metadata(
                self.cell_measures(), result.add_cell_measure, Cube.cell_measure
            )
            create_metadata(
                self.ancillary_variables(),
                result.add_ancillary_variable,
                Cube.ancillary_variable,
            )
            for factory in self.aux_factories:
                result.add_aux_factory(factory.updated(coord_mapping))
        return result

    def _intersect_derive_subset(
        self,
        coord: AuxCoord | DimCoord,
        points: np.ndarray,
        bounds: np.ndarray,
        inside_indices: np.ndarray,
    ) -> list[slice]:
        # Return the subsets, i.e. the means to allow the slicing of
        # coordinates to ensure that they remain contiguous.
        modulus = coord.units.modulus
        delta = coord.points[inside_indices] - points[inside_indices]
        step = np.rint(np.diff(delta) / modulus)
        non_zero_step_indices = np.nonzero(step)[0]

        def dim_coord_subset():
            """Derive the subset for dimension coordinates.

            Ensure that we do not wrap if blocks are at the very edge.  That
            is, if the very edge is wrapped and corresponds to base + period,
            stop this unnecessary wraparound.

            """
            # A contiguous block at the start and another at the end.
            # (NB. We can't have more than two blocks because we've already
            # restricted the coordinate's range to its modulus).
            end_of_first_chunk = non_zero_step_indices[0]
            index_of_second_chunk = inside_indices[end_of_first_chunk + 1]
            final_index = points.size - 1

            # Condition1: The two blocks don't themselves wrap
            #             (inside_indices is contiguous).
            # Condition2: Are we chunked at either extreme edge.
            edge_wrap = (
                index_of_second_chunk == inside_indices[end_of_first_chunk] + 1
            ) and index_of_second_chunk in (final_index, 1)
            subsets = None
            if edge_wrap:
                # Increasing coord
                if coord.points[-1] > coord.points[0]:
                    index_end = -1
                    index_start = 0
                # Decreasing coord
                else:
                    index_end = 0
                    index_start = -1

                # Unwrap points and bounds (if present and equal base + period)
                if bounds is not None:
                    edge_equal_base_period = np.isclose(
                        coord.bounds[index_end, index_end],
                        coord.bounds[index_start, index_start] + modulus,
                    )
                    if edge_equal_base_period:
                        bounds[index_end, :] = coord.bounds[index_end, :]
                else:
                    edge_equal_base_period = np.isclose(
                        coord.points[index_end],
                        coord.points[index_start] + modulus,
                    )
                if edge_equal_base_period:
                    points[index_end] = coord.points[index_end]
                    subsets = [slice(inside_indices[0], inside_indices[-1] + 1)]

            # Either no edge wrap or edge wrap != base + period
            # i.e. derive subset without alteration
            if subsets is None:
                subsets = [
                    slice(index_of_second_chunk, None),
                    slice(None, inside_indices[end_of_first_chunk] + 1),
                ]

            return subsets

        if isinstance(coord, iris.coords.DimCoord):
            if non_zero_step_indices.size:
                subsets = dim_coord_subset()
            else:
                # A single, contiguous block.
                subsets = [slice(inside_indices[0], inside_indices[-1] + 1)]
        else:
            # An AuxCoord could have its values in an arbitrary
            # order, and hence a range of values can select an
            # arbitrary subset. Also, we want to preserve the order
            # from the original AuxCoord. So we just use the indices
            # directly.
            subsets = [inside_indices]
        return subsets

    def _intersect_modulus(
        self,
        coord: AuxCoord | DimCoord,
        minimum: float | int,
        maximum: float | int,
        min_inclusive: bool,
        max_inclusive: bool,
        ignore_bounds: bool,
        threshold: float | int,
    ) -> tuple[list[slice], np.ndarray, np.ndarray]:
        modulus = coord.units.modulus
        if maximum > minimum + modulus:
            raise ValueError("requested range greater than coordinate's unit's modulus")
        if coord.has_bounds():
            values = coord.bounds
        else:
            ignore_bounds = True
            values = coord.points
        if values.max() > values.min() + modulus:
            raise ValueError(
                "coordinate's range greater than coordinate's unit's modulus"
            )
        min_comp = np.less_equal if min_inclusive else np.less
        max_comp = np.less_equal if max_inclusive else np.less

        if ignore_bounds:
            points = wrap_lons(coord.points, minimum, modulus)
            bounds = coord.bounds
            if bounds is not None:
                # To avoid splitting any cells (by wrapping only one of its
                # bounds), apply exactly the same wrapping as the points.
                # Note that the offsets should be exact multiples of the
                # modulus, but may initially be slightly off and need rounding.
                wrap_offset = points - coord.points
                wrap_offset = np.round(wrap_offset / modulus) * modulus
                bounds = coord.bounds + wrap_offset[:, np.newaxis]

            # Check points only
            (inside_indices,) = np.where(
                np.logical_and(min_comp(minimum, points), max_comp(points, maximum))
            )

        else:
            # Set up slices to account for ascending/descending bounds
            if coord.bounds[0, 0] < coord.bounds[0, 1]:
                ilower = (slice(None), 0)
                iupper = (slice(None), 1)
            else:
                ilower = (slice(None), 1)
                iupper = (slice(None), 0)

            # Initially wrap such that upper bounds are in [min, min + modulus]
            # As with the ignore_bounds case, need to round to modulus due to
            # floating point precision
            upper = wrap_lons(coord.bounds[iupper], minimum, modulus)
            wrap_offset = upper - coord.bounds[iupper]
            wrap_offset = np.round(wrap_offset / modulus) * modulus
            lower = coord.bounds[ilower] + wrap_offset

            # Scale threshold for each bound
            thresholds = (upper - lower) * threshold

            # For a range that covers the whole modulus, there may be a
            # cell that is "split" and could appear at either side of
            # the range.  Choose lower, unless there is not enough overlap.
            if minimum + modulus == maximum and threshold == 0:
                # Special case: overlapping in a single point
                # (ie `minimum` itself) is always unintuitive
                is_split = np.isclose(upper, minimum)
            else:
                is_split = upper - minimum < thresholds
            wrap_offset += is_split * modulus

            # Apply wrapping
            points = coord.points + wrap_offset
            bounds = coord.bounds + wrap_offset[:, np.newaxis]

            # Interval [min, max] intersects [a, b] iff min <= b and a <= max
            # (or < for non-inclusive min/max respectively).
            # In this case, its length is L = min(max, b) - max(min, a)
            upper = bounds[iupper]
            lower = bounds[ilower]
            overlap = np.where(
                np.logical_and(min_comp(minimum, upper), max_comp(lower, maximum)),
                np.minimum(maximum, upper) - np.maximum(minimum, lower),
                np.nan,
            )
            (inside_indices,) = np.where(overlap >= thresholds)

        # Determine the subsets
        subsets = self._intersect_derive_subset(coord, points, bounds, inside_indices)
        return subsets, points, bounds

    def _as_list_of_coords(self, names_or_coords) -> list[AuxCoord | DimCoord]:
        """Convert a name, coord, or list of names/coords to a list of coords."""
        # If not iterable, convert to list of a single item
        if _is_single_item(names_or_coords):
            names_or_coords = [names_or_coords]

        coords = []
        for name_or_coord in names_or_coords:
            if isinstance(name_or_coord, str) or isinstance(
                name_or_coord, (iris.coords.DimCoord, iris.coords.AuxCoord)
            ):
                coords.append(self.coord(name_or_coord))
            else:
                # Don't know how to handle this type
                msg = (
                    "Don't know how to handle coordinate of type %s. "
                    "Ensure all coordinates are of type str "
                    "or iris.coords.Coord."
                ) % (type(name_or_coord),)
                raise TypeError(msg)
        return coords

    def slices_over(
        self,
        ref_to_slice: str
        | AuxCoord
        | DimCoord
        | int
        | Iterable[str | AuxCoord | DimCoord | int],
    ) -> Iterable[Cube]:
        """Return an iterator of all subcubes.

        Return an iterator of all subcubes along a given coordinate or
        dimension index, or multiple of these.

        Parameters
        ----------
        ref_to_slice :
            Determines which dimensions will be iterated along (i.e. the
            dimensions that are not returned in the subcubes).
            A mix of input types can also be provided.

        Returns
        -------
        An iterator of subcubes.

        Examples
        --------
        For example, for a cube with dimensions `realization`, `time`, `latitude` and
        `longitude`:

        >>> fname = iris.sample_data_path('GloSea4', 'ensemble_01[01].pp')
        >>> cube = iris.load_cube(fname, 'surface_temperature')
        >>> print(cube.summary(shorten=True))
        surface_temperature / (K)           (realization: 2; time: 6; latitude: 145; longitude: 192)

        To get all 12x2D longitude/latitude subcubes:

        >>> for sub_cube in cube.slices_over(['realization', 'time']):
        ...     print(sub_cube.summary(shorten=True))
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)
        surface_temperature / (K)           (latitude: 145; longitude: 192)

        To get realizations as 2x3D separate subcubes, using the `realization` dimension index:

        >>> for sub_cube in cube.slices_over(0):
        ...     print(sub_cube.summary(shorten=True))
        surface_temperature / (K)           (time: 6; latitude: 145; longitude: 192)
        surface_temperature / (K)           (time: 6; latitude: 145; longitude: 192)

        Notes
        -----
        .. note::

            The order of dimension references to slice along does not affect
            the order of returned items in the iterator; instead the ordering
            is based on the fastest-changing dimension.

        See Also
        --------
        iris.cube.Cube.slices :
            Return an iterator of all subcubes given the coordinates or dimension indices.

        """  # noqa: D214, D406, D407, D410, D411
        # Required to handle a mix between types.
        if _is_single_item(ref_to_slice):
            ref_to_slice = [ref_to_slice]

        slice_dims: set[int] = set()
        for ref in ref_to_slice:  # type: ignore[union-attr]
            try:
                (coord,) = self._as_list_of_coords(ref)
            except TypeError:
                dim = int(ref)  # type: ignore[arg-type]
                if dim < 0 or dim > self.ndim:
                    msg = (
                        "Requested an iterator over a dimension ({}) "
                        "which does not exist.".format(dim)
                    )
                    raise ValueError(msg)
                # Convert coord index to a single-element list to prevent a
                # TypeError when `slice_dims.update` is called with it.
                dims: tuple[int, ...] = (dim,)
            else:
                dims = self.coord_dims(coord)
            slice_dims.update(dims)

        all_dims = set(range(self.ndim))
        opposite_dims = list(all_dims - slice_dims)
        return self.slices(opposite_dims, ordered=False)

    def slices(
        self,
        ref_to_slice: str
        | AuxCoord
        | DimCoord
        | int
        | Iterable[str | AuxCoord | DimCoord | int],
        ordered: bool = True,
    ) -> Iterator[Cube]:
        """Return an iterator of all subcubes given the coordinates or dimension indices.

        Return an iterator of all subcubes given the coordinates or dimension
        indices desired to be present in each subcube.

        Parameters
        ----------
        ref_to_slice :
            Determines which dimensions will be returned in the subcubes (i.e.
            the dimensions that are not iterated over).
            A mix of input types can also be provided. They must all be
            orthogonal (i.e. point to different dimensions).
        ordered :
            If True, subcube dimensions are ordered to match the dimension order
            in `ref_to_slice`. If False, the order will follow that of
            the source cube.

        Returns
        -------
        An iterator of subcubes.

        Examples
        --------
        For example, for a cube with dimensions `realization`, `time`, `latitude` and
        `longitude`:

        >>> fname = iris.sample_data_path('GloSea4', 'ensemble_01[01].pp')
        >>> cube = iris.load_cube(fname, 'surface_temperature')
        >>> print(cube.summary(shorten=True))
        surface_temperature / (K)           (realization: 2; time: 6; latitude: 145; longitude: 192)

        To get all 12x2D longitude/latitude subcubes:

        >>> for sub_cube in cube.slices(['longitude', 'latitude']):
        ...     print(sub_cube.summary(shorten=True))
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)
        surface_temperature / (K)           (longitude: 192; latitude: 145)


        .. warning::
            Note that the dimension order returned in the sub_cubes matches the order specified
            in the ``cube.slices`` call, *not* the order of the dimensions in the original cube.

        To get all realizations as 2x3D separate subcubes, using the `time`, `latitude`
        and `longitude` dimensions' indices:

        >>> for sub_cube in cube.slices([1, 2, 3]):
        ...     print(sub_cube.summary(shorten=True))
        surface_temperature / (K)           (time: 6; latitude: 145; longitude: 192)
        surface_temperature / (K)           (time: 6; latitude: 145; longitude: 192)

        See Also
        --------
        iris.cube.Cube.slices_over :
            Return an iterator of all subcubes along a given coordinate or
            dimension index.

        """  # noqa: D214, D406, D407, D410, D411
        if not isinstance(ordered, bool):
            raise TypeError("'ordered' argument to slices must be boolean.")

        # Required to handle a mix between types
        if _is_single_item(ref_to_slice):
            ref_to_slice = [ref_to_slice]

        dim_to_slice: list[int] = []
        for ref in ref_to_slice:  # type: ignore[union-attr]
            try:
                # attempt to handle as coordinate
                coord = self._as_list_of_coords(ref)[0]
                dims = self.coord_dims(coord)
                if not dims:
                    msg = (
                        "Requested an iterator over a coordinate ({}) "
                        "which does not describe a dimension."
                    )
                    msg = msg.format(coord.name())
                    raise ValueError(msg)
                dim_to_slice.extend(dims)

            except TypeError:
                try:
                    # attempt to handle as dimension index
                    dim = int(ref)  # type: ignore[arg-type]
                except ValueError:
                    raise ValueError(
                        "{} Incompatible type {} for slicing".format(ref, type(ref))
                    )
                if dim < 0 or dim > self.ndim:
                    msg = (
                        "Requested an iterator over a dimension ({}) "
                        "which does not exist.".format(dim)
                    )
                    raise ValueError(msg)
                dim_to_slice.append(dim)

        if len(set(dim_to_slice)) != len(dim_to_slice):
            msg = "The requested coordinates are not orthogonal."
            raise ValueError(msg)

        # Create a list with of the shape of our data
        dims_index = list(self.shape)

        # Set the dimensions which have been requested to length 1
        for d in dim_to_slice:
            dims_index[d] = 1

        return _SliceIterator(self, dims_index, dim_to_slice, ordered)

    def transpose(self, new_order: list[int] | None = None) -> None:
        """Re-order the data dimensions of the cube in-place.

        Parameters
        ----------
        new_order :
            By default, reverse the dimensions, otherwise permute the
            axes according to the values given.

        Notes
        -----
        .. note:: If defined, new_order must span all of the data dimensions.

        Examples
        --------
        ::

            # put the second dimension first, followed by the third dimension,
            # and finally put the first dimension third::

                >>> cube.transpose([1, 2, 0])

        """
        if new_order is None:
            new_order = np.arange(self.ndim)[::-1]

        # `new_order` must be an iterable for checking with `self.ndim`.
        # Dask transpose only supports lists, so ensure `new_order` is
        # always a list.
        new_order = list(new_order)

        if len(new_order) != self.ndim:
            raise ValueError("Incorrect number of dimensions.")

        # Transpose the data payload.
        dm = self._data_manager
        data = dm.core_data().transpose(new_order)
        self._data_manager = DataManager(data)

        dim_mapping = {src: dest for dest, src in enumerate(new_order)}

        # Remap all cube dimensional metadata (dim and aux coords and cell
        # measures).
        def remap_cube_metadata(metadata_and_dims):
            metadata, dims = metadata_and_dims
            if isinstance(dims, Iterable):
                dims = tuple(dim_mapping[dim] for dim in dims)
            else:
                dims = dim_mapping[dims]
            return metadata, dims

        self._dim_coords_and_dims = list(
            map(remap_cube_metadata, self._dim_coords_and_dims)
        )
        self._aux_coords_and_dims = list(
            map(remap_cube_metadata, self._aux_coords_and_dims)
        )
        self._cell_measures_and_dims = list(
            map(remap_cube_metadata, self._cell_measures_and_dims)
        )
        self._ancillary_variables_and_dims = list(
            map(remap_cube_metadata, self._ancillary_variables_and_dims)
        )

    def xml(
        self,
        checksum: bool = False,
        order: bool = True,
        byteorder: bool = True,
    ) -> str:
        """Return a fully valid CubeML string representation of the Cube."""
        doc = Document()

        cube_xml_element = self._xml_element(
            doc, checksum=checksum, order=order, byteorder=byteorder
        )
        cube_xml_element.setAttribute("xmlns", XML_NAMESPACE_URI)
        doc.appendChild(cube_xml_element)

        # Print our newly created XML
        doc = self._sort_xml_attrs(doc)
        return doc.toprettyxml(indent="  ")

    def _xml_element(self, doc, checksum=False, order=True, byteorder=True):
        cube_xml_element = doc.createElement("cube")

        if self.standard_name:
            cube_xml_element.setAttribute("standard_name", self.standard_name)
        if self.long_name:
            cube_xml_element.setAttribute("long_name", self.long_name)
        if self.var_name:
            cube_xml_element.setAttribute("var_name", self.var_name)
        cube_xml_element.setAttribute("units", str(self.units))
        cube_xml_element.setAttribute("dtype", self.dtype.name)

        if self.attributes:
            attributes_element = doc.createElement("attributes")
            for name in sorted(self.attributes.keys()):
                attribute_element = doc.createElement("attribute")
                attribute_element.setAttribute("name", name)

                value = self.attributes[name]
                # Strict check because we don't want namedtuples.
                if type(value) in (list, tuple):
                    delimiter = "[]" if isinstance(value, list) else "()"
                    value = ", ".join(
                        ("'%s'" if isinstance(item, str) else "%s") % (item,)
                        for item in value
                    )
                    value = delimiter[0] + value + delimiter[1]
                else:
                    value = str(value)

                attribute_element.setAttribute("value", value)
                attributes_element.appendChild(attribute_element)

            cube_xml_element.appendChild(attributes_element)

        def dimmeta_xml_element(element, typename, dimscall):
            # Make an inner xml element for a cube DimensionalMetadata element, with a
            # 'datadims' property showing how it maps to the parent cube dims.
            xml_element = doc.createElement(typename)
            dims = list(dimscall(element))
            if dims:
                xml_element.setAttribute("datadims", repr(dims))
            xml_element.appendChild(element.xml_element(doc))
            return xml_element

        coords_xml_element = doc.createElement("coords")
        for coord in sorted(self.coords(), key=lambda coord: coord.name()):
            # make a "cube coordinate" element which holds the dimensions (if
            # appropriate) which itself will have a sub-element of the
            # coordinate instance itself.
            coords_xml_element.appendChild(
                dimmeta_xml_element(coord, "coord", self.coord_dims)
            )
        cube_xml_element.appendChild(coords_xml_element)

        # cell methods (no sorting!)
        cell_methods_xml_element = doc.createElement("cellMethods")
        for cm in self.cell_methods:
            cell_method_xml_element = cm.xml_element(doc)
            cell_methods_xml_element.appendChild(cell_method_xml_element)
        cube_xml_element.appendChild(cell_methods_xml_element)

        # cell measures
        cell_measures = sorted(self.cell_measures(), key=lambda cm: cm.name())
        if cell_measures:
            # This one is an optional subelement.
            cms_xml_element = doc.createElement("cellMeasures")
            for cm in cell_measures:
                cms_xml_element.appendChild(
                    dimmeta_xml_element(cm, "cell-measure", self.cell_measure_dims)
                )
            cube_xml_element.appendChild(cms_xml_element)

        # ancillary variables
        ancils = sorted(self.ancillary_variables(), key=lambda anc: anc.name())
        if ancils:
            # This one is an optional subelement.
            ancs_xml_element = doc.createElement("ancillaryVariables")
            for anc in ancils:
                ancs_xml_element.appendChild(
                    dimmeta_xml_element(
                        anc, "ancillary-var", self.ancillary_variable_dims
                    )
                )
            cube_xml_element.appendChild(ancs_xml_element)

        # data
        data_xml_element = doc.createElement("data")
        data_xml_element.setAttribute("shape", str(self.shape))

        # NB. Getting a checksum triggers any deferred loading,
        # in which case it also has the side-effect of forcing the
        # byte order to be native.
        if checksum:
            data = self.data

            # Ensure consistent memory layout for checksums.
            def normalise(data):
                data = np.ascontiguousarray(data)
                if data.dtype.newbyteorder("<") != data.dtype:
                    data = data.byteswap(False)
                    data.dtype = data.dtype.newbyteorder("<")
                return data

            if ma.isMaskedArray(data):
                # Fill in masked values to avoid the checksum being
                # sensitive to unused numbers. Use a fixed value so
                # a change in fill_value doesn't affect the
                # checksum.
                crc = "0x%08x" % (zlib.crc32(normalise(data.filled(0))) & 0xFFFFFFFF,)
                data_xml_element.setAttribute("checksum", crc)
                if ma.is_masked(data):
                    crc = "0x%08x" % (zlib.crc32(normalise(data.mask)) & 0xFFFFFFFF,)
                else:
                    crc = "no-masked-elements"
                data_xml_element.setAttribute("mask_checksum", crc)
            else:
                crc = "0x%08x" % (zlib.crc32(normalise(data)) & 0xFFFFFFFF,)
                data_xml_element.setAttribute("checksum", crc)
        elif self.has_lazy_data():
            data_xml_element.setAttribute("state", "deferred")
        else:
            data_xml_element.setAttribute("state", "loaded")

        # Add the dtype, and also the array and mask orders if the
        # data is loaded.
        if not self.has_lazy_data():
            data = self.data
            dtype = data.dtype

            def _order(array):
                order = ""
                if array.flags["C_CONTIGUOUS"]:
                    order = "C"
                elif array.flags["F_CONTIGUOUS"]:
                    order = "F"
                return order

            if order:
                data_xml_element.setAttribute("order", _order(data))

            # NB. dtype.byteorder can return '=', which is bad for
            # cross-platform consistency - so we use dtype.str
            # instead.
            if byteorder:
                array_byteorder = {">": "big", "<": "little"}.get(dtype.str[0])
                if array_byteorder is not None:
                    data_xml_element.setAttribute("byteorder", array_byteorder)

            if order and ma.isMaskedArray(data):
                data_xml_element.setAttribute("mask_order", _order(data.mask))
        else:
            dtype = self.lazy_data().dtype
        data_xml_element.setAttribute("dtype", dtype.name)

        cube_xml_element.appendChild(data_xml_element)

        return cube_xml_element

    def copy(self, data: np.typing.ArrayLike | None = None) -> Cube:
        """Return a deep copy of this cube.

        Parameters
        ----------
        data :
            Replace the data of the cube copy with provided data payload.

        Returns
        -------
        A copy instance of the :class:`Cube`.

        """
        memo: dict[int, Any] = {}
        cube = self._deepcopy(memo, data=data)
        return cube

    def __copy__(self):
        """Shallow copying is disallowed for Cubes."""
        raise copy.Error("Cube shallow-copy not allowed. Use deepcopy() or Cube.copy()")

    def __deepcopy__(self, memo):
        return self._deepcopy(memo)

    def _deepcopy(self, memo, data=None):
        dm = self._data_manager.copy(data=data)

        new_dim_coords_and_dims = deepcopy(self._dim_coords_and_dims, memo)
        new_aux_coords_and_dims = deepcopy(self._aux_coords_and_dims, memo)
        new_cell_measures_and_dims = deepcopy(self._cell_measures_and_dims, memo)
        new_ancillary_variables_and_dims = deepcopy(
            self._ancillary_variables_and_dims, memo
        )

        # Record a mapping from old coordinate IDs to new coordinates,
        # for subsequent use in creating updated aux_factories.
        coord_mapping = {}

        for old_pair, new_pair in zip(
            self._dim_coords_and_dims, new_dim_coords_and_dims
        ):
            coord_mapping[id(old_pair[0])] = new_pair[0]

        for old_pair, new_pair in zip(
            self._aux_coords_and_dims, new_aux_coords_and_dims
        ):
            coord_mapping[id(old_pair[0])] = new_pair[0]

        new_cube = Cube(
            dm.core_data(),
            dim_coords_and_dims=new_dim_coords_and_dims,
            aux_coords_and_dims=new_aux_coords_and_dims,
            cell_measures_and_dims=new_cell_measures_and_dims,
            ancillary_variables_and_dims=new_ancillary_variables_and_dims,
        )

        new_cube.metadata = deepcopy(self.metadata, memo)

        for factory in self.aux_factories:
            new_cube.add_aux_factory(factory.updated(coord_mapping))

        return new_cube

    # START OPERATOR OVERLOADS
    def __eq__(self, other):
        if other is self:
            return True

        result = NotImplemented

        if isinstance(other, Cube):
            result = self.metadata == other.metadata

            # having checked the metadata, now check the coordinates
            if result:
                coord_compares = iris.analysis._dimensional_metadata_comparison(
                    self, other
                )
                # if there are any coordinates which are not equal
                result = not (
                    coord_compares["not_equal"]
                    or coord_compares["non_equal_data_dimension"]
                )

            if result:
                cm_compares = iris.analysis._dimensional_metadata_comparison(
                    self, other, object_get=Cube.cell_measures
                )
                # if there are any cell measures which are not equal
                result = not (
                    cm_compares["not_equal"] or cm_compares["non_equal_data_dimension"]
                )

            if result:
                av_compares = iris.analysis._dimensional_metadata_comparison(
                    self, other, object_get=Cube.ancillary_variables
                )
                # if there are any ancillary variables which are not equal
                result = not (
                    av_compares["not_equal"] or av_compares["non_equal_data_dimension"]
                )

            # Having checked everything else, check approximate data equality.
            if result:
                # TODO: why do we use allclose() here, but strict equality in
                #  _DimensionalMetadata (via util.array_equal())?
                result = bool(
                    np.allclose(
                        self.core_data(),
                        other.core_data(),
                        equal_nan=True,
                    )
                )
        return result

    # Must supply __ne__, Python does not defer to __eq__ for negative equality
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    # Must supply __hash__ as Python 3 does not enable it if __eq__ is defined.
    # NOTE: Violates "objects which compare equal must have the same hash".
    # We ought to remove this, as equality of two cube can *change*, so they
    # really should not be hashable.
    # However, current code needs it, e.g. so we can put them in sets.
    # Fixing it will require changing those uses.  See #962 and #1772.
    def __hash__(self):
        return hash(id(self))

    def __add__(self, other):
        return iris.analysis.maths.add(self, other)

    def __iadd__(self, other):
        return iris.analysis.maths.add(self, other, in_place=True)

    __radd__ = __add__

    def __sub__(self, other):
        return iris.analysis.maths.subtract(self, other)

    def __isub__(self, other):
        return iris.analysis.maths.subtract(self, other, in_place=True)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        return iris.analysis.maths.multiply(self, other)

    def __imul__(self, other):
        return iris.analysis.maths.multiply(self, other, in_place=True)

    __rmul__ = __mul__

    __div__ = iris.analysis.maths.divide

    def __idiv__(self, other):
        return iris.analysis.maths.divide(self, other, in_place=True)

    def __rdiv__(self, other):
        data = 1 / self.core_data()
        reciprocal = self.copy(data=data)
        reciprocal.units = reciprocal.units**-1
        return iris.analysis.maths.multiply(reciprocal, other)

    __truediv__ = __div__

    __itruediv__ = __idiv__

    __rtruediv__ = __rdiv__

    __pow__ = iris.analysis.maths.exponentiate

    def __neg__(self):
        return self.copy(data=-self.core_data())

    # END OPERATOR OVERLOADS

    def collapsed(
        self,
        coords: str | AuxCoord | DimCoord | Iterable[str | AuxCoord | DimCoord],
        aggregator: iris.analysis.Aggregator,
        **kwargs,
    ) -> Cube:
        """Collapse one or more dimensions over the cube given the coordinate/s and an aggregation.

        Examples of aggregations that may be used include
        :data:`~iris.analysis.COUNT` and :data:`~iris.analysis.MAX`.

        Weighted aggregations (:class:`iris.analysis.WeightedAggregator`) may
        also be supplied. These include :data:`~iris.analysis.MEAN` and
        sum :data:`~iris.analysis.SUM`.

        Weighted aggregations support an optional *weights* keyword argument.
        If set, this can be supplied as an array, cube, or (names of)
        :meth:`~iris.cube.Cube.coords`, :meth:`~iris.cube.Cube.cell_measures`,
        or :meth:`~iris.cube.Cube.ancillary_variables`. In all cases, the
        weights should be 1d (for collapsing over a 1d coordinate) or match the
        shape of the cube. When weights are not given as arrays, units are
        correctly handled for weighted sums, i.e., the original unit of the
        cube is multiplied by the units of the weights. Values for
        latitude-longitude area weights may be calculated using
        :func:`iris.analysis.cartography.area_weights`.

        Some Iris aggregators support "lazy" evaluation, meaning that
        cubes resulting from this method may represent data arrays which are
        not computed until the data is requested (e.g. via ``cube.data`` or
        ``iris.save``). If lazy evaluation exists for the given aggregator
        it will be used wherever possible when this cube's data is itself
        a deferred array.

        Parameters
        ----------
        coords :
            Coordinate names/coordinates over which the cube should be
            collapsed.
        aggregator :
            Aggregator to be applied for collapse operation.
        **kwargs :
            Aggregation function keyword arguments.

        Returns
        -------
        Collapsed cube.

        Examples
        --------
            >>> import iris
            >>> import iris.analysis
            >>> path = iris.sample_data_path('ostia_monthly.nc')
            >>> cube = iris.load_cube(path)
            >>> new_cube = cube.collapsed('longitude', iris.analysis.MEAN)
            >>> print(new_cube)
            surface_temperature / (K)           (time: 54; latitude: 18)
                Dimension coordinates:
                    time                             x             -
                    latitude                         -             x
                Auxiliary coordinates:
                    forecast_reference_time          x             -
                Scalar coordinates:
                    forecast_period             0 hours
                    longitude                   \
180.0 degrees, bound=(0.0, 360.0) degrees
                Cell methods:
                    0                           month: year: mean
                    1                           longitude: mean
                Attributes:
                    Conventions                 'CF-1.5'
                    STASH                       m01s00i024

        Notes
        -----
        .. note::

            Some aggregations are not commutative and hence the order of
            processing is important i.e.::

                tmp = cube.collapsed('realization', iris.analysis.VARIANCE)
                result = tmp.collapsed('height', iris.analysis.VARIANCE)

            is not necessarily the same result as::

                tmp = cube.collapsed('height', iris.analysis.VARIANCE)
                result2 = tmp.collapsed('realization', iris.analysis.VARIANCE)

            Conversely operations which operate on more than one coordinate
            at the same time are commutative as they are combined internally
            into a single operation. Hence the order of the coordinates
            supplied in the list does not matter::

                cube.collapsed(['longitude', 'latitude'],
                               iris.analysis.VARIANCE)

            is the same (apart from the logically equivalent cell methods that
            may be created etc.) as::

                cube.collapsed(['latitude', 'longitude'],
                               iris.analysis.VARIANCE)
        """
        # Update weights kwargs (if necessary) to handle different types of
        # weights
        weights_info = None
        if kwargs.get("weights") is not None:
            weights_info = _Weights(kwargs["weights"], self)
            kwargs["weights"] = weights_info.array

        # Convert any coordinate names to coordinates
        coordinates = self._as_list_of_coords(coords)

        if isinstance(
            aggregator, iris.analysis.WeightedAggregator
        ) and not aggregator.uses_weighting(**kwargs):
            msg = "Collapsing spatial coordinate {!r} without weighting"
            lat_match = [coord for coord in coordinates if "latitude" in coord.name()]
            if lat_match:
                for coord in lat_match:
                    warnings.warn(
                        msg.format(coord.name()),
                        category=iris.warnings.IrisUserWarning,
                    )

        # Determine the dimensions we need to collapse (and those we don't)
        # Remove duplicate dimensions and reverse the dimensions so the order
        # can be maintained when reshaping the data.
        dims_to_collapse = list(
            dict.fromkeys(d for coord in coordinates for d in self.coord_dims(coord))
        )[::-1]

        if aggregator.name() == "max_run" and len(dims_to_collapse) > 1:
            msg = "Not possible to calculate runs over more than one dimension"
            raise ValueError(msg)

        if not dims_to_collapse:
            msg = "Cannot collapse a dimension which does not describe any data."
            raise iris.exceptions.CoordinateCollapseError(msg)

        untouched_dims = sorted(set(range(self.ndim)) - set(dims_to_collapse))

        collapsed_cube = iris.util._strip_metadata_from_dims(self, dims_to_collapse)

        # Remove the collapsed dimension(s) from the metadata
        indices: list[slice | int] = [slice(None, None)] * self.ndim
        for dim in dims_to_collapse:
            indices[dim] = 0
        collapsed_cube = collapsed_cube[tuple(indices)]

        # Collapse any coordinates that span the dimension(s) being collapsed
        for coord in self.dim_coords + self.aux_coords:
            coord_dims = self.coord_dims(coord)
            if set(dims_to_collapse).intersection(coord_dims):
                local_dims = [
                    coord_dims.index(dim)
                    for dim in dims_to_collapse
                    if dim in coord_dims
                ]
                collapsed_cube.replace_coord(coord.collapsed(local_dims))

        # Record the axis(s) argument passed to 'aggregation', so the same is
        # passed to the 'update_metadata' function.
        collapse_axis = -1

        data_result = None

        # Perform the actual aggregation.
        if aggregator.cell_method == "peak":
            # The PEAK aggregator must collapse each coordinate separately.
            untouched_shape = [self.shape[d] for d in untouched_dims]
            collapsed_shape = [self.shape[d] for d in dims_to_collapse]
            new_shape = untouched_shape + collapsed_shape

            array_dims = untouched_dims + dims_to_collapse
            unrolled_data = np.transpose(self.data, array_dims).reshape(new_shape)

            for dim in dims_to_collapse:
                unrolled_data = aggregator.aggregate(unrolled_data, axis=-1, **kwargs)
            data_result = unrolled_data

        # Perform the aggregation in lazy form if possible.
        elif aggregator.lazy_func is not None and self.has_lazy_data():
            # Use a lazy operation separately defined by the aggregator, based
            # on the cube lazy array.
            # NOTE: do not reform the data in this case, as 'lazy_aggregate'
            # accepts multiple axes (unlike 'aggregate').
            if len(dims_to_collapse) == 1:
                # Replace a "list of 1 axes" with just a number :  This single-axis form is *required* by functions
                # like da.average (and np.average), if a 1d weights array is specified.
                collapse_axes: int | list[int] = dims_to_collapse[0]
            else:
                collapse_axes = list(dims_to_collapse)

            try:
                data_result = aggregator.lazy_aggregate(
                    self.lazy_data(), axis=collapse_axes, **kwargs
                )
            except TypeError:
                # TypeError - when unexpected keywords passed through (such as
                # weights to mean)
                pass

        # If we weren't able to complete a lazy aggregation, compute it
        # directly now.
        if data_result is None:
            # Perform the (non-lazy) aggregation over the cube data
            # First reshape the data so that the dimensions being aggregated
            # over are grouped 'at the end' (i.e. axis=-1).
            dims_to_collapse = sorted(dims_to_collapse)

            end_size = reduce(
                operator.mul, (self.shape[dim] for dim in dims_to_collapse)
            )
            untouched_shape = [self.shape[dim] for dim in untouched_dims]
            new_shape = untouched_shape + [end_size]
            dims = untouched_dims + dims_to_collapse
            unrolled_data = np.transpose(self.data, dims).reshape(new_shape)

            # Perform the same operation on the weights if applicable
            weights = kwargs.get("weights")
            if weights is not None and weights.ndim > 1:
                # Note: *don't* adjust 1d weights arrays, these have a special meaning for statistics functions.
                weights = weights.view()
                kwargs["weights"] = np.transpose(weights, dims).reshape(new_shape)

            data_result = aggregator.aggregate(unrolled_data, axis=-1, **kwargs)

        aggregator.update_metadata(
            collapsed_cube,
            coordinates,
            axis=collapse_axis,
            _weights_units=getattr(weights_info, "units", None),
            **kwargs,
        )
        result = aggregator.post_process(
            collapsed_cube, data_result, coordinates, **kwargs
        )
        return result

    def aggregated_by(
        self,
        coords: str | AuxCoord | DimCoord | Iterable[str | AuxCoord | DimCoord],
        aggregator: iris.analysis.Aggregator,
        climatological: bool = False,
        **kwargs,
    ) -> Cube:
        """Perform aggregation over the cube given one or more "group coordinates".

        A "group coordinate" is a coordinate where repeating values represent a
        single group, such as a month coordinate on a daily time slice. Repeated
        values will form a group even if they are not consecutive.

        The group coordinates must all be over the same cube dimension. Each
        common value group identified over all the group-by coordinates is
        collapsed using the provided aggregator.

        Weighted aggregations (:class:`iris.analysis.WeightedAggregator`) may
        also be supplied. These include :data:`~iris.analysis.MEAN` and
        :data:`~iris.analysis.SUM`.

        Weighted aggregations support an optional *weights* keyword argument.
        If set, this can be supplied as an array, cube, or (names of)
        :meth:`~iris.cube.Cube.coords`, :meth:`~iris.cube.Cube.cell_measures`,
        or :meth:`~iris.cube.Cube.ancillary_variables`. In all cases, the
        weights should be 1d or match the shape of the cube. When weights are
        not given as arrays, units are correctly handled for weighted sums,
        i.e., the original unit of the cube is multiplied by the units of the
        weights.

        Parameters
        ----------
        coords :
            One or more coordinates over which group aggregation is to be
            performed.
        aggregator :
            Aggregator to be applied to each group.
        climatological :
            Indicates whether the output is expected to be climatological. For
            any aggregated time coord(s), this causes the climatological flag to
            be set and the point for each cell to equal its first bound, thereby
            preserving the time of year.
        **kwargs :
            Aggregator and aggregation function keyword arguments.

        Returns
        -------
        :class:`iris.cube.Cube`

        Examples
        --------
            >>> import iris
            >>> import iris.analysis
            >>> import iris.coord_categorisation as cat
            >>> fname = iris.sample_data_path('ostia_monthly.nc')
            >>> cube = iris.load_cube(fname, 'surface_temperature')
            >>> cat.add_year(cube, 'time', name='year')
            >>> new_cube = cube.aggregated_by('year', iris.analysis.MEAN)
            >>> print(new_cube)
            surface_temperature / (K)           \
(time: 5; latitude: 18; longitude: 432)
                Dimension coordinates:
                    time                             \
x            -              -
                    latitude                         \
-            x              -
                    longitude                        \
-            -              x
                Auxiliary coordinates:
                    forecast_reference_time          \
x            -              -
                    year                             \
x            -              -
                Scalar coordinates:
                    forecast_period             0 hours
                Cell methods:
                    0                           month: year: mean
                    1                           year: mean
                Attributes:
                    Conventions                 'CF-1.5'
                    STASH                       m01s00i024

        """
        # Update weights kwargs (if necessary) to handle different types of
        # weights
        weights_info = None
        if kwargs.get("weights") is not None:
            weights_info = _Weights(kwargs["weights"], self)
            kwargs["weights"] = weights_info.array

        groupby_coords = []
        dimension_to_groupby: int | None = None

        coordinates = self._as_list_of_coords(coords)
        for coord in sorted(coordinates, key=lambda coord: coord.metadata):
            if coord.ndim > 1:
                msg = (
                    "Cannot aggregate_by coord %s as it is "
                    "multidimensional." % coord.name()
                )
                raise iris.exceptions.CoordinateMultiDimError(msg)
            dimensions = self.coord_dims(coord)
            if not dimensions:
                msg = (
                    'Cannot group-by the coordinate "%s", as its '
                    "dimension does not describe any data." % coord.name()
                )
                raise iris.exceptions.CoordinateCollapseError(msg)
            if dimension_to_groupby is None:
                dimension_to_groupby = dimensions[0]
            if dimension_to_groupby != dimensions[0]:
                msg = "Cannot group-by coordinates over different dimensions."
                raise iris.exceptions.CoordinateCollapseError(msg)
            groupby_coords.append(coord)

        if dimension_to_groupby is None:
            raise ValueError("Unable to aggregate by an empty list of `coords`.")

        # Check shape of weights. These must either match the shape of the cube
        # or be 1D (in this case, their length must be equal to the length of the
        # dimension we are aggregating over).
        weights = kwargs.get("weights")
        return_weights = kwargs.get("returned", False)
        if weights is not None:
            if weights.ndim == 1:
                if len(weights) != self.shape[dimension_to_groupby]:
                    raise ValueError(
                        f"1D weights must have the same length as the dimension "
                        f"that is aggregated, got {len(weights):d}, expected "
                        f"{self.shape[dimension_to_groupby]:d}"
                    )
                weights = iris.util.broadcast_to_shape(
                    weights,
                    self.shape,
                    (dimension_to_groupby,),
                )
            if weights.shape != self.shape:
                raise ValueError(
                    f"Weights must either be 1D or have the same shape as the "
                    f"cube, got shape {weights.shape} for weights, "
                    f"{self.shape} for cube"
                )

        # Determine the other coordinates that share the same group-by
        # coordinate dimension.
        shared_coords = list(
            filter(
                lambda coord_: coord_ not in groupby_coords
                and dimension_to_groupby in self.coord_dims(coord_),
                self.dim_coords + self.aux_coords,
            )
        )

        # Determine which of each shared coord's dimensions will be aggregated.
        shared_coords_and_dims = [
            (coord_, index)
            for coord_ in shared_coords
            for (index, dim) in enumerate(self.coord_dims(coord_))
            if dim == dimension_to_groupby
        ]

        # Create the aggregation group-by instance.
        groupby = iris.analysis._Groupby(
            groupby_coords,
            shared_coords_and_dims,
            climatological=climatological,
        )

        # Create the resulting aggregate-by cube and remove the original
        # coordinates that are going to be groupedby.
        aggregateby_cube = iris.util._strip_metadata_from_dims(
            self, [dimension_to_groupby]
        )
        key: list[slice | tuple[int, ...]] = [slice(None, None)] * self.ndim
        # Generate unique index tuple key to maintain monotonicity.
        key[dimension_to_groupby] = tuple(range(len(groupby)))
        aggregateby_cube = aggregateby_cube[tuple(key)]
        for coord in groupby_coords + shared_coords:
            aggregateby_cube.remove_coord(coord)

        coord_mapping = {}
        for coord in aggregateby_cube.coords():
            orig_id = id(self.coord(coord))
            coord_mapping[orig_id] = coord

        # Determine the group-by cube data shape.
        data_shape = list(self.shape + aggregator.aggregate_shape(**kwargs))
        data_shape[dimension_to_groupby] = len(groupby)

        # Choose appropriate data and functions for data aggregation.
        if aggregator.lazy_func is not None and self.has_lazy_data():
            input_data = self.lazy_data()
            agg_method = aggregator.lazy_aggregate
        else:
            input_data = self.data
            agg_method = aggregator.aggregate

        # Create data and weights slices.
        front_slice = (slice(None),) * dimension_to_groupby
        back_slice = (slice(None),) * (len(data_shape) - dimension_to_groupby - 1)

        groupby_subarrs = (
            iris.util._slice_data_with_keys(
                input_data, front_slice + (groupby_slice,) + back_slice
            )[1]
            for groupby_slice in groupby.group()
        )

        if weights is not None:
            groupby_subweights = (
                weights[front_slice + (groupby_slice,) + back_slice]
                for groupby_slice in groupby.group()
            )
        else:
            groupby_subweights = (None for _ in range(len(groupby)))

        # Aggregate data slices.
        agg = iris.analysis.create_weighted_aggregator_fn(
            agg_method, axis=dimension_to_groupby, **kwargs
        )
        result = tuple(map(agg, groupby_subarrs, groupby_subweights))

        # If weights are returned, "result" is a list of tuples (each tuple
        # contains two elements; the first is the aggregated data, the
        # second is the aggregated weights). Convert these to two lists
        # (one for the aggregated data and one for the aggregated weights)
        # before combining the different slices.
        if return_weights:
            data_result, weights_result = list(zip(*result))
            aggregateby_weights = _lazy.stack(weights_result, axis=dimension_to_groupby)
        else:
            data_result = result
            aggregateby_weights = None

        aggregateby_data = _lazy.stack(data_result, axis=dimension_to_groupby)
        # Ensure plain ndarray is output if plain ndarray was input.
        if ma.isMaskedArray(aggregateby_data) and not ma.isMaskedArray(input_data):
            aggregateby_data = ma.getdata(aggregateby_data)

        # Add the aggregation meta data to the aggregate-by cube.
        aggregator.update_metadata(
            aggregateby_cube,
            groupby_coords,
            aggregate=True,
            _weights_units=getattr(weights_info, "units", None),
            **kwargs,
        )
        # Replace the appropriate coordinates within the aggregate-by cube.
        dim_coords = self.coords(dimensions=dimension_to_groupby, dim_coords=True)
        dim_coord = dim_coords[0] if dim_coords else None

        for coord in groupby.coords:
            new_coord = coord.copy()

            # The metadata may have changed (e.g. climatology), so check if
            # there's a better coord to pass to self.coord_dims
            lookup_coord = coord
            for (
                cube_coord,
                groupby_coord,
            ) in groupby.coord_replacement_mapping:
                if coord == groupby_coord:
                    lookup_coord = cube_coord

            if (
                dim_coord is not None
                and dim_coord.metadata == lookup_coord.metadata
                and isinstance(coord, iris.coords.DimCoord)
            ):
                aggregateby_cube.add_dim_coord(new_coord, dimension_to_groupby)
            else:
                aggregateby_cube.add_aux_coord(new_coord, self.coord_dims(lookup_coord))
            coord_mapping[id(self.coord(lookup_coord))] = new_coord

        aggregateby_cube._aux_factories = []
        for factory in self.aux_factories:
            aggregateby_cube.add_aux_factory(factory.updated(coord_mapping))

        # Attach the aggregate-by data into the aggregate-by cube.
        if aggregateby_weights is None:
            data_result = aggregateby_data
        else:
            data_result = (aggregateby_data, aggregateby_weights)
        aggregateby_cube = aggregator.post_process(
            aggregateby_cube, data_result, coordinates, **kwargs
        )

        return aggregateby_cube

    def rolling_window(
        self,
        coord: str | AuxCoord | DimCoord,
        aggregator: iris.analysis.Aggregator,
        window: int,
        **kwargs,
    ) -> Cube:
        """Perform rolling window aggregation on a cube.

        Perform rolling window aggregation on a cube given a coordinate, an
        aggregation method and a window size.

        Parameters
        ----------
        coord :
            The coordinate over which to perform the rolling window
            aggregation.
        aggregator :
            Aggregator to be applied to the data.
        window :
            Size of window to use.
        **kwargs :
            Aggregator and aggregation function keyword arguments. The weights
            argument to the aggregator, if any, should be a 1d array, cube, or
            (names of) :meth:`~iris.cube.Cube.coords`,
            :meth:`~iris.cube.Cube.cell_measures`, or
            :meth:`~iris.cube.Cube.ancillary_variables` with the same length as
            the chosen window.

        Returns
        -------
        :class:`iris.cube.Cube`.

        Examples
        --------
            >>> import iris, iris.analysis
            >>> fname = iris.sample_data_path('GloSea4', 'ensemble_010.pp')
            >>> cube = iris.load_cube(fname, 'surface_temperature')
            >>> print(cube)
            surface_temperature / (K)           \
(time: 6; latitude: 145; longitude: 192)
                Dimension coordinates:
                    time                             \
x            -               -
                    latitude                         \
-            x               -
                    longitude                        \
-            -               x
                Auxiliary coordinates:
                    forecast_period                  \
x            -               -
                Scalar coordinates:
                    forecast_reference_time     2011-07-23 00:00:00
                    realization                 10
                Cell methods:
                    0                           time: mean (interval: 1 hour)
                Attributes:
                    STASH                       m01s00i024
                    source                      \
'Data from Met Office Unified Model'
                    um_version                  '7.6'

            >>> print(cube.rolling_window('time', iris.analysis.MEAN, 3))
            surface_temperature / (K)           \
(time: 4; latitude: 145; longitude: 192)
                Dimension coordinates:
                    time                             \
x            -               -
                    latitude                         \
-            x               -
                    longitude                        \
-            -               x
                Auxiliary coordinates:
                    forecast_period                  \
x            -               -
                Scalar coordinates:
                    forecast_reference_time     2011-07-23 00:00:00
                    realization                 10
                Cell methods:
                    0                           time: mean (interval: 1 hour)
                    1                           time: mean
                Attributes:
                    STASH                       m01s00i024
                    source                      \
'Data from Met Office Unified Model'
                    um_version                  '7.6'

            Notice that the forecast_period dimension now represents the 4
            possible windows of size 3 from the original cube.

        """  # noqa: D214, D406, D407, D410, D411
        # Update weights kwargs (if necessary) to handle different types of
        # weights
        weights_info = None
        if kwargs.get("weights") is not None:
            weights_info = _Weights(kwargs["weights"], self)
            kwargs["weights"] = weights_info.array

        coord = self._as_list_of_coords(coord)[0]

        if getattr(coord, "circular", False):
            raise iris.exceptions.NotYetImplementedError(
                "Rolling window over a circular coordinate."
            )

        if window < 2:
            raise ValueError(
                "Cannot perform rolling window with a window size less than 2."
            )

        if coord.ndim > 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)

        dimensions = self.coord_dims(coord)
        if len(dimensions) != 1:
            raise iris.exceptions.CoordinateCollapseError(
                'Cannot perform rolling window with coordinate "%s", '
                "must map to one data dimension." % coord.name()
            )
        dimension = dimensions[0]

        # Use indexing to get a result-cube of the correct shape.
        # NB. This indexes the data array which is wasted work.
        # As index-to-get-shape-then-fiddle is a common pattern, perhaps
        # some sort of `cube.prepare()` method would be handy to allow
        # re-shaping with given data, and returning a mapping of
        # old-to-new-coords (to avoid having to use metadata identity)?
        new_cube = iris.util._strip_metadata_from_dims(self, [dimension])
        key = [slice(None, None)] * self.ndim
        key[dimension] = slice(None, self.shape[dimension] - window + 1)
        new_cube = new_cube[tuple(key)]

        # take a view of the original data using the rolling_window function
        # this will add an extra dimension to the data at dimension + 1 which
        # represents the rolled window (i.e. will have a length of window)
        rolling_window_data = iris.util.rolling_window(
            self.core_data(), window=window, axis=dimension
        )

        # now update all of the coordinates to reflect the aggregation
        for coord_ in self.coords(dimensions=dimension):
            if coord_.has_bounds():
                warnings.warn(
                    "The bounds of coordinate %r were ignored in "
                    "the rolling window operation." % coord_.name(),
                    category=iris.warnings.IrisIgnoringBoundsWarning,
                )

            if coord_.ndim != 1:
                raise ValueError(
                    "Cannot calculate the rolling "
                    "window of %s as it is a multidimensional "
                    "coordinate." % coord_.name()
                )

            new_bounds = iris.util.rolling_window(coord_.core_points(), window)

            if np.issubdtype(new_bounds.dtype, np.str_):
                # Handle case where the AuxCoord contains string. The points
                # are the serialized form of the points contributing to each
                # window and the bounds are the first and last points in the
                # window as with numeric coordinates.
                new_points = np.apply_along_axis(lambda x: "|".join(x), -1, new_bounds)
                new_bounds = new_bounds[:, (0, -1)]
            else:
                # Take the first and last element of the rolled window (i.e.
                # the bounds) and the new points are the midpoints of these
                # bounds.
                new_bounds = new_bounds[:, (0, -1)]
                new_points = np.mean(new_bounds, axis=-1)

            # wipe the coords points and set the bounds
            new_coord = new_cube.coord(coord_)
            new_coord.points = new_points
            new_coord.bounds = new_bounds

        # update the metadata of the cube itself
        aggregator.update_metadata(
            new_cube,
            [coord],
            action="with a rolling window of length %s over" % window,
            _weights_units=getattr(weights_info, "units", None),
            **kwargs,
        )
        # and perform the data transformation, generating weights first if
        # needed
        if isinstance(
            aggregator, iris.analysis.WeightedAggregator
        ) and aggregator.uses_weighting(**kwargs):
            if "weights" in kwargs:
                weights = kwargs["weights"]
                if weights.ndim > 1 or weights.shape[0] != window:
                    raise ValueError(
                        "Weights for rolling window aggregation "
                        "must be a 1d array with the same length "
                        "as the window."
                    )
                kwargs = dict(kwargs)
                kwargs["weights"] = iris.util.broadcast_to_shape(
                    weights, rolling_window_data.shape, (dimension + 1,)
                )

        if aggregator.lazy_func is not None and self.has_lazy_data():
            agg_method = aggregator.lazy_aggregate
        else:
            agg_method = aggregator.aggregate
        data_result = agg_method(rolling_window_data, axis=dimension + 1, **kwargs)
        result = aggregator.post_process(new_cube, data_result, [coord], **kwargs)
        return result

    def interpolate(
        self,
        sample_points: Iterable[tuple[AuxCoord | DimCoord | str, np.typing.ArrayLike]],
        scheme: iris.analysis.InterpolationScheme,
        collapse_scalar: bool = True,
    ) -> Cube:
        """Interpolate from this :class:`~iris.cube.Cube` to the given sample points.

        Interpolate from this :class:`~iris.cube.Cube` to the given
        sample points using the given interpolation scheme.

        Parameters
        ----------
        sample_points :
            A sequence of (coordinate, points) pairs over which to
            interpolate. The values for coordinates that correspond to
            dates or times may optionally be supplied as datetime.datetime or
            cftime.datetime instances.
            The N pairs supplied will be used to create an N-d grid of points
            that will then be sampled (rather than just N points).
        scheme :
            An instance of the type of interpolation to use to interpolate from this
            :class:`~iris.cube.Cube` to the given sample points. The
            interpolation schemes currently available in Iris are:

            * :class:`iris.analysis.Linear`, and
            * :class:`iris.analysis.Nearest`.
        collapse_scalar : bool, default=True
            Whether to collapse the dimension of scalar sample points
            in the resulting cube. Default is True.

        Returns
        -------
        cube
            A cube interpolated at the given sample points.
            If `collapse_scalar` is True then the dimensionality of the cube
            will be the number of original cube dimensions minus
            the number of scalar coordinates.

        Examples
        --------
            >>> import datetime
            >>> import iris
            >>> path = iris.sample_data_path('uk_hires.pp')
            >>> cube = iris.load_cube(path, 'air_potential_temperature')
            >>> print(cube.summary(shorten=True))
            air_potential_temperature / (K)     \
(time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
            >>> print(cube.coord('time'))
            DimCoord :  time / (hours since 1970-01-01 00:00:00, standard calendar)
                points: [2009-11-19 10:00:00, 2009-11-19 11:00:00, 2009-11-19 12:00:00]
                shape: (3,)
                dtype: float64
                standard_name: 'time'
            >>> print(cube.coord('time').points)
            [349618. 349619. 349620.]
            >>> samples = [('time', 349618.5)]
            >>> result = cube.interpolate(samples, iris.analysis.Linear())
            >>> print(result.summary(shorten=True))
            air_potential_temperature / (K)     \
(model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
            >>> print(result.coord('time'))
            DimCoord :  time / (hours since 1970-01-01 00:00:00, standard calendar)
                points: [2009-11-19 10:30:00]
                shape: (1,)
                dtype: float64
                standard_name: 'time'
            >>> print(result.coord('time').points)
            [349618.5]
            >>> # For datetime-like coordinates, we can also use
            >>> # datetime-like objects.
            >>> samples = [('time', datetime.datetime(2009, 11, 19, 10, 30))]
            >>> result2 = cube.interpolate(samples, iris.analysis.Linear())
            >>> print(result2.summary(shorten=True))
            air_potential_temperature / (K)     \
(model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
            >>> print(result2.coord('time'))
            DimCoord :  time / (hours since 1970-01-01 00:00:00, standard calendar)
                points: [2009-11-19 10:30:00]
                shape: (1,)
                dtype: float64
                standard_name: 'time'
            >>> print(result2.coord('time').points)
            [349618.5]
            >>> print(result == result2)
            True

        """
        coords, points = zip(*sample_points)
        interp = scheme.interpolator(self, coords)  # type: ignore[arg-type]
        return interp(points, collapse_scalar=collapse_scalar)

    def regrid(self, grid: Cube, scheme: iris.analysis.RegriddingScheme) -> Cube:
        r"""Regrid this :class:`~iris.cube.Cube` on to the given target `grid`.

        Regrid this :class:`~iris.cube.Cube` on to the given target `grid`
        using the given regridding `scheme`.

        Parameters
        ----------
        grid :
            A :class:`~iris.cube.Cube` that defines the target grid.
        scheme :
            An instance of the type of regridding to use to regrid this cube onto the
            target grid. The regridding schemes in Iris currently include:

            * :class:`iris.analysis.Linear`\*,
            * :class:`iris.analysis.Nearest`\*,
            * :class:`iris.analysis.AreaWeighted`\*,
            * :class:`iris.analysis.UnstructuredNearest`,
            * :class:`iris.analysis.PointInCell`,

            \* Supports lazy regridding.

        Returns
        -------
        :class:`~iris.cube`
            A cube defined with the horizontal dimensions of the target grid
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid
            according to the given regridding scheme.

            The returned cube will have lazy data if the original cube has
            lazy data and the regridding scheme supports lazy regridding.

        Notes
        -----
        .. note::

            Both the source and target cubes must have a CoordSystem, otherwise
            this function is not applicable.

        """
        regridder = scheme.regridder(self, grid)
        return regridder(self)


class ClassDict(MutableMapping):
    """A mapping that stores objects keyed on their superclasses and their names.

    The mapping has a root class, all stored objects must be a subclass of the
    root class. The superclasses used for an object include the class of the
    object, but do not include the root class. Only one object is allowed for
    any key.

    """

    def __init__(self, superclass):
        if not isinstance(superclass, type):
            raise TypeError("The superclass must be a Python type or new style class.")
        self._superclass = superclass
        self._basic_map = {}
        self._retrieval_map = {}

    def add(self, object_, replace=False):
        """Add an object to the dictionary."""
        if not isinstance(object_, self._superclass):
            msg = "Only subclasses of {!r} are allowed as values.".format(
                self._superclass.__name__
            )
            raise TypeError(msg)
        # Find all the superclasses of the given object, starting with the
        # object's class.
        superclasses = type.mro(type(object_))
        if not replace:
            # Ensure nothing else is already registered against those
            # superclasses.
            # NB. This implies the _basic_map will also be empty for this
            # object.
            for key_class in superclasses:
                if key_class in self._retrieval_map:
                    msg = (
                        "Cannot add instance of '%s' because instance of "
                        "'%s' already added."
                        % (type(object_).__name__, key_class.__name__)
                    )
                    raise ValueError(msg)
        # Register the given object against those superclasses.
        for key_class in superclasses:
            self._retrieval_map[key_class] = object_
            self._retrieval_map[key_class.__name__] = object_
        self._basic_map[type(object_)] = object_

    def __getitem__(self, class_):
        try:
            return self._retrieval_map[class_]
        except KeyError:
            raise KeyError("Coordinate system %r does not exist." % class_)

    def __setitem__(self, key, value):
        raise NotImplementedError("You must call the add method instead.")

    def __delitem__(self, class_):
        cs = self[class_]
        keys = [k for k, v in self._retrieval_map.items() if v == cs]
        for key in keys:
            del self._retrieval_map[key]
        del self._basic_map[type(cs)]
        return cs

    def __len__(self):
        return len(self._basic_map)

    def __iter__(self):
        for item in self._basic_map:
            yield item

    def keys(self):
        """Return the keys of the dictionary mapping."""
        return self._basic_map.keys()


def sorted_axes(axes):
    """Return the axis names sorted alphabetically.

    Return the axis names sorted alphabetically, with the exception that
    't', 'z', 'y', and, 'x' are sorted to the end.

    """
    return sorted(
        axes,
        key=lambda name: ({"x": 4, "y": 3, "z": 2, "t": 1}.get(name, 0), name),
    )


# See Cube.slice() for the definition/context.
class _SliceIterator(Iterator):
    def __init__(self, cube, dims_index, requested_dims, ordered):
        self._cube = cube

        # Let Numpy do some work in providing all of the permutations of our
        # data shape. This functionality is something like:
        # ndindex(2, 1, 3) -> [(0, 0, 0), (0, 0, 1), (0, 0, 2),
        #                      (1, 0, 0), (1, 0, 1), (1, 0, 2)]
        self._ndindex = np.ndindex(*dims_index)

        self._requested_dims = requested_dims
        # indexing relating to sliced cube
        self._mod_requested_dims = np.argsort(requested_dims)
        self._ordered = ordered

    def __next__(self):
        # NB. When self._ndindex runs out it will raise StopIteration for us.
        index_tuple = next(self._ndindex)

        # Turn the given tuple into a list so that we can do something with it
        index_list = list(index_tuple)

        # For each of the spanning dimensions requested, replace the 0 with a
        # spanning slice
        for d in self._requested_dims:
            index_list[d] = slice(None, None)

        # Request the slice
        cube = self._cube[tuple(index_list)]

        if self._ordered:
            if any(self._mod_requested_dims != list(range(len(cube.shape)))):
                n = len(self._mod_requested_dims)
                sliced_dims = np.empty(n, dtype=int)
                sliced_dims[self._mod_requested_dims] = np.arange(n)
                cube.transpose(sliced_dims)

        return cube

    next = __next__

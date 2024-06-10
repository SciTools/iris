# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Definitions of derived coordinates."""

from abc import ABCMeta, abstractmethod
import warnings

import cf_units
import dask.array as da
import numpy as np

from iris._lazy_data import concatenate
from iris.common import CFVariableMixin, CoordMetadata, metadata_manager_factory
import iris.coords
from iris.warnings import IrisIgnoringBoundsWarning


class AuxCoordFactory(CFVariableMixin, metaclass=ABCMeta):
    """Represents a "factory" which can manufacture additional auxiliary coordinate.

    Represents a "factory" which can manufacture an additional auxiliary
    coordinate on demand, by combining the values of other coordinates.

    Each concrete subclass represents a specific formula for deriving
    values from other coordinates.

    The `standard_name`, `long_name`, `var_name`, `units`, `attributes` and
    `coord_system` of the factory are used to set the corresponding
    properties of the resulting auxiliary coordinates.

    """

    def __init__(self):
        # Configure the metadata manager.
        if not hasattr(self, "_metadata_manager"):
            self._metadata_manager = metadata_manager_factory(CoordMetadata)

        #: Descriptive name of the coordinate made by the factory
        self.long_name = None

        #: netCDF variable name for the coordinate made by the factory
        self.var_name = None

        self.coord_system = None
        # See the climatological property getter.
        self._metadata_manager.climatological = False

    @property
    def coord_system(self):
        """The coordinate-system (if any) of the coordinate made by the factory."""
        return self._metadata_manager.coord_system

    @coord_system.setter
    def coord_system(self, value):
        self._metadata_manager.coord_system = value

    @property
    def climatological(self):
        """Return False, as a factory itself can never have points/bounds.

        Always returns False, as a factory itself can never have points/bounds
        and therefore can never be climatological by definition.

        """
        return self._metadata_manager.climatological

    @property
    @abstractmethod
    def dependencies(self):
        """Return a dict mapping from constructor argument.

        Return a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """

    @abstractmethod
    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate.

            See :meth:`iris.cube.Cube.coord_dims()`.

        """

    def update(self, old_coord, new_coord=None):
        """Notify the factory of the removal/replacement of a coordinate.

        Notify the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Parameters
        ----------
        old_coord :
            The coordinate to be removed/replaced.
        new_coord : optional
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        new_dependencies = self.dependencies
        for name, coord in self.dependencies.items():
            if old_coord is coord:
                new_dependencies[name] = new_coord
                try:
                    self._check_dependencies(**new_dependencies)
                except ValueError as e:
                    msg = "Failed to update dependencies. " + str(e)
                    raise ValueError(msg)
                else:
                    setattr(self, name, new_coord)
                break

    def __repr__(self):
        def arg_text(item):
            key, coord = item
            return "{}={}".format(key, str(coord and repr(coord.name())))

        items = sorted(self.dependencies.items(), key=lambda item: item[0])
        args = map(arg_text, items)
        return "<{}({})>".format(type(self).__name__, ", ".join(args))

    def derived_dims(self, coord_dims_func):
        """Return the cube dimensions for the derived coordinate.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant to a given
            coordinate.
            See :meth:`iris.cube.Cube.coord_dims()`.

        Returns
        -------
        A sorted list of cube dimension numbers.

        """
        # Which dimensions are relevant?
        # e.g. If sigma -> [1] and orog -> [2, 3] then result = [1, 2, 3]
        derived_dims = set()
        for coord in self.dependencies.values():
            if coord:
                derived_dims.update(coord_dims_func(coord))

        # Apply a fixed order so we know how to map dependency dims to
        # our own dims (and so the Cube can map them to Cube dims).
        derived_dims = tuple(sorted(derived_dims))
        return derived_dims

    def updated(self, new_coord_mapping):
        """Create a new instance of this factory.

        Create a new instance of this factory where the dependencies
        are replaced according to the given mapping.

        Parameters
        ----------
        new_coord_mapping :
            A dictionary mapping from the object IDs potentially used
            by this factory, to the coordinate objects that should be
            used instead.

        """
        new_dependencies = {}
        for key, coord in self.dependencies.items():
            if coord:
                coord = new_coord_mapping[id(coord)]
            new_dependencies[key] = coord
        return type(self)(**new_dependencies)

    def xml_element(self, doc):
        """Return a DOM element describing this coordinate factory."""
        element = doc.createElement("coordFactory")
        for key, coord in self.dependencies.items():
            element.setAttribute(key, coord._xml_id())
        element.appendChild(self.make_coord().xml_element(doc))
        return element

    def _dependency_dims(self, coord_dims_func):
        dependency_dims = {}
        for key, coord in self.dependencies.items():
            if coord:
                dependency_dims[key] = coord_dims_func(coord)
        return dependency_dims

    @staticmethod
    def _nd_bounds(coord, dims, ndim):
        """Return a lazy bounds array for a dependency coordinate, 'coord'.

        The result is aligned to the first 'ndim' cube dimensions, and
        expanded to the full ('ndim'+1)-dimensional shape.

        The value of 'ndim' must be >= the highest cube dimension of the
        dependency coordinate.

        The extra final result dimension ('ndim'-th) is the bounds dimension.

        Example::
            coord.shape == (70,)
            coord.nbounds = 2
            dims == [3]
            ndim == 5

        results in::

            nd_bounds.shape == (1, 1, 1, 70, 1, 2)

        """
        # Transpose to be consistent with the Cube.
        sorted_pairs = sorted(enumerate(dims), key=lambda pair: pair[1])
        transpose_order = [pair[0] for pair in sorted_pairs] + [len(dims)]
        bounds = coord.lazy_bounds()
        if dims and transpose_order != list(range(len(dims))):
            bounds = bounds.transpose(transpose_order)

        # Figure out the n-dimensional shape.
        nd_shape = [1] * ndim + [coord.nbounds]
        for dim, size in zip(dims, coord.shape):
            nd_shape[dim] = size
        bounds = bounds.reshape(nd_shape)
        return bounds

    @staticmethod
    def _nd_points(coord, dims, ndim):
        """Return a lazy points array for a dependency coordinate, 'coord'.

        The result is aligned to the first 'ndim' cube dimensions, and
        expanded to the full 'ndim'-dimensional shape.

        The value of 'ndim' must be >= the highest cube dimension of the
        dependency coordinate.

        Examples
        --------
        ::
            coord.shape == (4, 3)
            dims == [3, 2]
            ndim == 5

        results in::

            nd_points.shape == (1, 1, 3, 4, 1)

        """
        # Transpose to be consistent with the Cube.
        sorted_pairs = sorted(enumerate(dims), key=lambda pair: pair[1])
        transpose_order = [pair[0] for pair in sorted_pairs]
        points = coord.lazy_points()
        if dims and transpose_order != list(range(len(dims))):
            points = points.transpose(transpose_order)

        # Expand dimensionality to be consistent with the Cube.
        if dims:
            keys = [None] * ndim
            for dim, size in zip(dims, coord.shape):
                keys[dim] = slice(None)
            points = points[tuple(keys)]
        else:
            # Scalar coordinates have one dimensional points despite
            # mapping to zero dimensions, so we only need to add N-1
            # new dimensions.
            keys = (None,) * (ndim - 1)
            points = points[keys]
        return points

    def _remap(self, dependency_dims, derived_dims):
        """Return a mapping from dependency names to coordinate points arrays.

        For dependencies that are present, the values are all expanded and
        aligned to the same dimensions, which is the full set of all the
        dependency dimensions.
        These non-missing values are all lazy arrays.
        Missing dependencies, however, are assigned a scalar value of 0.0.

        """
        if derived_dims:
            ndim = max(derived_dims) + 1
        else:
            ndim = 1

        nd_points_by_key = {}
        for key, coord in self.dependencies.items():
            if coord:
                # Get the points as consistent with the Cube.
                nd_points = self._nd_points(coord, dependency_dims[key], ndim)

                # Restrict to just the dimensions relevant to the
                # derived coord. NB. These are always in Cube-order, so
                # no transpose is needed.
                if derived_dims:
                    keys = tuple(
                        slice(None) if dim in derived_dims else 0 for dim in range(ndim)
                    )
                    nd_points = nd_points[keys]
            else:
                # If no coord, treat value as zero.
                # Use a float16 to provide `shape` attribute and avoid
                # promoting other arguments to a higher precision.
                nd_points = np.float16(0)

            nd_points_by_key[key] = nd_points
        return nd_points_by_key

    def _remap_with_bounds(self, dependency_dims, derived_dims):
        """Return a mapping from dependency names to coordinate bounds arrays.

        For dependencies that are present, the values are all expanded and
        aligned to the same dimensions, which is the full set of all the
        dependency dimensions, plus an extra bounds dimension.
        These non-missing values are all lazy arrays.
        Missing dependencies, however, are assigned a scalar value of 0.0.

        Where a dependency coordinate has no bounds, then the associated value
        is taken from its points array, but reshaped to have an extra bounds
        dimension of length 1.

        """
        if derived_dims:
            ndim = max(derived_dims) + 1
        else:
            ndim = 1

        nd_values_by_key = {}
        for key, coord in self.dependencies.items():
            if coord:
                # Get the bounds or points as consistent with the Cube.
                if coord.nbounds:
                    nd_values = self._nd_bounds(coord, dependency_dims[key], ndim)
                else:
                    nd_values = self._nd_points(coord, dependency_dims[key], ndim)

                # Restrict to just the dimensions relevant to the
                # derived coord. NB. These are always in Cube-order, so
                # no transpose is needed.
                shape = []
                for dim in derived_dims:
                    shape.append(nd_values.shape[dim])
                # Ensure the array always has at least one dimension to be
                # compatible with normal coordinates.
                if not derived_dims:
                    shape.append(1)
                # Add on the N-bounds dimension
                if coord.nbounds:
                    shape.append(nd_values.shape[-1])
                else:
                    # NB. For a non-bounded coordinate we still need an
                    # extra dimension to make the shape compatible, so
                    # we just add an extra 1.
                    shape.append(1)
                nd_values = nd_values.reshape(shape)
            else:
                # If no coord, treat value as zero.
                # Use a float16 to provide `shape` attribute and avoid
                # promoting other arguments to a higher precision.
                nd_values = np.float16(0)

            nd_values_by_key[key] = nd_values
        return nd_values_by_key


class AtmosphereSigmaFactory(AuxCoordFactory):
    """Define an atmosphere sigma coordinate factory with the following formula.

    .. math::
        p = ptop + sigma * (ps - ptop)

    """

    def __init__(self, pressure_at_top=None, sigma=None, surface_air_pressure=None):
        """Create an atmosphere sigma coordinate factory with a formula.

        .. math::
            p(n, k, j, i) = pressure_at_top + sigma(k) *
                            (surface_air_pressure(n, j, i) - pressure_at_top)

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(pressure_at_top, sigma, surface_air_pressure)

        # Initialize instance attributes
        self.units = pressure_at_top.units
        self.pressure_at_top = pressure_at_top
        self.sigma = sigma
        self.surface_air_pressure = surface_air_pressure
        self.standard_name = "air_pressure"
        self.attributes = {}

    @staticmethod
    def _check_dependencies(pressure_at_top, sigma, surface_air_pressure):
        """Check for sufficient coordinates."""
        if any(
            [
                pressure_at_top is None,
                sigma is None,
                surface_air_pressure is None,
            ]
        ):
            raise ValueError(
                "Unable to construct atmosphere sigma coordinate factory due "
                "to insufficient source coordinates"
            )

        # Check dimensions
        if pressure_at_top.shape not in ((), (1,)):
            raise ValueError(
                f"Expected scalar 'pressure_at_top' coordinate, got shape "
                f"{pressure_at_top.shape}"
            )

        # Check bounds
        if sigma.nbounds not in (0, 2):
            raise ValueError(
                f"Invalid 'sigma' coordinate: must have either 0 or 2 bounds, "
                f"got {sigma.nbounds:d}"
            )
        for coord in (pressure_at_top, surface_air_pressure):
            if coord.nbounds:
                msg = (
                    f"Coordinate '{coord.name()}' has bounds. These will "
                    "be disregarded"
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        # Check units
        if sigma.units.is_unknown():
            # Be graceful, and promote unknown to dimensionless units.
            sigma.units = cf_units.Unit("1")
        if not sigma.units.is_dimensionless():
            raise ValueError(
                f"Invalid units: 'sigma' must be dimensionless, got " f"'{sigma.units}'"
            )
        if pressure_at_top.units != surface_air_pressure.units:
            raise ValueError(
                f"Incompatible units: 'pressure_at_top' and "
                f"'surface_air_pressure' must have the same units, got "
                f"'{pressure_at_top.units}' and "
                f"'{surface_air_pressure.units}'"
            )
        if not pressure_at_top.units.is_convertible("Pa"):
            raise ValueError(
                "Invalid units: 'pressure_at_top' and 'surface_air_pressure' "
                "must have units of pressure"
            )

    @property
    def dependencies(self):
        """Return dependencies."""
        dependencies = {
            "pressure_at_top": self.pressure_at_top,
            "sigma": self.sigma,
            "surface_air_pressure": self.surface_air_pressure,
        }
        return dependencies

    @staticmethod
    def _derive(pressure_at_top, sigma, surface_air_pressure):
        """Derive coordinate."""
        return pressure_at_top + sigma * (surface_air_pressure - pressure_at_top)

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate.

            See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Which dimensions are relevant?
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["pressure_at_top"],
            nd_points_by_key["sigma"],
            nd_points_by_key["surface_air_pressure"],
        )

        # Bounds
        bounds = None
        if self.sigma.nbounds:
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            pressure_at_top = nd_values_by_key["pressure_at_top"]
            sigma = nd_values_by_key["sigma"]
            surface_air_pressure = nd_values_by_key["surface_air_pressure"]
            ok_bound_shapes = [(), (1,), (2,)]
            if sigma.shape[-1:] not in ok_bound_shapes:
                raise ValueError("Invalid sigma coordinate bounds")
            if pressure_at_top.shape[-1:] not in [(), (1,)]:
                warnings.warn(
                    "Pressure at top coordinate has bounds. These are being "
                    "disregarded",
                    category=IrisIgnoringBoundsWarning,
                )
                pressure_at_top_pts = nd_points_by_key["pressure_at_top"]
                bds_shape = list(pressure_at_top_pts.shape) + [1]
                pressure_at_top = pressure_at_top_pts.reshape(bds_shape)
            if surface_air_pressure.shape[-1:] not in [(), (1,)]:
                warnings.warn(
                    "Surface pressure coordinate has bounds. These are being "
                    "disregarded",
                    category=IrisIgnoringBoundsWarning,
                )
                surface_air_pressure_pts = nd_points_by_key["surface_air_pressure"]
                bds_shape = list(surface_air_pressure_pts.shape) + [1]
                surface_air_pressure = surface_air_pressure_pts.reshape(bds_shape)
            bounds = self._derive(pressure_at_top, sigma, surface_air_pressure)

        # Create coordinate
        return iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )


class HybridHeightFactory(AuxCoordFactory):
    """Defines a hybrid-height coordinate factory."""

    def __init__(self, delta=None, sigma=None, orography=None):
        """Create a hybrid-height coordinate factory with the following formula.

        .. math::
            z = a + b * orog

        At least one of `delta` or `orography` must be provided.

        Parameters
        ----------
        delta : Coord, optional
            The coordinate providing the `a` term.
        sigma : Coord, optional
            The coordinate providing the `b` term.
        orography : Coord, optional
            The coordinate providing the `orog` term.

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        if delta and delta.nbounds not in (0, 2):
            raise ValueError(
                "Invalid delta coordinate: must have either 0 or 2 bounds."
            )
        if sigma and sigma.nbounds not in (0, 2):
            raise ValueError(
                "Invalid sigma coordinate: must have either 0 or 2 bounds."
            )
        if orography and orography.nbounds:
            msg = (
                "Orography coordinate {!r} has bounds."
                " These will be disregarded.".format(orography.name())
            )
            warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        self.delta = delta
        self.sigma = sigma
        self.orography = orography

        self.standard_name = "altitude"
        if delta is None and orography is None:
            emsg = "Unable to determine units: no delta or orography available."
            raise ValueError(emsg)
        if delta and orography and delta.units != orography.units:
            emsg = "Incompatible units: delta and orography must have the same units."
            raise ValueError(emsg)
        self.units = (delta and delta.units) or orography.units
        if not self.units.is_convertible("m"):
            emsg = (
                "Invalid units: delta and/or orography must be expressed "
                "in length units."
            )
            raise ValueError(emsg)
        self.attributes = {"positive": "up"}

    @property
    def dependencies(self):
        """Return a dict mapping from constructor arg names to coordinates.

        Return a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return {
            "delta": self.delta,
            "sigma": self.sigma,
            "orography": self.orography,
        }

    def _derive(self, delta, sigma, orography):
        return delta + sigma * orography

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate.

            See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Which dimensions are relevant?
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["delta"],
            nd_points_by_key["sigma"],
            nd_points_by_key["orography"],
        )

        bounds = None
        if (self.delta and self.delta.nbounds) or (self.sigma and self.sigma.nbounds):
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            delta = nd_values_by_key["delta"]
            sigma = nd_values_by_key["sigma"]
            orography = nd_values_by_key["orography"]
            ok_bound_shapes = [(), (1,), (2,)]
            if delta.shape[-1:] not in ok_bound_shapes:
                raise ValueError("Invalid delta coordinate bounds.")
            if sigma.shape[-1:] not in ok_bound_shapes:
                raise ValueError("Invalid sigma coordinate bounds.")
            if orography.shape[-1:] not in [(), (1,)]:
                warnings.warn(
                    "Orography coordinate has bounds. These are being disregarded.",
                    category=IrisIgnoringBoundsWarning,
                    stacklevel=2,
                )
                orography_pts = nd_points_by_key["orography"]
                bds_shape = list(orography_pts.shape) + [1]
                orography = orography_pts.reshape(bds_shape)

            bounds = self._derive(delta, sigma, orography)

        hybrid_height = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return hybrid_height

    def update(self, old_coord, new_coord=None):
        """Notify the factory of the removal/replacement of a coordinate.

        Notify the factory of the removal/replacement of a coordinate
        which might be a dependency.

        Parameters
        ----------
        old_coord :
            The coordinate to be removed/replaced.
        new_coord : optional
            If None, any dependency using old_coord is removed, otherwise
            any dependency using old_coord is updated to use new_coord.

        """
        if self.delta is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError(
                    "Invalid delta coordinate: must have either 0 or 2 bounds."
                )
            self.delta = new_coord
        elif self.sigma is old_coord:
            if new_coord and new_coord.nbounds not in (0, 2):
                raise ValueError(
                    "Invalid sigma coordinate: must have either 0 or 2 bounds."
                )
            self.sigma = new_coord
        elif self.orography is old_coord:
            if new_coord and new_coord.nbounds:
                msg = (
                    "Orography coordinate {!r} has bounds."
                    " These will be disregarded.".format(new_coord.name())
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)
            self.orography = new_coord


class HybridPressureFactory(AuxCoordFactory):
    """Define a hybrid-pressure coordinate factory."""

    def __init__(self, delta=None, sigma=None, surface_air_pressure=None):
        """Create a hybrid-height coordinate factory with the following formula.

        .. math::
            p = ap + b * ps

        At least one of `delta` or `surface_air_pressure` must be provided.

        Parameters
        ----------
        delta : Coord, optional
            The coordinate providing the `ap` term.
        sigma : Coord, optional
            The coordinate providing the `b` term.
        surface_air_pressure : Coord, optional
            The coordinate providing the `ps` term.

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coords meet necessary conditions.
        self._check_dependencies(delta, sigma, surface_air_pressure)
        self.units = (delta and delta.units) or surface_air_pressure.units

        self.delta = delta
        self.sigma = sigma
        self.surface_air_pressure = surface_air_pressure

        self.standard_name = "air_pressure"
        self.attributes = {}

    @staticmethod
    def _check_dependencies(delta, sigma, surface_air_pressure):
        # Check for sufficient coordinates.
        if delta is None and (sigma is None or surface_air_pressure is None):
            msg = (
                "Unable to construct hybrid pressure coordinate factory "
                "due to insufficient source coordinates."
            )
            raise ValueError(msg)

        # Check bounds.
        if delta and delta.nbounds not in (0, 2):
            raise ValueError(
                "Invalid delta coordinate: must have either 0 or 2 bounds."
            )
        if sigma and sigma.nbounds not in (0, 2):
            raise ValueError(
                "Invalid sigma coordinate: must have either 0 or 2 bounds."
            )
        if surface_air_pressure and surface_air_pressure.nbounds:
            msg = (
                "Surface pressure coordinate {!r} has bounds. These will"
                " be disregarded.".format(surface_air_pressure.name())
            )
            warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        # Check units.
        if sigma is not None and sigma.units.is_unknown():
            # Be graceful, and promote unknown to dimensionless units.
            sigma.units = cf_units.Unit("1")

        if sigma is not None and not sigma.units.is_dimensionless():
            raise ValueError("Invalid units: sigma must be dimensionless.")
        if (
            delta is not None
            and surface_air_pressure is not None
            and delta.units != surface_air_pressure.units
        ):
            msg = (
                "Incompatible units: delta and "
                "surface_air_pressure must have the same units."
            )
            raise ValueError(msg)

        if delta is not None:
            units = delta.units
        else:
            units = surface_air_pressure.units

        if not units.is_convertible("Pa"):
            msg = (
                "Invalid units: delta and "
                "surface_air_pressure must have units of pressure."
            )
            raise ValueError(msg)

    @property
    def dependencies(self):
        """Return a dict mapping from constructor arg names to coordinates.

        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return {
            "delta": self.delta,
            "sigma": self.sigma,
            "surface_air_pressure": self.surface_air_pressure,
        }

    def _derive(self, delta, sigma, surface_air_pressure):
        return delta + sigma * surface_air_pressure

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate.

            See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Which dimensions are relevant?
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["delta"],
            nd_points_by_key["sigma"],
            nd_points_by_key["surface_air_pressure"],
        )

        bounds = None
        if (self.delta and self.delta.nbounds) or (self.sigma and self.sigma.nbounds):
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            delta = nd_values_by_key["delta"]
            sigma = nd_values_by_key["sigma"]
            surface_air_pressure = nd_values_by_key["surface_air_pressure"]
            ok_bound_shapes = [(), (1,), (2,)]
            if delta.shape[-1:] not in ok_bound_shapes:
                raise ValueError("Invalid delta coordinate bounds.")
            if sigma.shape[-1:] not in ok_bound_shapes:
                raise ValueError("Invalid sigma coordinate bounds.")
            if surface_air_pressure.shape[-1:] not in [(), (1,)]:
                warnings.warn(
                    "Surface pressure coordinate has bounds. "
                    "These are being disregarded.",
                    category=IrisIgnoringBoundsWarning,
                )
                surface_air_pressure_pts = nd_points_by_key["surface_air_pressure"]
                bds_shape = list(surface_air_pressure_pts.shape) + [1]
                surface_air_pressure = surface_air_pressure_pts.reshape(bds_shape)

            bounds = self._derive(delta, sigma, surface_air_pressure)

        hybrid_pressure = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return hybrid_pressure


class OceanSigmaZFactory(AuxCoordFactory):
    """Defines an ocean sigma over z coordinate factory."""

    def __init__(
        self,
        sigma=None,
        eta=None,
        depth=None,
        depth_c=None,
        nsigma=None,
        zlev=None,
    ):
        """Create an ocean sigma over z coordinate factory with the following formula.

        if k < nsigma:

        .. math::
            z(n, k, j, i) = eta(n, j, i) + sigma(k) *
                            (min(depth_c, depth(j, i)) + eta(n, j, i))

        if k >= nsigma:

        .. math::
            z(n, k, j, i) = zlev(k)

        The `zlev` and 'nsigma' coordinates must be provided, and at least
        either `eta`, or 'sigma' and `depth` and `depth_c` coordinates.

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(sigma, eta, depth, depth_c, nsigma, zlev)
        self.units = zlev.units

        self.sigma = sigma
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c
        self.nsigma = nsigma
        self.zlev = zlev

        self.standard_name = "sea_surface_height_above_reference_ellipsoid"
        self.attributes = {"positive": "up"}

    @staticmethod
    def _check_dependencies(sigma, eta, depth, depth_c, nsigma, zlev):
        # Check for sufficient factory coordinates.
        if zlev is None:
            raise ValueError("Unable to determine units: no zlev coordinate available.")
        if nsigma is None:
            raise ValueError("Missing nsigma coordinate.")

        if eta is None and (sigma is None or depth_c is None or depth is None):
            msg = (
                "Unable to construct ocean sigma over z coordinate "
                "factory due to insufficient source coordinates."
            )
            raise ValueError(msg)

        # Check bounds and shape.
        for coord, term in ((sigma, "sigma"), (zlev, "zlev")):
            if coord is not None and coord.nbounds not in (0, 2):
                msg = (
                    "Invalid {} coordinate {!r}: must have either "
                    "0 or 2 bounds.".format(term, coord.name())
                )
                raise ValueError(msg)

        if sigma and sigma.nbounds != zlev.nbounds:
            msg = (
                "The sigma coordinate {!r} and zlev coordinate {!r} "
                "must be equally bounded.".format(sigma.name(), zlev.name())
            )
            raise ValueError(msg)

        coords = (
            (eta, "eta"),
            (depth, "depth"),
            (depth_c, "depth_c"),
            (nsigma, "nsigma"),
        )
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = (
                    "The {} coordinate {!r} has bounds. "
                    "These are being disregarded.".format(term, coord.name())
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        for coord, term in ((depth_c, "depth_c"), (nsigma, "nsigma")):
            if coord is not None and coord.shape != (1,):
                msg = "Expected scalar {} coordinate {!r}: got shape {!r}.".format(
                    term, coord.name(), coord.shape
                )
                raise ValueError(msg)

        # Check units.
        if not zlev.units.is_convertible("m"):
            msg = (
                "Invalid units: zlev coordinate {!r} "
                "must have units of distance.".format(zlev.name())
            )
            raise ValueError(msg)

        if sigma is not None and sigma.units.is_unknown():
            # Be graceful, and promote unknown to dimensionless units.
            sigma.units = cf_units.Unit("1")

        if sigma is not None and not sigma.units.is_dimensionless():
            msg = "Invalid units: sigma coordinate {!r} must be dimensionless.".format(
                sigma.name()
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth_c, "depth_c"), (depth, "depth"))
        for coord, term in coords:
            if coord is not None and coord.units != zlev.units:
                msg = (
                    "Incompatible units: {} coordinate {!r} and zlev "
                    "coordinate {!r} must have "
                    "the same units.".format(term, coord.name(), zlev.name())
                )
                raise ValueError(msg)

    @property
    def dependencies(self):
        """Return a dict mapping from constructor arg names to coordinates.

        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(
            sigma=self.sigma,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
            nsigma=self.nsigma,
            zlev=self.zlev,
        )

    def _derive(self, sigma, eta, depth, depth_c, zlev, nsigma, coord_dims_func):
        # Calculate the index of the 'z' dimension in the input arrays.
        # First find the cube 'z' dimension ...
        [cube_z_dim] = coord_dims_func(self.dependencies["zlev"])
        # ... then calculate the corresponding dependency dimension.
        derived_cubedims = self.derived_dims(coord_dims_func)
        z_dim = derived_cubedims.index(cube_z_dim)

        # Calculate the result shape as a combination of all the inputs.
        # Note: all the inputs have the same number of dimensions >= 1, except
        # for any missing dependencies, which have scalar values.
        allshapes = np.array(
            [el.shape for el in (sigma, eta, depth, depth_c, zlev) if el.ndim > 0]
        )
        result_shape = list(np.max(allshapes, axis=0))
        ndims = len(result_shape)

        # Make a slice tuple to index the first nsigma z-levels.
        z_slices_nsigma = [slice(None)] * ndims
        z_slices_nsigma[z_dim] = slice(0, int(nsigma))
        z_slices_nsigma = tuple(z_slices_nsigma)
        # Make a slice tuple to index the remaining z-levels.
        z_slices_rest = [slice(None)] * ndims
        z_slices_rest[z_dim] = slice(int(nsigma), None)
        z_slices_rest = tuple(z_slices_rest)

        # Perform the ocean sigma over z coordinate nsigma slice.
        if eta.ndim:
            eta = eta[z_slices_nsigma]
        if sigma.ndim:
            sigma = sigma[z_slices_nsigma]
        if depth.ndim:
            depth = depth[z_slices_nsigma]
        # Note that, this performs a point-wise minimum.
        nsigma_levs = eta + sigma * (da.minimum(depth_c, depth) + eta)

        # Make a result-shaped lazy "ones" array for expanding partial results.
        # Note: for the 'chunks' arg, we try to use [1, 1, ... ny, nx].
        # This calculation could be assuming too much in some cases, as we
        # don't actually check the dimensions of our dependencies anywhere.
        result_chunks = result_shape
        if len(result_shape) > 1:
            result_chunks = [1] * len(result_shape)
            result_chunks[-2:] = result_shape[-2:]
        ones_full_result = da.ones(result_shape, chunks=result_chunks, dtype=zlev.dtype)

        # Expand nsigma_levs to its full required shape : needed as the
        # calculated result may have a fixed size of 1 in some dimensions.
        result_nsigma_levs = nsigma_levs * ones_full_result[z_slices_nsigma]

        # Likewise, expand zlev to its full required shape.
        result_rest_levs = zlev[z_slices_rest] * ones_full_result[z_slices_rest]

        # Combine nsigma and 'rest' levels for the final result.
        result = concatenate([result_nsigma_levs, result_rest_levs], axis=z_dim)
        return result

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)

        [nsigma] = nd_points_by_key["nsigma"]
        points = self._derive(
            nd_points_by_key["sigma"],
            nd_points_by_key["eta"],
            nd_points_by_key["depth"],
            nd_points_by_key["depth_c"],
            nd_points_by_key["zlev"],
            nsigma,
            coord_dims_func,
        )

        bounds = None
        if self.zlev.nbounds or (self.sigma and self.sigma.nbounds):
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            valid_shapes = [(), (1,), (2,)]
            for key in ("sigma", "zlev"):
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = "Invalid bounds for {} coordinate {!r}.".format(key, name)
                    raise ValueError(msg)
            valid_shapes.pop()
            for key in ("eta", "depth", "depth_c", "nsigma"):
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = (
                        "The {} coordinate {!r} has bounds. "
                        "These are being disregarded.".format(key, name)
                    )
                    warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)
                    # Swap bounds with points.
                    bds_shape = list(nd_points_by_key[key].shape) + [1]
                    bounds = nd_points_by_key[key].reshape(bds_shape)
                    nd_values_by_key[key] = bounds

            bounds = self._derive(
                nd_values_by_key["sigma"],
                nd_values_by_key["eta"],
                nd_values_by_key["depth"],
                nd_values_by_key["depth_c"],
                nd_values_by_key["zlev"],
                nsigma,
                coord_dims_func,
            )

        coord = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return coord


class OceanSigmaFactory(AuxCoordFactory):
    """Defines an ocean sigma coordinate factory."""

    def __init__(self, sigma=None, eta=None, depth=None):
        """Create an ocean sigma coordinate factory with the following formula.

        .. math::
            z(n, k, j, i) = eta(n, j, i) + sigma(k) *
                            (depth(j, i) + eta(n, j, i))

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(sigma, eta, depth)
        self.units = depth.units

        self.sigma = sigma
        self.eta = eta
        self.depth = depth

        self.standard_name = "sea_surface_height_above_reference_ellipsoid"
        self.attributes = {"positive": "up"}

    @staticmethod
    def _check_dependencies(sigma, eta, depth):
        # Check for sufficient factory coordinates.
        if eta is None or sigma is None or depth is None:
            msg = (
                "Unable to construct ocean sigma coordinate "
                "factory due to insufficient source coordinates."
            )
            raise ValueError(msg)

        # Check bounds and shape.
        coord, term = (sigma, "sigma")
        if coord is not None and coord.nbounds not in (0, 2):
            msg = "Invalid {} coordinate {!r}: must have either 0 or 2 bounds.".format(
                term, coord.name()
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = (
                    "The {} coordinate {!r} has bounds. "
                    "These are being disregarded.".format(term, coord.name())
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        # Check units.
        if sigma is not None and sigma.units.is_unknown():
            # Be graceful, and promote unknown to dimensionless units.
            sigma.units = cf_units.Unit("1")

        if sigma is not None and not sigma.units.is_dimensionless():
            msg = "Invalid units: sigma coordinate {!r} must be dimensionless.".format(
                sigma.name()
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = (
                    "Incompatible units: {} coordinate {!r} and depth "
                    "coordinate {!r} must have "
                    "the same units.".format(term, coord.name(), depth.name())
                )
                raise ValueError(msg)

    @property
    def dependencies(self):
        """Return a dict mapping from constructor arg names to coordinates.

        Returns a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(sigma=self.sigma, eta=self.eta, depth=self.depth)

    def _derive(self, sigma, eta, depth):
        return eta + sigma * (depth + eta)

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["sigma"],
            nd_points_by_key["eta"],
            nd_points_by_key["depth"],
        )

        bounds = None
        if self.sigma and self.sigma.nbounds:
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            valid_shapes = [(), (1,), (2,)]
            key = "sigma"
            if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                name = self.dependencies[key].name()
                msg = "Invalid bounds for {} coordinate {!r}.".format(key, name)
                raise ValueError(msg)
            valid_shapes.pop()
            for key in ("eta", "depth"):
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = (
                        "The {} coordinate {!r} has bounds. "
                        "These are being disregarded.".format(key, name)
                    )
                    warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)
                    # Swap bounds with points.
                    bds_shape = list(nd_points_by_key[key].shape) + [1]
                    bounds = nd_points_by_key[key].reshape(bds_shape)
                    nd_values_by_key[key] = bounds

            bounds = self._derive(
                nd_values_by_key["sigma"],
                nd_values_by_key["eta"],
                nd_values_by_key["depth"],
            )

        coord = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return coord


class OceanSg1Factory(AuxCoordFactory):
    """Defines an Ocean s-coordinate, generic form 1 factory."""

    def __init__(self, s=None, c=None, eta=None, depth=None, depth_c=None):
        """Create an Ocean s-coordinate, generic form 1 factory with the following formula.

        .. math::
            z(n,k,j,i) = S(k,j,i) + eta(n,j,i) * (1 + S(k,j,i) / depth(j,i))

        where:

        .. math::
            S(k,j,i) = depth_c * s(k) + (depth(j,i) - depth_c) * C(k)

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(s, c, eta, depth, depth_c)
        self.units = depth.units

        self.s = s
        self.c = c
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c

        self.standard_name = "sea_surface_height_above_reference_ellipsoid"
        self.attributes = {"positive": "up"}

    @staticmethod
    def _check_dependencies(s, c, eta, depth, depth_c):
        # Check for sufficient factory coordinates.
        if eta is None or s is None or c is None or depth is None or depth_c is None:
            msg = (
                "Unable to construct Ocean s-coordinate, generic form 1 "
                "factory due to insufficient source coordinates."
            )
            raise ValueError(msg)

        # Check bounds and shape.
        coords = ((s, "s"), (c, "c"))
        for coord, term in coords:
            if coord is not None and coord.nbounds not in (0, 2):
                msg = (
                    "Invalid {} coordinate {!r}: must have either "
                    "0 or 2 bounds.".format(term, coord.name())
                )
                raise ValueError(msg)

        if s and s.nbounds != c.nbounds:
            msg = (
                "The s coordinate {!r} and c coordinate {!r} "
                "must be equally bounded.".format(s.name(), c.name())
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = (
                    "The {} coordinate {!r} has bounds. "
                    "These are being disregarded.".format(term, coord.name())
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        if depth_c is not None and depth_c.shape != (1,):
            msg = "Expected scalar {} coordinate {!r}: got shape {!r}.".format(
                term, coord.name(), coord.shape
            )
            raise ValueError(msg)

        # Check units.
        coords = ((s, "s"), (c, "c"))
        for coord, term in coords:
            if coord is not None and coord.units.is_unknown():
                # Be graceful, and promote unknown to dimensionless units.
                coord.units = cf_units.Unit("1")

            if coord is not None and not coord.units.is_dimensionless():
                msg = (
                    "Invalid units: {} coordinate {!r} "
                    "must be dimensionless.".format(term, coord.name())
                )
                raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"), (depth_c, "depth_c"))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = (
                    "Incompatible units: {} coordinate {!r} and depth "
                    "coordinate {!r} must have "
                    "the same units.".format(term, coord.name(), depth.name())
                )
                raise ValueError(msg)

    @property
    def dependencies(self):
        """Return a dict mapping from constructor arg names to coordinates.

        Return a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(
            s=self.s,
            c=self.c,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
        )

    def _derive(self, s, c, eta, depth, depth_c):
        S = depth_c * s + (depth - depth_c) * c
        return S + eta * (1 + S / depth)

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["s"],
            nd_points_by_key["c"],
            nd_points_by_key["eta"],
            nd_points_by_key["depth"],
            nd_points_by_key["depth_c"],
        )

        bounds = None
        if self.s.nbounds or (self.c and self.c.nbounds):
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            valid_shapes = [(), (1,), (2,)]
            key = "s"
            if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                name = self.dependencies[key].name()
                msg = "Invalid bounds for {} coordinate {!r}.".format(key, name)
                raise ValueError(msg)
            valid_shapes.pop()
            for key in ("eta", "depth", "depth_c"):
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = (
                        "The {} coordinate {!r} has bounds. "
                        "These are being disregarded.".format(key, name)
                    )
                    warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)
                    # Swap bounds with points.
                    bds_shape = list(nd_points_by_key[key].shape) + [1]
                    bounds = nd_points_by_key[key].reshape(bds_shape)
                    nd_values_by_key[key] = bounds

            bounds = self._derive(
                nd_values_by_key["s"],
                nd_values_by_key["c"],
                nd_values_by_key["eta"],
                nd_values_by_key["depth"],
                nd_values_by_key["depth_c"],
            )

        coord = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return coord


class OceanSFactory(AuxCoordFactory):
    """Defines an Ocean s-coordinate factory."""

    def __init__(self, s=None, eta=None, depth=None, a=None, b=None, depth_c=None):
        """Create an Ocean s-coordinate factory with a formula.

        .. math::

            z(n,k,j,i) = eta(n,j,i)*(1+s(k)) + depth_c*s(k) +
                         (depth(j,i)-depth_c)*C(k)

        where:

        .. math::

            C(k) = (1-b) * sinh(a*s(k)) / sinh(a) +
                   b * [tanh(a * (s(k) + 0.5)) / (2 * tanh(0.5*a)) - 0.5]

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(s, eta, depth, a, b, depth_c)
        self.units = depth.units

        self.s = s
        self.eta = eta
        self.depth = depth
        self.a = a
        self.b = b
        self.depth_c = depth_c

        self.standard_name = "sea_surface_height_above_reference_ellipsoid"
        self.attributes = {"positive": "up"}

    @staticmethod
    def _check_dependencies(s, eta, depth, a, b, depth_c):
        # Check for sufficient factory coordinates.
        if (
            eta is None
            or s is None
            or depth is None
            or a is None
            or b is None
            or depth_c is None
        ):
            msg = (
                "Unable to construct Ocean s-coordinate "
                "factory due to insufficient source coordinates."
            )
            raise ValueError(msg)

        # Check bounds and shape.
        if s is not None and s.nbounds not in (0, 2):
            msg = "Invalid s coordinate {!r}: must have either 0 or 2 bounds.".format(
                s.name()
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = (
                    "The {} coordinate {!r} has bounds. "
                    "These are being disregarded.".format(term, coord.name())
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        coords = ((a, "a"), (b, "b"), (depth_c, "depth_c"))
        for coord, term in coords:
            if coord is not None and coord.shape != (1,):
                msg = "Expected scalar {} coordinate {!r}: got shape {!r}.".format(
                    term, coord.name(), coord.shape
                )
                raise ValueError(msg)

        # Check units.
        if s is not None and s.units.is_unknown():
            # Be graceful, and promote unknown to dimensionless units.
            s.units = cf_units.Unit("1")

        if s is not None and not s.units.is_dimensionless():
            msg = "Invalid units: s coordinate {!r} must be dimensionless.".format(
                s.name()
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"), (depth_c, "depth_c"))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = (
                    "Incompatible units: {} coordinate {!r} and depth "
                    "coordinate {!r} must have "
                    "the same units.".format(term, coord.name(), depth.name())
                )
                raise ValueError(msg)

    @property
    def dependencies(self):
        """Return a dict mapping from constructor arg names to coordinates.

        Return a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(
            s=self.s,
            eta=self.eta,
            depth=self.depth,
            a=self.a,
            b=self.b,
            depth_c=self.depth_c,
        )

    def _derive(self, s, eta, depth, a, b, depth_c):
        c = (1 - b) * da.sinh(a * s) / da.sinh(a) + b * (
            da.tanh(a * (s + 0.5)) / (2 * da.tanh(0.5 * a)) - 0.5
        )
        return eta * (1 + s) + depth_c * s + (depth - depth_c) * c

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["s"],
            nd_points_by_key["eta"],
            nd_points_by_key["depth"],
            nd_points_by_key["a"],
            nd_points_by_key["b"],
            nd_points_by_key["depth_c"],
        )

        bounds = None
        if self.s.nbounds:
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            valid_shapes = [(), (1,), (2,)]
            key = "s"
            if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                name = self.dependencies[key].name()
                msg = "Invalid bounds for {} coordinate {!r}.".format(key, name)
                raise ValueError(msg)
            valid_shapes.pop()
            for key in ("eta", "depth", "a", "b", "depth_c"):
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = (
                        "The {} coordinate {!r} has bounds. "
                        "These are being disregarded.".format(key, name)
                    )
                    warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)
                    # Swap bounds with points.
                    bds_shape = list(nd_points_by_key[key].shape) + [1]
                    bounds = nd_points_by_key[key].reshape(bds_shape)
                    nd_values_by_key[key] = bounds

            bounds = self._derive(
                nd_values_by_key["s"],
                nd_values_by_key["eta"],
                nd_values_by_key["depth"],
                nd_values_by_key["a"],
                nd_values_by_key["b"],
                nd_values_by_key["depth_c"],
            )

        coord = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return coord


class OceanSg2Factory(AuxCoordFactory):
    """Defines an Ocean s-coordinate, generic form 2 factory."""

    def __init__(self, s=None, c=None, eta=None, depth=None, depth_c=None):
        """Create an Ocean s-coordinate, generic form 2 factory with the following formula.

        .. math::
            z(n,k,j,i) = eta(n,j,i) + (eta(n,j,i) + depth(j,i)) * S(k,j,i)

        where:

        .. math::
            S(k,j,i) = (depth_c * s(k) + depth(j,i) * C(k)) /
                       (depth_c + depth(j,i))

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CoordMetadata)
        super().__init__()

        # Check that provided coordinates meet necessary conditions.
        self._check_dependencies(s, c, eta, depth, depth_c)
        self.units = depth.units

        self.s = s
        self.c = c
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c

        self.standard_name = "sea_surface_height_above_reference_ellipsoid"
        self.attributes = {"positive": "up"}

    @staticmethod
    def _check_dependencies(s, c, eta, depth, depth_c):
        # Check for sufficient factory coordinates.
        if eta is None or s is None or c is None or depth is None or depth_c is None:
            msg = (
                "Unable to construct Ocean s-coordinate, generic form 2 "
                "factory due to insufficient source coordinates."
            )
            raise ValueError(msg)

        # Check bounds and shape.
        coords = ((s, "s"), (c, "c"))
        for coord, term in coords:
            if coord is not None and coord.nbounds not in (0, 2):
                msg = (
                    "Invalid {} coordinate {!r}: must have either "
                    "0 or 2 bounds.".format(term, coord.name())
                )
                raise ValueError(msg)

        if s and s.nbounds != c.nbounds:
            msg = (
                "The s coordinate {!r} and c coordinate {!r} "
                "must be equally bounded.".format(s.name(), c.name())
            )
            raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"))
        for coord, term in coords:
            if coord is not None and coord.nbounds:
                msg = (
                    "The {} coordinate {!r} has bounds. "
                    "These are being disregarded.".format(term, coord.name())
                )
                warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)

        if depth_c is not None and depth_c.shape != (1,):
            msg = "Expected scalar depth_c coordinate {!r}: got shape {!r}.".format(
                depth_c.name(), depth_c.shape
            )
            raise ValueError(msg)

        # Check units.
        coords = ((s, "s"), (c, "c"))
        for coord, term in coords:
            if coord is not None and coord.units.is_unknown():
                # Be graceful, and promote unknown to dimensionless units.
                coord.units = cf_units.Unit("1")

            if coord is not None and not coord.units.is_dimensionless():
                msg = (
                    "Invalid units: {} coordinate {!r} "
                    "must be dimensionless.".format(term, coord.name())
                )
                raise ValueError(msg)

        coords = ((eta, "eta"), (depth, "depth"), (depth_c, "depth_c"))
        for coord, term in coords:
            if coord is not None and coord.units != depth.units:
                msg = (
                    "Incompatible units: {} coordinate {!r} and depth "
                    "coordinate {!r} must have "
                    "the same units.".format(term, coord.name(), depth.name())
                )
                raise ValueError(msg)

    @property
    def dependencies(self):
        """Return a dicti mapping from constructor arg names to coordinates.

        Return a dictionary mapping from constructor argument names to
        the corresponding coordinates.

        """
        return dict(
            s=self.s,
            c=self.c,
            eta=self.eta,
            depth=self.depth,
            depth_c=self.depth_c,
        )

    def _derive(self, s, c, eta, depth, depth_c):
        S = (depth_c * s + depth * c) / (depth_c + depth)
        return eta + (eta + depth) * S

    def make_coord(self, coord_dims_func):
        """Return a new :class:`iris.coords.AuxCoord` as defined by this factory.

        Parameters
        ----------
        coord_dims_func :
            A callable which can return the list of dimensions relevant
            to a given coordinate. See :meth:`iris.cube.Cube.coord_dims()`.

        """
        # Determine the relevant dimensions.
        derived_dims = self.derived_dims(coord_dims_func)
        dependency_dims = self._dependency_dims(coord_dims_func)

        # Build the points array.
        nd_points_by_key = self._remap(dependency_dims, derived_dims)
        points = self._derive(
            nd_points_by_key["s"],
            nd_points_by_key["c"],
            nd_points_by_key["eta"],
            nd_points_by_key["depth"],
            nd_points_by_key["depth_c"],
        )

        bounds = None
        if self.s.nbounds or (self.c and self.c.nbounds):
            # Build the bounds array.
            nd_values_by_key = self._remap_with_bounds(dependency_dims, derived_dims)
            valid_shapes = [(), (1,), (2,)]
            key = "s"
            if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                name = self.dependencies[key].name()
                msg = "Invalid bounds for {} coordinate {!r}.".format(key, name)
                raise ValueError(msg)
            valid_shapes.pop()
            for key in ("eta", "depth", "depth_c"):
                if nd_values_by_key[key].shape[-1:] not in valid_shapes:
                    name = self.dependencies[key].name()
                    msg = (
                        "The {} coordinate {!r} has bounds. "
                        "These are being disregarded.".format(key, name)
                    )
                    warnings.warn(msg, category=IrisIgnoringBoundsWarning, stacklevel=2)
                    # Swap bounds with points.
                    bds_shape = list(nd_points_by_key[key].shape) + [1]
                    bounds = nd_points_by_key[key].reshape(bds_shape)
                    nd_values_by_key[key] = bounds

            bounds = self._derive(
                nd_values_by_key["s"],
                nd_values_by_key["c"],
                nd_values_by_key["eta"],
                nd_values_by_key["depth"],
                nd_values_by_key["depth_c"],
            )

        coord = iris.coords.AuxCoord(
            points,
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            bounds=bounds,
            attributes=self.attributes,
            coord_system=self.coord_system,
        )
        return coord

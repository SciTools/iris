# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Generalised mechanisms for metadata translation and cube construction.

"""

import collections
import warnings

import cf_units

from iris.analysis import Linear
import iris.cube
import iris.exceptions
import iris.fileformats.um_cf_map

Factory = collections.namedtuple("Factory", ["factory_class", "args"])
ReferenceTarget = collections.namedtuple(
    "ReferenceTarget", ("name", "transform")
)


class ConcreteReferenceTarget:
    """Everything you need to make a real Cube for a named reference."""

    def __init__(self, name, transform=None):
        #: The name used to connect references with references.
        self.name = name
        #: An optional transformation to apply to the cubes.
        self.transform = transform
        self._src_cubes = iris.cube.CubeList()
        self._final_cube = None

    def add_cube(self, cube):
        self._src_cubes.append(cube)

    def as_cube(self):
        if self._final_cube is None:
            src_cubes = self._src_cubes
            if len(src_cubes) > 1:
                # Merge the reference cubes to allow for
                # time-varying surface pressure in hybrid-presure.
                src_cubes = src_cubes.merge(unique=False)
                if len(src_cubes) > 1:
                    warnings.warn(
                        "Multiple reference cubes for {}".format(self.name)
                    )
            src_cube = src_cubes[-1]

            if self.transform is None:
                self._final_cube = src_cube
            else:
                final_cube = src_cube.copy()
                attributes = self.transform(final_cube)
                for name, value in attributes.items():
                    setattr(final_cube, name, value)
                self._final_cube = final_cube

        return self._final_cube


class Reference(iris.util._OrderedHashable):
    _names = ("name",)
    """
    A named placeholder for inter-field references.

    """


def scalar_coord(cube, coord_name):
    """Try to find a single-valued coord with the given name."""
    found_coord = None
    for coord in cube.coords(coord_name):
        if coord.shape == (1,):
            found_coord = coord
            break
    return found_coord


def vector_coord(cube, coord_name):
    """Try to find a one-dimensional, multi-valued coord with the given name."""
    found_coord = None
    for coord in cube.coords(coord_name):
        if len(coord.shape) == 1 and coord.shape[0] > 1:
            found_coord = coord
            break
    return found_coord


def scalar_cell_method(cube, method, coord_name):
    """Try to find the given type of cell method over a single coord with the given name."""
    found_cell_method = None
    for cell_method in cube.cell_methods:
        if cell_method.method == method and len(cell_method.coord_names) == 1:
            name = cell_method.coord_names[0]
            if name == coord_name:
                coords = cube.coords(name)
                if len(coords) == 1:
                    found_cell_method = cell_method
    return found_cell_method


def has_aux_factory(cube, aux_factory_class):
    """
    Try to find an class:`~iris.aux_factory.AuxCoordFactory` instance of the
    specified type on the cube.

    """
    for factory in cube.aux_factories:
        if isinstance(factory, aux_factory_class):
            return True
    return False


def aux_factory(cube, aux_factory_class):
    """
    Return the class:`~iris.aux_factory.AuxCoordFactory` instance of the
    specified type from a cube.

    """
    aux_factories = [
        aux_factory
        for aux_factory in cube.aux_factories
        if isinstance(aux_factory, aux_factory_class)
    ]
    if not aux_factories:
        raise ValueError(
            "Cube does not have an aux factory of "
            "type {!r}.".format(aux_factory_class)
        )
    elif len(aux_factories) > 1:
        raise ValueError(
            "Cube has more than one aux factory of "
            "type {!r}.".format(aux_factory_class)
        )
    return aux_factories[0]


class _ReferenceError(Exception):
    """Signals an invalid/missing reference field."""

    pass


def _dereference_args(factory, reference_targets, regrid_cache, cube):
    """Converts all the arguments for a factory into concrete coordinates."""
    args = []
    for arg in factory.args:
        if isinstance(arg, Reference):
            if arg.name in reference_targets:
                src = reference_targets[arg.name].as_cube()
                # If necessary, regrid the reference cube to
                # match the grid of this cube.
                src = _ensure_aligned(regrid_cache, src, cube)
                if src is not None:
                    new_coord = iris.coords.AuxCoord(
                        src.data,
                        src.standard_name,
                        src.long_name,
                        src.var_name,
                        src.units,
                        attributes=src.attributes,
                    )
                    dims = [
                        cube.coord_dims(src_coord)[0]
                        for src_coord in src.dim_coords
                    ]
                    cube.add_aux_coord(new_coord, dims)
                    args.append(new_coord)
                else:
                    raise _ReferenceError(
                        "Unable to regrid reference for"
                        " {!r}".format(arg.name)
                    )
            else:
                raise _ReferenceError(
                    "The source data contains no "
                    "field(s) for {!r}.".format(arg.name)
                )
        else:
            # If it wasn't a Reference, then arg is a dictionary
            # of keyword arguments for cube.coord(...).
            args.append(cube.coord(**arg))
    return args


def _regrid_to_target(src_cube, target_coords, target_cube):
    # Interpolate onto the target grid.
    sample_points = [(coord, coord.points) for coord in target_coords]
    result_cube = src_cube.interpolate(sample_points, Linear())

    # Any scalar coords on the target_cube will have become vector
    # coords on the resample src_cube (i.e. result_cube).
    # These unwanted vector coords need to be pushed back to scalars.
    index = [slice(None, None)] * result_cube.ndim
    for target_coord in target_coords:
        if not target_cube.coord_dims(target_coord):
            result_dim = result_cube.coord_dims(target_coord)[0]
            index[result_dim] = 0
    if not all(key == slice(None, None) for key in index):
        result_cube = result_cube[tuple(index)]
    return result_cube


def _ensure_aligned(regrid_cache, src_cube, target_cube):
    """
    Returns a version of `src_cube` suitable for use as an AuxCoord
    on `target_cube`, or None if no version can be made.

    """
    result_cube = None

    # Check that each of src_cube's dim_coords matches up with a single
    # coord on target_cube.
    try:
        target_coords = []
        for dim_coord in src_cube.dim_coords:
            target_coords.append(target_cube.coord(dim_coord))
    except iris.exceptions.CoordinateNotFoundError:
        # One of the src_cube's dim_coords didn't exist on the
        # target_cube... so we can't regrid (i.e. just return None).
        pass
    else:
        # So we can use `iris.analysis.interpolate.linear()` later,
        # ensure each target coord is either a scalar or maps to a
        # single, distinct dimension.
        target_dims = [
            target_cube.coord_dims(coord) for coord in target_coords
        ]
        target_dims = list(filter(None, target_dims))
        unique_dims = set()
        for dims in target_dims:
            unique_dims.update(dims)
        compatible = len(target_dims) == len(unique_dims)

        if compatible:
            cache_key = id(src_cube)
            if cache_key not in regrid_cache:
                regrid_cache[cache_key] = ([src_cube.dim_coords], [src_cube])
            grids, cubes = regrid_cache[cache_key]
            # 'grids' is a list of tuples of coordinates, so convert
            # the 'target_coords' list into a tuple to be consistent.
            target_coords = tuple(target_coords)
            try:
                # Look for this set of target coordinates in the cache.
                i = grids.index(target_coords)
                result_cube = cubes[i]
            except ValueError:
                # Not already cached, so do the hard work of interpolating.
                result_cube = _regrid_to_target(
                    src_cube, target_coords, target_cube
                )
                # Add it to the cache.
                grids.append(target_coords)
                cubes.append(result_cube)

    return result_cube


_loader_attrs = ("field_generator", "field_generator_kwargs", "converter")


class Loader(collections.namedtuple("Loader", _loader_attrs)):
    def __new__(cls, field_generator, field_generator_kwargs, converter):
        """
        Create a definition of a field-based Cube loader.

        Args:

        * field_generator
            A callable that accepts a filename as its first argument and
            returns an iterable of field objects.

        * field_generator_kwargs
            Additional arguments to be passed to the field_generator.

        * converter
            A callable that converts a field object into a Cube.

        """
        return tuple.__new__(
            cls, (field_generator, field_generator_kwargs, converter)
        )


ConversionMetadata = collections.namedtuple(
    "ConversionMetadata",
    (
        "factories",
        "references",
        "standard_name",
        "long_name",
        "units",
        "attributes",
        "cell_methods",
        "dim_coords_and_dims",
        "aux_coords_and_dims",
    ),
)


def _make_cube(field, converter):
    # Convert the field to a Cube.
    metadata = converter(field)

    cube_data = field.core_data()
    cube = iris.cube.Cube(
        cube_data,
        attributes=metadata.attributes,
        cell_methods=metadata.cell_methods,
        dim_coords_and_dims=metadata.dim_coords_and_dims,
        aux_coords_and_dims=metadata.aux_coords_and_dims,
    )

    # Temporary code to deal with invalid standard names in the
    # translation table.
    if metadata.standard_name is not None:
        cube.rename(metadata.standard_name)
    if metadata.long_name is not None:
        cube.long_name = metadata.long_name
    if metadata.units is not None:
        # Temporary code to deal with invalid units in the translation
        # table.
        try:
            cube.units = metadata.units
        except ValueError:
            msg = "Ignoring PP invalid units {!r}".format(metadata.units)
            warnings.warn(msg)
            cube.attributes["invalid_units"] = metadata.units
            cube.units = cf_units._UNKNOWN_UNIT_STRING

    return cube, metadata.factories, metadata.references


def _resolve_factory_references(
    cube, factories, concrete_reference_targets, regrid_cache={}
):
    # Attach the factories for a cube, building them from references.
    # Note: the regrid_cache argument lets us share and reuse regridded data
    # across multiple result cubes.
    for factory in factories:
        try:
            args = _dereference_args(
                factory, concrete_reference_targets, regrid_cache, cube
            )
        except _ReferenceError as e:
            msg = "Unable to create instance of {factory}. " + str(e)
            factory_name = factory.factory_class.__name__
            warnings.warn(msg.format(factory=factory_name))
        else:
            aux_factory = factory.factory_class(*args)
            cube.add_aux_factory(aux_factory)


def _load_pairs_from_fields_and_filenames(
    fields_and_filenames, converter, user_callback_wrapper=None
):
    # The underlying mechanism for the public 'load_pairs_from_fields' and
    # 'load_cubes'.
    # Slightly more complicated than 'load_pairs_from_fields', only because it
    # needs a filename associated with each field to support the load callback.
    concrete_reference_targets = {}
    results_needing_reference = []
    for field, filename in fields_and_filenames:
        # Convert the field to a Cube, passing down the 'converter' function.
        cube, factories, references = _make_cube(field, converter)

        # Post modify the new cube with a user-callback.
        # This is an ordinary Iris load callback, so it takes the filename.
        cube = iris.io.run_callback(
            user_callback_wrapper, cube, field, filename
        )
        # Callback mechanism may return None, which must not be yielded.
        if cube is None:
            continue

        # Cross referencing.
        for reference in references:
            name = reference.name
            # Register this cube as a source cube for the named reference.
            target = concrete_reference_targets.get(name)
            if target is None:
                target = ConcreteReferenceTarget(name, reference.transform)
                concrete_reference_targets[name] = target
            target.add_cube(cube)

        if factories:
            results_needing_reference.append((cube, factories, field))
        else:
            yield (cube, field)

    regrid_cache = {}
    for cube, factories, field in results_needing_reference:
        _resolve_factory_references(
            cube, factories, concrete_reference_targets, regrid_cache
        )
        yield (cube, field)


def load_pairs_from_fields(fields, converter):
    """
    Convert an iterable of fields into an iterable of Cubes using the
    provided converter.

    Args:

    * fields:
        An iterable of fields.

    * converter:
        An Iris converter function, suitable for use with the supplied fields.
        See the description in :class:`iris.fileformats.rules.Loader`.

    Returns:
        An iterable of (:class:`iris.cube.Cube`, field) pairs.

    """
    return _load_pairs_from_fields_and_filenames(
        ((field, None) for field in fields), converter
    )


def load_cubes(filenames, user_callback, loader, filter_function=None):
    if isinstance(filenames, str):
        filenames = [filenames]

    def _generate_all_fields_and_filenames():
        for filename in filenames:
            for field in loader.field_generator(
                filename, **loader.field_generator_kwargs
            ):
                # evaluate field against format specific desired attributes
                # load if no format specific desired attributes are violated
                if filter_function is None or filter_function(field):
                    yield (field, filename)

    def loadcubes_user_callback_wrapper(cube, field, filename):
        # Run user-provided original callback function.
        result = cube
        if user_callback is not None:
            result = user_callback(cube, field, filename)
        return result

    all_fields_and_filenames = _generate_all_fields_and_filenames()
    for cube, field in _load_pairs_from_fields_and_filenames(
        all_fields_and_filenames,
        converter=loader.converter,
        user_callback_wrapper=loadcubes_user_callback_wrapper,
    ):
        yield cube

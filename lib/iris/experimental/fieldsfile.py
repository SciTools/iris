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
High-speed loading of structured FieldsFiles.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import range

from iris.coords import DimCoord
from iris.cube import CubeList
from iris.exceptions import TranslationError
from iris.fileformats.ff import FF2PP
from iris.fileformats.pp_rules import (_convert_time_coords,
                                       _convert_vertical_coords,
                                       _convert_scalar_realization_coords,
                                       _convert_scalar_pseudo_level_coords,
                                       _all_other_rules)
from iris.fileformats.rules import ConversionMetadata, Loader, load_cubes
from iris.fileformats.um._fast_load_structured_fields import \
    group_structured_fields


# Seed the preferred order of candidate dimension coordinates.
_HINT_COORDS = ['time', 'forecast_reference_time', 'model_level_number']
_HINTS = {name: i for i, name in zip(range(len(_HINT_COORDS)), _HINT_COORDS)}


def _collations_from_filename(filename):
    fields = iter(FF2PP(filename))
    return group_structured_fields(fields)


def load(filenames, callback=None):
    """
    Load structured FieldsFiles.

    Args:

    * filenames:
        One or more filenames.


    Kwargs:

    * callback:
        A modifier/filter function. Please see the module documentation
        for :mod:`iris`.

        .. note::

            Unlike the standard :func:`iris.load` operation, the callback is
            applied to the final result cubes, not individual input fields.

    Returns:
        An :class:`iris.cube.CubeList`.


    This is a streamlined load operation, to be used only on fieldsfiles whose
    fields repeat regularly over the same vertical levels and times.
    The results are equivalent to those from :func:`iris.load`, but operation
    is substantially faster for suitable input.

    The input files should conform to the following requirements:

    *  the file must contain fields for all possible combinations of the
       vertical levels and time points found in the file.

    *  the fields must occur in a regular repeating order within the file.

       (For example: a sequence of fields for NV vertical levels, repeated
       for NP different forecast periods, repeated for NT different forecast
       times).

    *  all other metadata must be identical across all fields of the same
       phenomenon.

    Each group of fields with the same values of LBUSER4, LBUSER7 and LBPROC
    is identified as a separate phenomenon:  These groups are processed
    independently and returned as separate result cubes.

    .. note::

        Each input file is loaded independently.  Thus a single result cube can
        not combine data from multiple input files.

    .. note::

        The resulting time-related cordinates ('time', 'forecast_time' and
        'forecast_period') may be mapped to shared cube dimensions and in some
        cases can also be multidimensional.  However, the vertical level
        information *must* have a simple one-dimensional structure, independent
        of the time points, otherwise an error will be raised.

    .. note::

        Where input data does *not* have a fully regular arrangement, the
        corresponding result cube will have a single anonymous extra dimension
        which indexes over all the input fields.

        This can happen if, for example, some fields are missing; or have
        slightly different metadata; or appear out of order in the file.

    .. warning::

        Any non-regular metadata variation in the input should be strictly
        avoided, as not all irregularities are detected, which can cause
        erroneous results.

    """
    loader = Loader(_collations_from_filename, {}, _convert_collation, None)
    return CubeList(load_cubes(filenames, callback, loader, None))


def _adjust_dims(coords_and_dims, n_dims):
    def adjust(dims):
        if dims is not None:
            dims += n_dims
        return dims
    return [(coord, adjust(dims)) for coord, dims in coords_and_dims]


def _bind_coords(coords_and_dims, dim_coord_dims, dim_coords_and_dims,
                 aux_coords_and_dims):
    key_func = lambda item: _HINTS.get(item[0].name(), len(_HINTS))
    # Target the first DimCoord for a dimension at dim_coords,
    # and target everything else at aux_coords.
    for coord, dims in sorted(coords_and_dims, key=key_func):
        if (isinstance(coord, DimCoord) and dims is not None and
                len(dims) == 1 and dims[0] not in dim_coord_dims):
            dim_coords_and_dims.append((coord, dims))
            dim_coord_dims.add(dims[0])
        else:
            aux_coords_and_dims.append((coord, dims))


def _convert_collation(collation):
    """
    Converts a FieldCollation into the corresponding items of Cube
    metadata.

    Args:

    * collation:
        A FieldCollation object.

    Returns:
        A :class:`iris.fileformats.rules.ConversionMetadata` object.

    """
    # For all the scalar conversions all fields in the collation will
    # give the same result, so the choice is arbitrary.
    field = collation.fields[0]

    # All the "other" rules.
    (references, standard_name, long_name, units, attributes, cell_methods,
     dim_coords_and_dims, aux_coords_and_dims) = _all_other_rules(field)

    # Adjust any dimension bindings to account for the extra leading
    # dimensions added by the collation.
    if collation.vector_dims_shape:
        n_collation_dims = len(collation.vector_dims_shape)
        dim_coords_and_dims = _adjust_dims(dim_coords_and_dims,
                                           n_collation_dims)
        aux_coords_and_dims = _adjust_dims(aux_coords_and_dims,
                                           n_collation_dims)

    # "Normal" (non-cross-sectional) time values
    vector_headers = collation.element_arrays_and_dims
    # If the collation doesn't define a vector of values for a
    # particular header then it must be constant over all fields in the
    # collation. In which case it's safe to get the value from any field.
    t1, t1_dims = vector_headers.get('t1', (field.t1, ()))
    t2, t2_dims = vector_headers.get('t2', (field.t2, ()))
    lbft, lbft_dims = vector_headers.get('lbft', (field.lbft, ()))
    coords_and_dims = _convert_time_coords(field.lbcode, field.lbtim,
                                           field.time_unit('hours'),
                                           t1, t2, lbft,
                                           t1_dims, t2_dims, lbft_dims)
    dim_coord_dims = set()
    _bind_coords(coords_and_dims, dim_coord_dims, dim_coords_and_dims,
                 aux_coords_and_dims)

    # "Normal" (non-cross-sectional) vertical levels
    blev, blev_dims = vector_headers.get('blev', (field.blev, ()))
    lblev, lblev_dims = vector_headers.get('lblev', (field.lblev, ()))
    bhlev, bhlev_dims = vector_headers.get('bhlev', (field.bhlev, ()))
    bhrlev, bhrlev_dims = vector_headers.get('bhrlev', (field.bhrlev, ()))
    brsvd1, brsvd1_dims = vector_headers.get('brsvd1', (field.brsvd[0], ()))
    brsvd2, brsvd2_dims = vector_headers.get('brsvd2', (field.brsvd[1], ()))
    brlev, brlev_dims = vector_headers.get('brlev', (field.brlev, ()))
    # Find all the non-trivial dimension values
    dims = set(filter(None, [blev_dims, lblev_dims, bhlev_dims, bhrlev_dims,
                             brsvd1_dims, brsvd2_dims, brlev_dims]))
    if len(dims) > 1:
        raise TranslationError('Unsupported multiple values for vertical '
                               'dimension.')
    if dims:
        v_dims = dims.pop()
        if len(v_dims) > 1:
            raise TranslationError('Unsupported multi-dimension vertical '
                                   'headers.')
    else:
        v_dims = ()
    coords_and_dims, factories = _convert_vertical_coords(field.lbcode,
                                                          field.lbvc,
                                                          blev, lblev,
                                                          field.stash,
                                                          bhlev, bhrlev,
                                                          brsvd1, brsvd2,
                                                          brlev, v_dims)
    _bind_coords(coords_and_dims, dim_coord_dims, dim_coords_and_dims,
                 aux_coords_and_dims)

    # Realization (aka ensemble) (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_realization_coords(
        lbrsvd4=field.lbrsvd[3]))

    # Pseudo-level coordinate (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_pseudo_level_coords(
        lbuser5=field.lbuser[4]))

    return ConversionMetadata(factories, references, standard_name, long_name,
                              units, attributes, cell_methods,
                              dim_coords_and_dims, aux_coords_and_dims)

# (C) British Crown Copyright 2014, Met Office
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

from iris.coords import DimCoord
from iris.exceptions import TranslationError
from iris.fileformats.pp_rules import (_convert_time_coords,
                                       _convert_vertical_coords,
                                       _convert_scalar_realization_coords,
                                       _convert_scalar_pseudo_level_coords,
                                       _all_other_rules)
from iris.fileformats.rules import ConversionMetadata


def _adjust_dims(coords_and_dims, n_dims):
    def adjust(dims):
        if dims is not None:
            dims += n_dims
        return dims
    return [(coord, adjust(dims)) for coord, dims in coords_and_dims]


def convert_collation(collation):
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
    for coord, dims in coords_and_dims:
        if dims is not None and len(dims) == 1:
            dim_coords_and_dims.append((coord, dims))
        else:
            aux_coords_and_dims.append((coord, dims))

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
    else:
        v_dims = ()
    coords_and_dims, factories = _convert_vertical_coords(field.lbcode,
                                                          field.lbvc,
                                                          blev, lblev,
                                                          field.stash,
                                                          bhlev, bhrlev,
                                                          brsvd1, brsvd2,
                                                          brlev, v_dims)
    done = False
    for coord, dims in coords_and_dims:
        # Assumes the first DimCoord, if any, should be used as *the*
        # dimension coordinate.
        if not done and isinstance(coord, DimCoord):
            dim_coords_and_dims.append((coord, dims))
            done = True
        else:
            aux_coords_and_dims.append((coord, dims))

    # Realization (aka ensemble) (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_realization_coords(
        lbrsvd4=field.lbrsvd[3]))

    # Pseudo-level coordinate (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_pseudo_level_coords(
        lbuser5=field.lbuser[4]))

    return ConversionMetadata(factories, references, standard_name, long_name,
                              units, attributes, cell_methods,
                              dim_coords_and_dims, aux_coords_and_dims)

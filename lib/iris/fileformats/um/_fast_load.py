# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Support for "fast" loading of structured UM files in iris load functions.

i.e. :meth:`iris.load` and its associates.

This provides a context manager to enable structured loading via all the iris
load function variants.  It is made public in :mod:`iris.fileformats.um`.

This works with either PP or Fieldsfiles.
The various existing PP and FF format loaders all call into
:meth:`iris.fileformats.pp._load_cubes_variable_loader`.
When enabled, that function calls private functions in this module, to replace
the 'normal' calls with 'structured' loading.

At present, there is *no* public low-level fields-to-cube interface, equivalent
to the pp "as_pairs" functions.

"""

from contextlib import contextmanager
import os.path
import threading

import numpy as np

# Be minimal about what we import from iris, to avoid circular imports.
# Below, other parts of iris.fileformats are accessed via deferred imports.
from iris.coords import DimCoord
from iris.cube import CubeList
from iris.exceptions import TranslationError
from iris.fileformats.um._fast_load_structured_fields import (
    BasicFieldCollation,
    group_structured_fields,
)

# Strings to identify the PP and FF file format handler specs.
_FF_SPEC_NAME = "UM Fieldsfile"
_PP_SPEC_NAME = "UM Post Processing file"


class FieldCollation(BasicFieldCollation):
    # This class specialises the BasicFieldCollation by adding the file-index
    # and file-path concepts.
    # This preserves the more abstract scope of the original 'FieldCollation'
    # class, now renamed 'BasicFieldCollation'.

    def __init__(self, fields, filepath):
        """FieldCollation initialise.

        Parameters
        ----------
        fields : iterable of :class:`iris.fileformats.pp.PPField`
            The fields in the collation.
        filepath : str
            The path of the file the collation is loaded from.

        """
        super().__init__(fields)
        self._load_filepath = filepath

    @property
    def data_filepath(self):
        return self._load_filepath

    @property
    def data_field_indices(self):
        """Field indices of the contained PPFields in the input file.

        This records the original file location of the individual data fields
        contained, within the input datafile.

        Returns
        -------
        An integer array of shape `self.vector_dims_shape`.

        """
        # Get shape :  N.B. this calculates (and caches) the structure.
        vector_dims_shape = self.vector_dims_shape
        # Get index-in-file of each contained field.
        indices = np.array(
            [field._index_in_structured_load_file for field in self._fields],
            dtype=np.int64,
        )
        return indices.reshape(vector_dims_shape)


def _basic_load_function(filename, pp_filter=None, **kwargs):
    # The low-level 'fields from filename' loader.
    #
    # This is the 'loader.generator' in the control structure passed to the
    # generic rules code, :meth:`iris.fileformats.rules.load_cubes`.
    #
    # As called by generic rules code, this replaces pp.load (and the like).
    # It yields a sequence of "fields", but in this case the 'fields' are
    # :class:`iris.fileformats.um._fast_load_structured_fields.FieldCollation`
    # objects.
    #
    # NOTE: so, a FieldCollation is passed as the 'field' in user callbacks.
    #
    # Also in our case, we need to support the basic single-field filtering
    # operation that speeds up phenomenon selection.
    # Therefore, the actual loader will pass this as the 'pp_filter' keyword,
    # when it is present.
    # Additional load keywords are 'passed on' to the lower-level function.

    # Helper function to select the correct fields loader call.
    def _select_raw_fields_loader(fname):
        # Return the PPfield loading function for a file name.
        #
        # This decides whether the underlying file is an FF or PP file.
        # Because it would be too awkward to modify the whole iris loading
        # callchain to "pass down" the file format, this function instead
        # 'recreates' that information by calling the format picker again.
        # NOTE: this may be inefficient, especially for web resources.
        from iris.fileformats import FORMAT_AGENT
        from iris.fileformats.pp import load as pp_load
        from iris.fileformats.um import um_to_pp

        with open(fname, "rb") as fh:
            spec = FORMAT_AGENT.get_spec(os.path.basename(fname), fh)
        if spec.name.startswith(_FF_SPEC_NAME):
            loader = um_to_pp
        elif spec.name.startswith(_PP_SPEC_NAME):
            loader = pp_load
        else:
            emsg = "Require {!r} to be a structured FieldsFile or a PP file."
            raise ValueError(emsg.format(fname))
        return loader

    loader = _select_raw_fields_loader(filename)

    def iter_fields_decorated_with_load_indices(fields_iter):
        for i_field, field in enumerate(fields_iter):
            field._index_in_structured_load_file = i_field
            yield field

    fields = iter_fields_decorated_with_load_indices(
        field
        for field in loader(filename, **kwargs)
        if pp_filter is None or pp_filter(field)
    )

    return group_structured_fields(
        fields, collation_class=FieldCollation, filepath=filename
    )


# Define the preferred order of candidate dimension coordinates, as used by
# _convert_collation.
_HINT_COORDS = ["time", "forecast_reference_time", "model_level_number"]
_HINTS = {name: i for i, name in zip(range(len(_HINT_COORDS)), _HINT_COORDS)}


def _convert_collation(collation):
    """Convert a FieldCollation into the corresponding items of Cube metadata.

    Parameters
    ----------
    collation :
        A FieldCollation object.

    Returns
    -------
    A :class:`iris.fileformats.rules.ConversionMetadata` object.

    Notes
    -----
    .. note:

        This is the 'loader.converter', in the control structure passed to the
        generic rules code, :meth:`iris.fileformats.rules.load_cubes`.

    """
    from iris.fileformats.pp_load_rules import (
        _all_other_rules,
        _convert_scalar_pseudo_level_coords,
        _convert_scalar_realization_coords,
        _convert_time_coords,
        _convert_vertical_coords,
    )
    from iris.fileformats.rules import ConversionMetadata

    # For all the scalar conversions, all fields in the collation will
    # give the same result, so the choice is arbitrary.
    field = collation.fields[0]

    # Call "all other" rules.
    (
        references,
        standard_name,
        long_name,
        units,
        attributes,
        cell_methods,
        dim_coords_and_dims,
        aux_coords_and_dims,
    ) = _all_other_rules(field)

    # Adjust any dimension bindings to account for the extra leading
    # dimensions added by the collation.
    if collation.vector_dims_shape:

        def _adjust_dims(coords_and_dims, n_dims):
            def adjust(dims):
                if dims is not None:
                    dims += n_dims
                return dims

            return [(coord, adjust(dims)) for coord, dims in coords_and_dims]

        n_collation_dims = len(collation.vector_dims_shape)
        dim_coords_and_dims = _adjust_dims(dim_coords_and_dims, n_collation_dims)
        aux_coords_and_dims = _adjust_dims(aux_coords_and_dims, n_collation_dims)

    # Dimensions to which we've already assigned dimension coordinates.
    dim_coord_dims = set()

    # Helper call to choose which coords are dimensions and which auxiliary.
    def _bind_coords(
        coords_and_dims,
        dim_coord_dims,
        dim_coords_and_dims,
        aux_coords_and_dims,
    ):
        def key_func(item):
            return _HINTS.get(item[0].name(), len(_HINTS))

        # Target the first DimCoord for a dimension at dim_coords,
        # and target everything else at aux_coords.
        for coord, dims in sorted(coords_and_dims, key=key_func):
            if (
                isinstance(coord, DimCoord)
                and dims is not None
                and len(dims) == 1
                and dims[0] not in dim_coord_dims
            ):
                dim_coords_and_dims.append((coord, dims))
                dim_coord_dims.add(dims[0])
            else:
                aux_coords_and_dims.append((coord, dims))

    # Call "time" rules.
    #
    # For "normal" (non-cross-sectional) time values.
    vector_headers = collation.element_arrays_and_dims
    # If the collation doesn't define a vector of values for a
    # particular header then it must be constant over all fields in the
    # collation. In which case it's safe to get the value from any field.
    t1, t1_dims = vector_headers.get("t1", (field.t1, ()))
    t2, t2_dims = vector_headers.get("t2", (field.t2, ()))
    lbft, lbft_dims = vector_headers.get("lbft", (field.lbft, ()))
    coords_and_dims = _convert_time_coords(
        field.lbcode,
        field.lbtim,
        field.time_unit("hours"),
        t1,
        t2,
        lbft,
        t1_dims,
        t2_dims,
        lbft_dims,
    )
    # Bind resulting coordinates to dimensions, where suitable.
    _bind_coords(
        coords_and_dims,
        dim_coord_dims,
        dim_coords_and_dims,
        aux_coords_and_dims,
    )

    # Call "vertical" rules.
    #
    # "Normal" (non-cross-sectional) vertical levels
    blev, blev_dims = vector_headers.get("blev", (field.blev, ()))
    lblev, lblev_dims = vector_headers.get("lblev", (field.lblev, ()))
    bhlev, bhlev_dims = vector_headers.get("bhlev", (field.bhlev, ()))
    bhrlev, bhrlev_dims = vector_headers.get("bhrlev", (field.bhrlev, ()))
    brsvd1, brsvd1_dims = vector_headers.get("brsvd1", (field.brsvd[0], ()))
    brsvd2, brsvd2_dims = vector_headers.get("brsvd2", (field.brsvd[1], ()))
    brlev, brlev_dims = vector_headers.get("brlev", (field.brlev, ()))
    # Find all the non-trivial dimension values
    dims = set(
        filter(
            None,
            [
                blev_dims,
                lblev_dims,
                bhlev_dims,
                bhrlev_dims,
                brsvd1_dims,
                brsvd2_dims,
                brlev_dims,
            ],
        )
    )
    if len(dims) > 1:
        raise TranslationError("Unsupported multiple values for vertical dimension.")
    if dims:
        v_dims = dims.pop()
        if len(v_dims) > 1:
            raise TranslationError("Unsupported multi-dimension vertical headers.")
    else:
        v_dims = ()
    coords_and_dims, factories = _convert_vertical_coords(
        field.lbcode,
        field.lbvc,
        blev,
        lblev,
        field.stash,
        bhlev,
        bhrlev,
        brsvd1,
        brsvd2,
        brlev,
        v_dims,
    )
    # Bind resulting coordinates to dimensions, where suitable.
    _bind_coords(
        coords_and_dims,
        dim_coord_dims,
        dim_coords_and_dims,
        aux_coords_and_dims,
    )

    # Realization (aka ensemble) (--> scalar coordinates)
    aux_coords_and_dims.extend(
        _convert_scalar_realization_coords(lbrsvd4=field.lbrsvd[3])
    )

    # Pseudo-level coordinate (--> scalar coordinates)
    aux_coords_and_dims.extend(
        _convert_scalar_pseudo_level_coords(lbuser5=field.lbuser[4])
    )

    return ConversionMetadata(
        factories,
        references,
        standard_name,
        long_name,
        units,
        attributes,
        cell_methods,
        dim_coords_and_dims,
        aux_coords_and_dims,
    )


def _combine_structured_cubes(cubes):
    # Combine structured cubes from different sourcefiles, in the style of
    # merge/concatenate.
    #
    # Because standard Cube.merge employed in loading can't do this.
    if STRUCTURED_LOAD_CONTROLS.structured_load_is_raw:
        # Cross-file concatenate is disabled, during a "load_raw" call.
        result = cubes
    else:
        result = iter(CubeList(cubes).concatenate())
    return result


class StructuredLoadFlags(threading.local):
    # A thread-safe object to control structured loading operations.
    # The object properties are the control flags.
    #
    # Inheriting from 'threading.local' provides a *separate* set of the
    # object properties for each thread.
    def __init__(self):
        # Control whether iris load functions do structured loads.
        self.loads_use_structured = False
        # Control whether structured load 'combine' is enabled.
        self.structured_load_is_raw = False

    @contextmanager
    def context(self, loads_use_structured=None, structured_load_is_raw=None):
        # Snapshot current states, for restoration afterwards.
        old_structured = self.loads_use_structured
        old_raw_load = self.structured_load_is_raw
        try:
            # Set flags for duration, as requested.
            if loads_use_structured is not None:
                self.loads_use_structured = loads_use_structured
            if structured_load_is_raw is not None:
                self.structured_load_is_raw = structured_load_is_raw
            # Yield to caller operation.
            yield
        finally:
            # Restore entry state of flags.
            self.loads_use_structured = old_structured
            self.structured_load_is_raw = old_raw_load


# A singleton structured-load-control object.
# Used in :meth:`iris.fileformats.pp._load_cubes_variable_loader`.
STRUCTURED_LOAD_CONTROLS = StructuredLoadFlags()


@contextmanager
def structured_um_loading():
    """Load cubes from structured UM Fieldsfile and PP files.

    "Structured" loading is a streamlined, fast load operation, to be used
    **only** on fieldsfiles or PP files whose fields repeat regularly over
    the same vertical levels and times (see full details below).

    This method is a context manager which enables an alternative loading
    mechanism for 'structured' UM files, providing much faster load times.
    Within the scope of the context manager, this affects all standard Iris
    load functions (:func:`~iris.load`, :func:`~iris.load_cube`,
    :func:`~iris.load_cubes` and :func:`~iris.load_raw`), when loading from UM
    format files (PP or fieldsfiles).

    For example:

        >>> import iris
        >>> filepath = iris.sample_data_path('uk_hires.pp')
        >>> from iris.fileformats.um import structured_um_loading
        >>> with structured_um_loading():
        ...     cube = iris.load_cube(filepath, 'air_potential_temperature')
        ...
        >>> cube
        <iris 'Cube' of air_potential_temperature / (K) \
(time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)>

    The results from this are normally equivalent to those generated by
    :func:`iris.load`, but the operation is substantially faster for input
    which is structured.

    For calls other than :meth:`~iris.load_raw`, the resulting cubes are
    concatenated over all the input files, so there is normally just one
    output cube per phenomenon.

    However, actual loaded results are somewhat different from non-structured
    loads in many cases, and in a variety of ways.  Most commonly, dimension
    ordering and the choice of dimension coordinates are often different.

    Use of load callbacks:

        When a user callback function is used with structured-loading, it is
        called in a somewhat different way than in a 'normal' load :
        The callback is called once for each basic *structured* cube loaded,
        which is normally the whole of one phenomenon from a single input file.
        In particular, the callback's "field" argument is a
        :class:`~iris.fileformats.um.FieldCollation`, from which "field.fields"
        gives a *list* of PPFields from which that cube was built, and the
        properties "field.load_filepath" and "field.load_file_indices"
        reference the original file locations of the cube data.
        The code required is therefore different from a 'normal' callback.
        For an example of this, see `this example in the Iris test code
        <https://github.com/SciTools/iris/
        blob/ddb46f78e54b6ef4110357dfe9cfcffa7d186d90/
        lib/iris/tests/integration/fast_load/test_fast_load.py#L409>`_.

    Notes on applicability:

        For results to be **correct and reliable**, the input files must
        conform to the following requirements :

        *  the file must contain fields for all possible combinations of the
           vertical levels and time points found in the file.

        *  the fields must occur in a regular repeating order within the file,
           within the fields of each phenomenon.

           For example: a sequence of fields for NV vertical levels, repeated
           for NP different forecast periods, repeated for NT different
           forecast times.

        *  all other metadata must be identical across all fields of the same
           phenomenon.

        Each group of fields with the same values of LBUSER4, LBUSER7 and
        LBPROC is identified as a separate phenomenon:  These groups are
        processed independently and returned as separate result cubes.
        The need for a regular sequence of fields applies separately to the
        fields of each phenomenon, such that different phenomena may have
        different field structures, and can be interleaved in any way at all.

        .. note::

             At present, fields with different values of 'LBUSER5'
             (pseudo-level) are *also* treated internally as different
             phenomena, yielding a raw cube per level.
             The effects of this are not normally noticed, as the resulting
             multiple raw cubes merge together again in a 'normal' load.
             However, it is not an ideal solution as operation is less
             efficient (in particular, slower) :  it is done to avoid a
             limitation in the underlying code which would otherwise load data
             on pseudo-levels incorrectly.  In future, this may be corrected.

    Known current shortcomings:

        * orography fields may be returned with extra dimensions, e.g. time,
          where multiple fields exist in an input file.

        * if some input files contain a *single* coordinate value while others
          contain *multiple* values, these will not be merged into a single
          cube over all input files :  Instead, the single- and multiple-valued
          sets will typically produce two separate cubes with overlapping
          coordinates.

          * this can be worked around by loading files individually, or with
            :meth:`~iris.load_raw`, and merging/concatenating explicitly.

    .. note::

        The resulting time-related coordinates ('time', 'forecast_time' and
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

        Restrictions and limitations:

        Any non-regular metadata variation in the input should be strictly
        avoided, as not all irregularities are detected, which can cause
        erroneous results.

        Various field header words which can in some cases vary are assumed to
        have a constant value throughout a given phenomenon.  This is **not**
        checked, and can lead to erroneous results if it is not the case.
        Header elements of potential concern include LBTIM, LBCODE, LBVC and
        LBRSVD4 (ensemble number).

    """
    with STRUCTURED_LOAD_CONTROLS.context(loads_use_structured=True):
        yield


@contextmanager
def _raw_structured_loading():
    """Prevent structured loading from concatenating its result cubes.

    Private context manager called by :func:`iris.load_raw` to prevent
    structured loading from concatenating its result cubes in that case.

    """
    with STRUCTURED_LOAD_CONTROLS.context(structured_load_is_raw=True):
        yield

.. _um_files_loading:

.. testsetup::

    import numpy as np
    import iris
    import iris.fileformats.pp
    np.set_printoptions(precision=2)


.. testcleanup::

    np.set_printoptions(precision=8)


===================================
Iris handling of PP and Fieldsfiles
===================================

This document provides a basic account of how PP and Fieldsfiles data is
represented within Iris.
It describes how Iris represents data from the Met Office Unified Model (UM),
in terms of the metadata elements found in PP and Fieldsfile formats.

For simplicity, we shall describe this mostly in terms of *loading of PP data into
Iris* (i.e. into cubes).  However most of the details are identical for
Fieldsfiles, and are relevant to saving in these formats as well as loading.

Notes:

#.  Iris treats Fieldsfile data almost exactly as if it were PP  -- i.e. it
    treats each field's lookup table entry like a PP header.
#.  The Iris datamodel is based on
    `NetCDF CF conventions <http://cfconventions.org/>`_, so most of this can
    also be seen as a metadata translation between PP and CF terms, but it is
    easier to discuss in terms of Iris elements.

For details of Iris terms (cubes, coordinates, attributes), refer to
:ref:`Iris data structures <iris_data_structures>`.

For details of CF conventions, see http://cfconventions.org/.

Overview of loading process
---------------------------

The basics of Iris loading are explained at :ref:`loading_iris_cubes`.
Loading as it specifically applies to PP and Fieldsfile data can be summarised
as follows:

#.  Input fields are first loaded from the given sources, using
    :func:`iris.fileformats.pp.load`.  This returns an iterator, which provides
    a 'stream' of :class:`~iris.fileformats.pp.PPField` input field objects.
    Each PPfield object represents a single source field:

    *   PP header elements are provided as named object attributes (e.g.
        :attr:`~iris.fileformats.pp.PPField.lbproc`).
    *   Some extra, calculated "convenience" properties are also provided (e.g.
        :attr:`~iris.fileformats.pp.PPField.t1` and
        :attr:`~iris.fileformats.pp.PPField.t2` time values).
    *   There is a :attr:`iris.fileformats.pp.PPField.data` attribute, but the
        field data is not actually loaded unless/until this is accessed, for
        greater speed and space efficiency.

#.  Each input field is translated into a two-dimensional Iris cube (with
    dimensions of latitude and longitude).  These are the 'raw' cubes, as
    returned by :meth:`iris.load_raw`.
    Within these:

    *   There are 2 horizontal dimension coordinates containing the latitude
        and longitude values for the field.
    *   Certain other header elements are interpreted as 'coordinate'-type
        values applying to the input fields, and  stored as auxiliary 'scalar'
        (i.e. 1-D) coordinates.  These include all header elements defining
        vertical and time coordinate values, and also more specialised factors
        such as ensemble number and pseudo-level.
    *   Other metadata is encoded on the cube in a variety of other forms, such
        as the cube 'name' and 'units' properties, attribute values and cell
        methods.

#.  Lastly, Iris attempts to merge the raw cubes into higher-dimensional ones
    (using :meth:`~iris.cube.CubeList.merge`):  This combines raw cubes with
    different values of a scalar coordinate to produce a higher-dimensional
    cube with the values contained in a new vector coordinate.  Where possible,
    the new vector coordinate is also a *dimension* coordinate, describing the
    new dimension.
    Apart from the original 2 horizontal dimensions, all cube dimensions and
    dimension coordinates arise in this way -- for example, 'time', 'height',
    'forecast_period', 'realization'.

.. note::
    This document covers the essential features of the UM data loading process.
    The complete details are implemented as follows:

    *   The conversion of fields to raw cubes is performed by the function
        :func:`iris.fileformats.pp_rules.convert`, which is called from
        :func:`iris.fileformats.pp.load_cubes` during loading.
    *   The corresponding save functionality for PP output is implemented by
        the :func:`iris.fileformats.pp.save` function.  The relevant
        'save rules' are defined in a text file
        ("lib/iris/etc/pp_save_rules.txt"), in a form defined by the
        :mod:`iris.fileformats.rules` module.

The rest of this document describes various independent sections of related
metadata items.

Horizontal Grid
---------------

**UM Field elements**
    LBCODE, BPLAT, BPLON, BZX, BZY, BDX, BDY, X, Y,
    X_LOWER_BOUNDS, Y_LOWER_BOUNDS

**Cube components**
    (unrotated) : coordinates ``longitude``, ``latitude``

    (rotated pole) : coordinates ``grid_latitude``, ``grid_longitude``

**Details**

At present, only latitude-longitude projections are supported (both normal and
rotated).
In these cases, LBCODE is typically 1 or 101 (though, in fact, cross-sections
with latitude and longitude axes are also supported).

For an ordinary latitude-longitude grid, the cubes have coordinates called
'longitude' and 'latitude':

 *  These are mapped to the appropriate data dimensions.
 *  They have units of 'degrees'.
 *  They have a coordinate system of type :class:`iris.coord_systems.GeogCS`.
 *  The coordinate points are normally set to the regular sequence
    ``ZDX/Y + BDX/Y * (1 .. LBNPT/LBROW)`` (*except*, if BDX/BDY is zero, the
    values are taken from the extra data vector X/Y, if present).
 *  If X/Y_LOWER_BOUNDS extra data is available, this appears as bounds values
    of the horizontal cooordinates.

For **rotated** latitude-longitude coordinates (as for LBCODE=101), the
horizontal coordinates differ only slightly --

 *  The names are 'grid_latitude' and 'grid_longitude'.
 *  The coord_system is a :class:`iris.coord_systems.RotatedGeogCS`, created
    with a pole defined by BPLAT, BPLON.

For example:
    >>> # Load a PP field.
    ... fname = iris.sample_data_path('air_temp.pp')
    >>> fields_iter = iris.fileformats.pp.load(fname)
    >>> field = next(fields_iter)
    >>> 
    >>> # Show grid details and first 5 longitude values.
    >>> print(' '.join(str(_) for _ in (field.lbcode, field.lbnpt, field.bzx,
    ...                                 field.bdx)))
    1 96 -3.749999 3.749999
    >>> print(field.bzx + field.bdx * np.arange(1, 6))
    [ 0.    3.75  7.5  11.25 15.  ]
    >>> 
    >>> # Show Iris equivalent information.
    ... cube = iris.load_cube(fname)
    >>> print(cube.coord('longitude').points[:5])
    [ 0.    3.75  7.5  11.25 15.  ]

.. note::
    Note that in Iris (as in CF) there is no special distinction between
    "regular" and "irregular" coordinates.  Thus on saving, X and Y extra data
    sections are written only if the actual values are unevenly spaced.


Phenomenon identification
-------------------------

**UM Field elements**
    LBFC, LBUSER4 (aka "stashcode"), LBUSER7 (aka "model code")

**Cube components**
    ``cube.standard_name``, ``cube.units``, ``cube.attributes['STASH']``

**Details**

This information is normally encoded in the cube ``standard_name`` property.
Iris identifies the stash section and item codes from LBUSER4 and the model
code in LBUSER7, and compares these against a list of phenomenon types with
known CF translations.  If the stashcode is recognised, it then defines the
appropriate ``standard_name`` and ``units`` properties of the cube
(i.e. :attr:`iris.cube.Cube.standard_name` and :attr:`iris.cube.Cube.units`).

Where any parts of the stash information are outside the valid range, Iris will
instead attempt to interpret LBFC, for which a set of known translations is
also stored.  This is often the case for fieldsfiles, where LBUSER4 is
frequently left as 0.

In all cases, Iris also constructs a :class:`~iris.fileformats.pp.STASH` item
to identify the phenomenon, which is stored as a cube attribute named
``STASH``.
This preserves the original STASH coding (as standard name translation is not
always one-to-one), and can be used when no standard_name translation is
identified (for example, to load only certain stashcodes with a constraint
-- see example at :ref:`Load constraint examples <constraint_egs>`).

For example:
    >>> # Show PPfield phenomenon details.
    >>> print(field.lbuser[3])
    16203
    >>> print(field.lbuser[6])
    1
    >>> 
    >>> 
    >>> # Show Iris equivalents.
    >>> print(cube.standard_name)
    air_temperature
    >>> print(cube.units)
    K
    >>> print(cube.attributes['STASH'])
    m01s16i203

.. note::
    On saving data, no attempt is made to translate a cube standard_name into a
    STASH code, but any attached 'STASH' attribute will be stored into the
    LBUSER4 and LBUSER7 elements.


Vertical coordinates
--------------------

**UM Field elements**
    LBVC, LBLEV, BRSVD1 (aka "bulev"), BRSVD2 (aka "bhulev"), BLEV, BRLEV,
    BHLEV, BHRLEV

**Cube components**
    for height levels : coordinate ``height``

    for pressure levels : coordinate ``pressure``

    for hybrid height levels :

    *   coordinates ``model_level_number``, ``sigma``, ``level_height``,
        ``altitude``
    *   ``cube.aux_factories()[0].orography``

    for hybrid pressure levels :

    *   coordinates ``model_level_number``, ``sigma``, ``level_pressure``,
        ``air_pressure``
    *   ``cube.aux_factories()[0].surface_air_pressure``


**Details**

Several vertical coordinate forms are supported, according to different values
of LBVC.  The commonest ones are:

* lbvc=1 : height levels
* lbvc=8 : pressure levels
* lbvc=65 : hybrid height

In all these cases, vertical coordinates are created, with points and bounds
values taken from the appropriate header elements.  In the raw cubes, each
vertical coordinate is just a single value, but multiple values will usually
occur.  The subsequent merge operation will then convert these into
multiple-valued coordinates, and create a new vertical data dimension (i.e. a
"Z" axis) which they map onto.

For height levels (LBVC=1):
    A ``height`` coordinate is created.  This has units 'm', points from
    BLEV, and no bounds.  When there are multiple vertical levels, this will
    become a dimension coordinate mapping to the vertical dimension.

For pressure levels (LBVC=8):
    A ``pressure`` coordinate is created.  This has units 'hPa', points from
    BLEV, and no bounds.  When there are multiple vertical levels, this will
    become a dimension coordinate mapping a vertical dimension.

For hybrid height levels (LBVC=65):
    Three basic vertical coordinates are created:

    *   ``model_level`` is dimensionless, with points from LBLEV and no bounds.
    *   ``sigma`` is dimensionless, with points from BHLEV and bounds from
        BHRLEV and BHULEV.
    *   ``level_height`` has units of 'm', points from BLEV and bounds from
        BRLEV and BULEV.

    Also in this case, a :class:`~iris.aux_factory.HybridHeightFactory` is
    created, which references the 'level_height' and 'sigma' coordinates.
    Following raw cube merging, an extra load stage occurs where the
    attached :class:`~iris.aux_factory.HybridHeightFactory` is called to
    manufacture a new ``altitude`` coordinate:

    *   The altitude coordinate is 3D, mapping to the 2 horizontal
        dimensions *and* the new vertical dimension.
    *   Its units are 'm'.
    *   Its points are calculated from those of the 'level_height' and
        'sigma' coordinates, and an orography field.  If 'sigma' and
        'level_height' possess bounds, then bounds are also created for
        'altitude'.

    To make the altitude coordinate, there must be an orography field present
    in the load sources.  This is a surface altitude reference field,
    identified (by stashcode) during the main loading operation, and recorded
    for later use in the hybrid height calculation.  If it is absent, a warning
    message is printed, and no altitude coordinate is produced.

    Note that on merging hybrid height data into a cube, only the 'model_level'
    coordinate becomes a dimension coordinate:  The other vertical coordinates
    remain as auxiliary coordinates, because they may be (variously)
    multidimensional or non-monotonic.

See an example printout of a hybrid height cube,
:ref:`here <hybrid_cube_printout>`:
    Notice that this contains all of the above coordinates --
    'model_level_number', 'sigma', 'level_height' and the derived 'altitude'.

.. note::

    Hybrid pressure levels can also be handled (for LBVC=9).  Without going
    into details, the mechanism is very similar to that for hybrid height:
    it produces basic coordinates 'model_level_number', 'sigma' and
    'level_pressure', and a manufactured 3D 'air_pressure' coordinate.


.. _um_time_metadata:

Time information
----------------

**UM Field elements**

*   "T1" (i.e. LBYR, LBMON, LBDAT, LBHR, LBMIN, LBDAY/LBSEC),
*   "T2" (i.e. LBYRD, LBMOND, LBDATD, LBHRD, LBMIND, LBDAYD/LBSECD),
*   LBTIM, LBFT

**Cube components**
    coordinates ``time``, ``forecast_reference_time``, ``forecast_period``


**Details**

In Iris (as in CF) times and time intervals are both expressed as simple
numbers, following the approach of the
`UDUNITS project <http://www.unidata.ucar.edu/software/udunits/>`_.
These values are stored as cube coordinates, where the scaling and calendar
information is contained in the :attr:`~iris.coords.Coord.units` property.

*   The units of a time interval (e.g. 'forecast_period'), can be 'seconds' or
    a simple derived unit such as 'hours' or 'days' -- but it does not contain
    a calendar, so 'months' or 'years' are not valid.
*   The units of calendar-based times (including 'time' and
    'forecast_reference_time'), are of the general form
    "<time-unit> since <base-date>", interpreted according to the unit's
    :attr:`~iris.unit.Unit.calendar` property.  The base date for this is
    always 1st Jan 1970 (times before this are represented as negative values).

The units.calendar property of time coordinates is set from the lowest decimal
digit of LBTIM, known as LBTIM.IC.  Note that the non-gregorian calendars (e.g.
360-day 'model' calendar) are defined in CF, not udunits.

There are a number of different time encoding methods used in UM data, but the
important distinctions are controlled by the next-to-lowest decimal digit of
LBTIM, known as "LBTIM.IB".
The most common cases are as follows:

Data at a single measurement timepoint (LBTIM.IB=0):
    A single ``time`` coordinate is created, with points taken from T1 values.
    It has no bounds, units of 'hours since 1970-01-01 00:00:00' and a calendar
    defined according to LBTIM.IC.

Values forecast from T2, valid at T1 (LBTIM.IB=1):
    Coordinates ``time` and ``forecast_reference_time`` are created from the T1
    and T2 values, respectively.  These have no bounds, and units of
    'hours since 1970-01-01 00:00:00', with the appropriate calendar.
    A ``forecast_period`` coordinate is also created, with values T1-T2, no
    bounds and units of 'hours'.

Time mean values between T1 and T2 (LBTIM.IB=2):
    The time coordinates ``time``, ``forecast_reference_times`` and
    ``forecast_reference_time``, are all present, as in the previous case.
    In this case, however, the 'time' and 'forecast_period' coordinates also
    have associated bounds:  The 'time' bounds are from T1 to T2, and the
    'forecast_period' bounds are from "LBFT - (T2-T1)" to "LBFT".

Note that, in those more complex cases where the input defines all three of the
'time', 'forecast_reference_time' and 'forecast_period' values, any or all of
these may become dimensions of the resulting data cube.  This will depend on
the values actually present in the source fields for each of the elements.

See an example printout of a forecast data cube,
:ref:`here <cube-statistics_forecast_printout>` :
    Notice that this example contains all of the above coordinates -- 'time',
    'forecast_period' and 'forecast_reference_time'.  In this case the data are
    forecasts, so 'time' is a dimension, 'forecast_period' varies with time and
    'forecast_reference_time' is a constant.


Statistical measures
--------------------

**UM Field elements**
    LBPROC, LBTIM

**Cube components**
    ``cube.cell_methods``


**Details**

Where a field contains statistically processed data, Iris will add an
appropriate :class:`iris.coords.CellMethod` to the cube, representing the
aggregation operation which was performed.

This is implemented for certain binary flag bits within the LBPROC element
value.  For example:

*   time mean, when (LBPROC & 128):
        Cube has a cell_method of the form "CellMethod('mean', 'time').
*   time period minimum value, when (LBPROC & 4096):
        Cube has a cell_method of the form "CellMethod('minimum', 'time').
*   time period maximum value, when (LBPROC & 8192):
        Cube has a cell_method of the form "CellMethod('maximum', 'time').

In all these cases, if the field LBTIM is also set to denote a time aggregate
field (i.e. "LBTIM.IB=2", see above :ref:`um_time_metadata`), then the
second-to-last digit of LBTIM, aka "LBTIM.IA" may also be non-zero, in which
case this indicates the aggregation time-interval.  In that case, the
cell-method :attr:`~iris.coords.CellMethod.intervals` attribute is also set to
this many hours.

For example:
    >>> # Show stats metadata in a test PP field.
    ... fname = iris.sample_data_path('pre-industrial.pp')
    >>> eg_field = next(iris.fileformats.pp.load(fname))
    >>> print(eg_field.lbtim)
    622
    >>> print(eg_field.lbproc)
    128
    >>> 
    >>> # Print out the Iris equivalent information.
    >>> print(iris.load_cube(fname).cell_methods)
    (CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),)


Other metadata
--------------

LBRSVD4
^^^^^^^
If non-zero, this is interpreted as an ensemble number.  This produces a cube
scalar coordinate named 'realization' (as defined in the CF conventions).

LBUSER5
^^^^^^^
If non-zero, this is interpreted as a 'pseudo_level' number.  This produces a
cube scalar coordinate named 'pseudo_level'.  In the UM documentation LBUSER5 is also sometimes referred to as LBPLEV.

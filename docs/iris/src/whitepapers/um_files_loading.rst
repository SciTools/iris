.. _um_files_loading:

.. testsetup::

    import numpy as np
    import iris
    import iris.fileformats.pp


===================================
Iris handling of PP and Fieldsfiles
===================================

This document provides a basic account of how PP and Fieldsfiles data is
represented within Iris.
It describes how Iris represents UM data, in terms of the metadata elements
found in PP and Fieldsfile data.

For simplicity, we shall describe this mostly in terms of *loading of PP data into
Iris* (i.e. into cubes).  However most of the details are identical for
Fieldsfiles, and are relevant to saving in these formats as well as loading.

Notes:

#.  Iris treats Fieldsfile data almost exactly as if it were PP  -- i.e. it
    treats each field's lookup table entry like a PP header.
#.  As the Iris datamodel is aligned to NetCDF-CF terms, most of this can
    also be seen as a metadata translation between PP and CF terms, but it
    is easier to discuss in terms of Iris elements.

For details of Iris terms (cubes, coordinates, attributes), refer to
:ref:`Iris data structures <iris_data_structures>`.

For details of CF conventions, see http://cf-convention.github.io/.

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
        :data:`iris.fileformats.pp.PPField.lbproc`).
    *   Extra, calculated "convenience" properties are also provided (e.g.
        :data:`iris.fileformats.pp.PPField.t1` and
        :data:`iris.fileformats.pp.PPField.t2` time values).
    *   The data payload is present (:data:`iris.fileformats.pp.PPField.data`),
        but is not actually loaded unless/until it is accessed, for greater
        speed and space efficiency.

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
    *   Other metadata encoded on the cube in a variety of other forms, such as
        the cube 'name' and 'units' properties, attribute values and cell
        methods.

#.  Lastly, Iris attempts to merge the raw cubes into higher-dimensional ones
    (using :meth:`~iris.cube.CubeList.merge`).  Where possible, this combines
    fields with different values of a scalar coordinates, to produce a
    higher-dimensional cube with the scalar coordinate values merged into a
    vector coordinate.  Where appropriate, created new vector coordinates are
    also *dimension* coordinates that describe the extra dimensions.
    Apart from the original 2 horizontal dimensions, all cube dimensions and
    dimension coordinates arise in this way -- for example, 'time', 'height',
    'forecast_period', 'realization'.

The rest of this document describes various independent sections of related
metadata items.

Horizontal Grid
---------------

**UM Field elements**
    LBCODE, BPLAT, BPLON, BZX, BZY, BDX, BDY, X, Y,
    X_LOWER_BOUNDS, Y_LOWER_BOUNDS

**Cube components**
    (unrotated) : coordinates 'longitude', 'latitude'

    (rotated pole) : coordinates 'grid_latitude', 'grid_longitude'

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
 *  The coordinate point values are normally set to the regular sequence
    "ZDX/Y + BDX/Y * (1 .. LBNPT/LBROW)" (or simply to the extra data
    vectors X/Y elements, if present).
 *  If X/Y_LOWER_BOUNDS extra data is available, this appears as bounds values
    of the horizontal cooordinates.

For **rotated** latitude-longitude coordinates (as for LBCODE=101), the
horizontal coordinates differ only slightly --
 *  The names are 'grid_latitude' and 'grid_longitude'.
 *  The coord_system is a :class:`iris.coord_systems.RotatedGeogCS`, created
    with a pole defined by BPLAT, BPLON.

Note that, in CF/Iris, there is no special distinction between "regular" and
"irregular" coordinates.  Thus on saving, X and Y extra data sections are only
written if the actual values are unevenly spaced.

For example:

    >>> # Get a path to a test file.
    >>> file_path = iris.sample_data_path('air_temp.pp')
    >>> # Load just the first field.
    >>> fields_iter = iris.fileformats.pp.load(file_path)
    >>> field = fields_iter.next()
    >>> 
    >>> # Print some details of the horizontal grid.
    >>> print 'lbcode={}, npt={}, bzx={}, bdx={}'.format(field.lbcode, field.lbnpt, field.bzx, field.bdx)
    lbcode=1, npt=96, bzx=-3.74999904633, bdx=3.74999904633
    >>> # Calculate + print the first 5 longitude values.
    >>> print 'points[1:5] = {}'.format(field.bzx + field.bdx*np.arange(1,6))
    points[1:5] = [  0.           3.74999905   7.49999809  11.24999714  14.99999619]
    >>> 
    >>> # Load the same data as an Iris cube.
    >>> cube = iris.load_cube(file_path)
    >>> 
    >>> # Print out Iris equivalent longitude details.
    >>> lons_coord = cube.coord('longitude')
    >>> print '{}/{}[{}] = {} + ...'.format(lons_coord.name(), lons_coord.units, lons_coord.shape, lons_coord.points[:5])
    longitude/degrees[(96,)] = [  0.           3.74999905   7.49999809  11.24999714  14.99999619] + ...


Phenomenon identification
-------------------------

**UM Field elements**
    LBFC, LBUSER4 (aka "stashcode"), LBUSER7 (aka "model code")

**Cube components**
    cube.standard_name, cube.units, cube.attributes['STASH']

**Details**

In Iris/CF, this information is normally encoded in the cube 'standard_name'
property.
Iris identifies the stash section and item codes from LBUSER4 and the model
code in LBUSER7, and compares these against a list of phenomenon types with
known CF translations.  If the stashcode is recognised, it then assigns the
appropriate 'standard_name' and 'units' properties of the cube.

Where any parts of the stash information are outside the valid range, Iris will
instead attempt to interpret LBFC, for which a set of known translations is
also stored.  This is often the case for fieldsfiles, where LBUSER4 is
frequently left as 0.

In all cases, Iris also constructs a (:class:`iris.fileformats.pp.STASH`) item
to identify the phenomenon, which is stored as a cube attribute named 'STASH'.
This preserves the original STASH coding (as standard name translation is not
always one-to-one), and can be used when no standard_name translation is
identified (for example, to load only certain stashcodes with a constraint
-- see example at :ref:`Load constraint examples <constraint_egs>`).

For example:

    >>> # Print PPfield phenomenon details.
    >>> print field.lbuser[3], field.lbuser[6]
    16203 1
    >>> 
    >>> # Print out Iris equivalent phenomenon details.
    >>> print '{} [{}]'.format(cube.standard_name, cube.units)
    air_temperature [K]
    >>> 
    >>> stash = cube.attributes['STASH']
    >>> print '{!r} : "{}"'.format(stash, stash)
    STASH(model=1, section=16, item=203) : "m01s16i203"
    >>> 

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
    for height levels : coordinate 'height'

    for pressure levels : coordinate 'pressure'

    for hybrid height levels :

    *   coordinates 'model_level_number', 'sigma', 'level_height', 'altitude'
    *   cube.aux_factories()[0].orography

    for hybrid pressure levels :

    *   coordinates 'model_level_number', 'sigma', 'level_pressure',
        'air_pressure'
    *   cube.aux_factories()[0].surface_air_pressure


**Details**

Several vertical coordinate forms are supported, according to different values
of LBVC.  The commonest ones are:

* lbvc=1 : height levels
* lbvc=8 : pressure levels
* lbvc=65 : hybrid height

In all these cases, vertical coordinates are created, with point and bounds
values taken from the appropriate header elements.  In the raw cubes, each
vertical coordinate is just a single value, but multiple values will usually
occur.  The subsequent merge operation will then convert these into
multiple-valued coordinates, and create a new data dimension (i.e. a "Z" axis)
which they map onto.

For height levels (LBVC=1):
    A 'height' coordinate is created.  This has units 'm', point values from
    BLEV, and no bounds.  When there are multiple vertical levels, this will
    become a dimension coordinate mapping to the vertical dimension.

For pressure levels (LBVC=8):
    A 'pressure' coordinate is created.  This has units 'hPa', point values
    from BLEV, and no bounds.  When there are multiple vertical levels, this
    will become a dimension coordinate mapping a vertical dimension.

For hybrid height levels (LBVC=65):
    Three basic vertical coordinates are created:

    *   The 'model_level' is dimensionless, with point values from LBLEV and
        no bounds.
    *   The 'sigma' coordinate is dimensionless, with point values from BHLEV
        and bounds from BHRLEV and BHULEV.
    *   The 'level_height' coordinate has units of 'm', point values from BLEV
        and bounds from BRLEV and BULEV.

    Also in this case, a :class:`~iris.aux_factory.HybridHeightFactory` is
    created, which references the 'level_height' and 'sigma' coordinates.
    Following raw cube merging, an extra load stage occurs where the
    attached :class:`~iris.aux_factory.HybridHeightFactory` is called to
    manufacture a new 'altitude' coordinate:

    *   The 'altitude' coordinate is 3D, mapping to the 2 horizontal dimensions
        *and* the new 'Z' dimension.
    *   Its units are 'm'.
    *   Its point values are calculated from those of the 'level_height' and
        'sigma' coordinates, and an orography field.  If 'sigma' and
        'level_height' possess bounds, then bounds are also created for
        'altitude'.

    To make the 'altitude' coordinate, there must be an orography field present
    in the load sources.  This is a surface altitude reference field,
    identified (by stashcode) during the main loading operation, and recorded
    for later use in the hybrid height calculation.  If it is absent, a warning
    message is printed, and no altitude coordinate is produced.

    Note that on merging hybrid height data into a cube, only the 'model_level'
    coordinate becomes a dimension coordinate:  The other vertical coordinates
    remain as auxiliary coordinates, because they may be (variously)
    multidimensional or non-monotonic.

    .. note::

        Hybrid pressure levels can also be handled (for LBVC=9).  Without going
        into detail, the mechanism is very simliar to that for hybrid height,
        and produces basic coordinates 'model_level_number', 'sigma' and
        'level_pressure', and a manufactured 3D 'air_pressure' coordinate.


For example:

    >>> # Get a path to a test file.
    ... hybrid_eg_path = iris.sample_data_path('uk_hires.pp')
    >>> # Load as PP fields.
    ... eg_fields = [f for f in iris.fileformats.pp.load(hybrid_eg_path) if f.lbuser[3] == 4]
    >>> # Print some details of the vertical data.
    ... for f_index, field in enumerate(eg_fields[:3]):
    ...     print '#{}: lbvc={}, level={}, sigma={}, height={}'.format(
    ...         f_index, field.lbvc, field.lblev, field.bhlev, field.blev)
    ... 
    #0: lbvc=65, level=1, sigma=0.999423801899, height=5.0
    #1: lbvc=65, level=4, sigma=0.991374611855, height=75.0
    #2: lbvc=65, level=7, sigma=0.976512432098, height=205.0
    >>> 
    >>> # Load the same data as an Iris cube.
    ... eg_cube = iris.load_cube(hybrid_eg_path, 'air_potential_temperature')
    >>> # Extract only first time, and first 3 levels.
    ... eg_cube = eg_cube[0, :3]
    >>> 
    >>> # Print out Iris equivalent vertical information (first 3 levels only).
    ... for coord_name in ('model_level_number', 'sigma', 'level_height'):
    ...     print eg_cube.coord(coord_name)
    ... 
    DimCoord(array([1, 4, 7], dtype=int32), standard_name='model_level_number', units=Unit('1'), attributes={'positive': 'up'})
    DimCoord(array([ 0.9994238 ,  0.99137461,  0.97651243], dtype=float32), bounds=array([[ 1.        ,  0.99846387],
           [ 0.99309671,  0.98927188],
           [ 0.97936183,  0.97328818]], dtype=float32), standard_name=None, units=Unit('1'), long_name='sigma')
    DimCoord(array([   5.,   75.,  205.], dtype=float32), bounds=array([[   0.        ,   13.33333206],
           [  60.        ,   93.33332062],
           [ 180.        ,  233.33331299]], dtype=float32), standard_name=None, units=Unit('m'), long_name='level_height', attributes={'positive': 'up'})
    >>> 
    >>> alt_coord = eg_cube.coord('altitude')
    >>> print 'Altitude{}, signature={}.'.format(alt_coord.shape, alt_coord._as_defn())
    Altitude(3, 204, 187), signature=CoordDefn(standard_name='altitude', long_name=None, var_name=None, units=Unit('m'), attributes={'positive': 'up'}, coord_system=None).
    >>> 


.. _um_time_metadata:

Time information
----------------

**UM Field elements**

*   "T1" (i.e. LBYR, LBMON, LBDAT, LBHR, LBMIN, LBDAY/LBSEC),
*   "T2" (i.e. LBYRD, LBMOND, LBDATD, LBHRD, LBMIND, LBDAYD/LBSECD),
*   LBTIM, LBFT

**Cube components**
    coordinates 'time', 'forecast_reference_time', 'forecast_period'

**Details**

In Iris/CF, times and time intervals are both expressed as simple numbers,
following the approach of the udunits project
(see http://www.unidata.ucar.edu/software/udunits/).
These values are stored as cube coordinates, where the scaling and calendar
information is contained in the :data:`iris.coords.Coord.units` property.

*   The units of a time interval (e.g. 'forecast_period'), can be 'seconds' or
    a simple derived unit such as 'hours' or 'days' -- but it does not contain
    a calendar, so 'months' or 'years' are not valid.
*   The units of calendar-based times (including 'time' and
    'forecast_reference_time'), are of the general form
    "<time-unit> since <base-date>", interpreted according to the unit's
    :data:`iris.unit.Unit.calendar` property.  The base date for this is
    always 1st Jan 1970 (times before this are represented as negative values).

The units.calendar property of time coordinates is set from the lowest decimal
digit of LBTIM, known as LBTIM.IC.  Note that the meanings of non-gregorian
calendars (e.g. 360-day 'model' calendar) are defined in CF, not udunits.

There are a number of different time encoding methods used in UM data, but the
important distinctions are controlled by the next-to-lowest decimal digit of
LBTIM, known as "LBTIM.IB".
The most common cases are as follows:

Data at a single measurement timepoint (LBTIM.IB=0):
    A single 'time' coordinate is created, with points taken from T1 values,
    and no bounds.  Its units is 'hours since 1970-01-01 00:00:00', with a
    calendar defined according to LBTIM.IC.

Values forecast from T2, valid at T1 (LBTIM.IB=1):
    Coordinates 'time' and 'forecast_reference_time' are created from the T1
    and T2 values, respectively.  These have no bounds, and units of
    'hours since 1970-01-01 00:00:00' with the appropriate calendar.
    A 'forecast_period' coordinate is also created, with values T1-T2, no
    bounds and units of 'hours'.

Time mean values between T1 and T2 (LBTIM.IB=2):
    The time coordinates 'time', 'forecast_reference_times' and 
    'forecast_reference_time', are all present, as in the previous case.
    In this case, however, the 'time' and 'forecast_period' coordinates also
    have associated bounds:  The 'time' bounds are from T1 to T2, and the
    'forecast_period' bounds are from "LBFT - (T2-T1)" to "LBFT".

Note that, in those more complex cases where the input defines all three of the
'time', 'forecast_reference_time' and 'forecast_period' values, any or all of
these may become dimensions of the resulting data cube.  This will depend on
the values actually present in the source fields for each of the elements.


For example:

    >>> # Get a path to a test file.
    ... times_eg_path = iris.sample_data_path('air_times.pp')
    >>> # Load as PP fields.
    ... eg_fields = list(iris.fileformats.pp.load(times_eg_path))
    >>> # Print details of the time metadata.
    ... for f_index, field in enumerate(eg_fields):
    ...     print 'field#{} : LBTIM.IA={} /IB={} /IC={}, LBFT={:03d}, T1={}, T2 ={}'.format(
    ...         f_index, field.lbtim.ia, field.lbtim.ib, field.lbtim.ic, field.lbft, field.t1, field.t2)
    ... 
    field#0 : LBTIM.IA=0 /IB=1 /IC=1, LBFT=000, T1=2010-02-08 03:00:00, T2 =2010-02-08 03:00:00
    field#1 : LBTIM.IA=0 /IB=1 /IC=1, LBFT=001, T1=2010-02-08 04:00:00, T2 =2010-02-08 03:00:00
    field#2 : LBTIM.IA=0 /IB=1 /IC=1, LBFT=002, T1=2010-02-08 05:00:00, T2 =2010-02-08 03:00:00
    >>> 
    >>> # Load the same data as an Iris cube.
    ... eg_cube = iris.load_cube(times_eg_path)
    >>> # Show cube structure.
    ... print eg_cube
    air_pressure_at_sea_level / (Pa)    (time: 3; grid_latitude: 928; grid_longitude: 744)
         Dimension coordinates:
              time                           x                 -                    -
              grid_latitude                  -                 x                    -
              grid_longitude                 -                 -                    x
         Auxiliary coordinates:
              forecast_period                x                 -                    -
         Scalar coordinates:
              forecast_reference_time: 2010-02-08 03:00:00
         Attributes:
              STASH: m01s16i222
              source: Data from Met Office Unified Model 7.03
    >>> 
    >>> # Print out the Iris equivalent time information.
    ... for coord_name in ('forecast_period', 'forecast_reference_time', 'time'):
    ...     print eg_cube.coord(coord_name)
    ... 
    DimCoord(array([ 0.,  1.,  2.]), standard_name='forecast_period', units=Unit('hours'))
    DimCoord([2010-02-08 03:00:00], standard_name='forecast_reference_time', calendar='gregorian')
    DimCoord([2010-02-08 03:00:00, 2010-02-08 04:00:00, 2010-02-08 05:00:00], standard_name='time', calendar='gregorian')
    >>> 



Statistical measures
--------------------

**UM Field elements**
    LBPROC, LBTIM

**Cube components**
    cube.cell_methods


**Details**

Where a field is a time statistic, Iris will also add an appropriate
:class:`iris.coords.CellMethod` to the cube, representing the aggregation
operation performed.
This is currently implemented only for certain specific binary flag values
within the LBPROC element value:

*   time mean, when (LBPROC & 128):
        Cube has a cell_method of the form "CellMethod('mean', 'time').
*   time period minimum value, when (LBPROC & 4096):
        Cube has a cell_method of the form "CellMethod('minimum', 'time').
*   time period maximum value, when (LBPROC & 8192):
        Cube has a cell_method of the form "CellMethod('maximum', 'time').

In all these cases, if the field LBTIM is also set to denote a time aggregate
field (i.e. "LBTIM.IB=2", see above :ref:`um_time_metadata`), then the
second-to-last digit of LBTIM, LBTIM.IA may also be non-zero to indicate the
aggregation time-interval.  In that case, the
:data:`iris.coords.CellMethod.intervals` is also set to this many hours.

For example:

    >>> # Get a path to a test file.
    ... stats_eg_path = iris.sample_data_path('pre-industrial.pp')
    >>> # Load as a single PP field.
    ... eg_field = iris.fileformats.pp.load(stats_eg_path).next()
    >>> 
    >>> # Print details of the statistical metadata.
    ... print 'LBTIM.IA={} /IB={} /IC={}, LBPROC={}'.format(
    ...     eg_field.lbtim.ia, eg_field.lbtim.ib, eg_field.lbtim.ic, eg_field.lbproc)
    ... 
    LBTIM.IA=6 /IB=2 /IC=2, LBPROC=128
    >>> 
    >>> # Load the same data as an Iris cube.
    ... eg_cube = iris.load_cube(stats_eg_path)
    >>> 
    >>> # Print out the Iris equivalent statistical information.
    >>> print eg_cube.cell_methods
    (CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),)
    >>> 


Other metadata
--------------

LBRSVD4
^^^^^^^
If non-zero, this is interpreted as an ensemble number.  This produces a cube
scalar coordinate named 'realization' (as defined in the CF conventions).

LBRSVD5
^^^^^^^
If if non-zero, is interpreted as an 'pseudo_level' number.  This produces a
cube scalar coordinate named 'pseudo_level'.


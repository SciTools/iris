.. testsetup:: chunk_control

    import iris
    from iris.fileformats.netcdf.loader import CHUNK_CONTROL

    from pathlib import Path
    import dask
    import shutil
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_filepath = tmp_dir / "tmp.nc"

    cube = iris.load(iris.sample_data_path("E1_north_america.nc"))[0]
    iris.save(cube, tmp_filepath, chunksizes=(120, 37, 49))
    old_dask = dask.config.get("array.chunk-size")
    dask.config.set({'array.chunk-size': '500KiB'})


.. testcleanup:: chunk_control

    dask.config.set({'array.chunk-size': old_dask})
    shutil.rmtree(tmp_dir)

.. _netcdf_io:

=============================
NetCDF I/O Handling in Iris
=============================

This document provides a basic account of how Iris loads and saves NetCDF files.

.. admonition:: Under Construction

    This document is still a work in progress, so might include blank or unfinished sections,
    watch this space!


Chunk Control
--------------

Default Chunking
^^^^^^^^^^^^^^^^

Chunks are, by default, optimised by Iris on load. This will automatically
decide the best chunksize for your data without any user input. This is
calculated based on a number of factors, including:

- File Variable Chunking
- Full Variable Shape
- Dask Default Chunksize
- Dimension Order: Earlier (outer) dimensions will be prioritised to be split over later (inner) dimensions.

.. doctest:: chunk_control

    >>> cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.shape)
    (240, 37, 49)
    >>> print(cube.core_data().chunksize)
    (60, 37, 49)

For more user control, functionality was updated in :pull:`5588`, with the
creation of the :data:`iris.fileformats.netcdf.loader.CHUNK_CONTROL` class.

Custom Chunking: Set
^^^^^^^^^^^^^^^^^^^^

There are three context managers within :data:`~iris.fileformats.netcdf.loader.CHUNK_CONTROL`. The most basic is
:meth:`~iris.fileformats.netcdf.loader.ChunkControl.set`. This allows you to specify the chunksize for each dimension,
and to specify a ``var_name`` specifically to change.

Using ``-1`` in place of a chunksize will ensure the chunksize stays the same
as the shape, i.e. no optimisation occurs on that dimension.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.set("air_temperature", time=180, latitude=-1, longitude=25):
    ...     cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (180, 37, 25)

Note that ``var_name`` is optional, and that you don't need to specify every dimension. If you
specify only one dimension, the rest will be optimised using Iris' default behaviour.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.set(longitude=25):
    ...     cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (120, 37, 25)

Custom Chunking: From File
^^^^^^^^^^^^^^^^^^^^^^^^^^

The second context manager is :meth:`~iris.fileformats.netcdf.loader.ChunkControl.from_file`.
This takes chunksizes as defined in the NetCDF file. Any dimensions without specified chunks
will default to Iris optimisation.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.from_file():
    ...     cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (120, 37, 49)

Custom Chunking: As Dask
^^^^^^^^^^^^^^^^^^^^^^^^

The final context manager, :meth:`~iris.fileformats.netcdf.loader.ChunkControl.as_dask`, bypasses
Iris' optimisation all together, and will take its chunksizes from Dask's behaviour.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.as_dask():
    ...    cube = iris.load_cube(tmp_filepath)
    >>>
    >>> print(cube.core_data().chunksize)
    (70, 37, 49)


Variable-length datatypes
-------------------------

The NetCDF4 module provides support for variable-length (or "ragged") data
types (``VLType``); see
`Variable-length data types <https://unidata.github.io/netcdf4-python/#variable-length-vlen-data-types>`_

The ``VLType`` allows for storing data where the length of the data in each array element
can vary. When ``VLType`` arrays are loaded into Iris cubes (or numpy), they are stored
as an array of ``Object`` types - essentially an array-of-arrays, rather than a single
multi-dimensional array.

The most likely case to encounter variable-length data types is when an array of
strings (not characters) are stored in a NetCDF file. As the string length for any
particular array element can vary the values are stored as an array of ``VLType``.

As each element of a variable-length array is stored as a ``VLType`` containing
an unknown number of vales, the total size of a variable-length NetCDF array
cannot be known without first loading the data. This makes it difficult for
Iris to make an informed decision on whether to the load the data lazily or not.
The user can aid this decision using *VLType size hinting* described below.

VLType size hinting
^^^^^^^^^^^^^^^^^^^

If the user has some *a priori* knowledge of the average length of the data in
variable-length ``VLType``, this can be provided as a hint to Iris via the
``CHUNK_CONTROL`` context manager and the special ``_vl_hint`` keyword
targeting the variable, e.g. ``CHUNK_CONTROL.set("varname", _vl_hint=5)``.
This allows Iris to make a more informed decision on whether to load the
data lazily.

For example, consider a netCDF file with an auxiliary coordinate
``experiment_version`` that is stored as a variable-length string type. By 
default, Iris will attempt to guess the total array size based on the known
dimension sizes (``time=150`` in this example) and load the data lazily.
However, if it is known prior to loading the file that the strings are all no
longer than 5 characters this information can be passed to the Iris NetCDF
loader so it can be make a more informed decision on lazy loading:

.. doctest::

    >>> import iris
    >>> from iris.fileformats.netcdf.loader import CHUNK_CONTROL
    >>>
    >>> sample_file = iris.sample_data_path("vlstr_type.nc")
    >>> cube = iris.load_cube(sample_file)
    >>> print(cube.coord('experiment_version').has_lazy_points())
    True
    >>> with CHUNK_CONTROL.set("expver", _vl_hint=5):
    ...     cube = iris.load_cube(sample_file)
    >>> print(cube.coord('experiment_version').has_lazy_points())
    False


Split Attributes
-----------------

TBC


Deferred Saving
----------------

TBC

.. _save_load_dataless:

Dataless Cubes in NetCDF files
------------------------------
It now possible to have "dataless" cubes, where ``cube.data is None``.
When these are saved to a NetCDF file interface, this results in a netcdf file variable
with all-unwritten data (meaning that it takes up no storage space).

In order to load such variables back correctly, we also add an extra
``iris_dataless_cube = "true"`` attribute : this tells the loader to skip array creation
when loading back in, so that the read-back cube is also dataless.


Guessing Coordinate Axes
------------------------

Iris will attempt to add an ``axis`` attribute when saving any coordinate
variable in a NetCDF file. E.g:

::

    float longitude(longitude) ;
        longitude:axis = "X" ;

This is achieved by calling :func:`iris.util.guess_coord_axis` on each
coordinate being saved.

Disabling Axis-Guessing
^^^^^^^^^^^^^^^^^^^^^^^

For some coordinates, :func:`~iris.util.guess_coord_axis` will derive an
axis that is not appropriate. If you have such a coordinate, you can disable
axis-guessing by setting the coordinate's
:attr:`~iris.coords.Coord.ignore_axis` property to ``True``.

One example (from https://github.com/SciTools/iris/issues/5003) is a
coordinate describing pressure thresholds, measured in hecto-pascals.
Iris interprets pressure units as indicating a Z-dimension coordinate, since
pressure is most commonly used to describe altitude/depth. But a
**pressure threshold** coordinate is instead describing alternate
**scenarios** - not a spatial dimension at all - and it is therefore
inappropriate to assign an axis to it.

Worked example:

.. doctest::

    >>> from iris.coords import DimCoord
    >>> from iris.util import guess_coord_axis
    >>> my_coord = DimCoord(
    ...    points=[1000, 1010, 1020],
    ...    long_name="pressure_threshold",
    ...    units="hPa",
    ... )
    >>> print(guess_coord_axis(my_coord))
    Z
    >>> my_coord.ignore_axis = True
    >>> print(guess_coord_axis(my_coord))
    None

Multiple Coordinate Systems and Ordered Axes
--------------------------------------------

In a CF compliant NetCDF file, the coordinate variables associated with a
data variable can specify a specific *coordinate system* that defines how
the coordinate values relate to physical locations on the globe. For example,
a coordinate might have values with units of metres that should be referenced
against a *Transverse Mercator* projection with a specific origin. This
information is not stored on the coordinate itself, but in a separate
*grid mapping* variable. Furthermore, the grid mapping for a set of
coordinates is associated with the data variable (not the coordinates
variables) via the ``grid_mapping`` attribute.

For example, a temperature variable defined on a *rotated pole* grid might
look like this in a NetCDF file (extract of relevant variables):

.. code-block:: text

  float T(rlat,rlon) ;
    T:long_name = "temperature" ;
    T:units = "K" ;
    T:grid_mapping = "rotated_pole" ;

  char rotated_pole ;
    rotated_pole:grid_mapping_name = "rotated_latitude_longitude" ;
    rotated_pole:grid_north_pole_latitude = 32.5 ;
    rotated_pole:grid_north_pole_longitude = 170. ;

  float rlon(rlon) ;
    rlon:long_name = "longitude in rotated pole grid" ;
    rlon:units = "degrees" ;
    rlon:standard_name = "grid_longitude";

  float rlat(rlat) ;
    rlat:long_name = "latitude in rotated pole grid" ;
    rlat:units = "degrees" ;
    rlat:standard_name = "grid_latitude";


Note how the ``rotated pole`` grid mapping (coordinate system) is referenced
from the data variable ``T:grid_mapping = "rotated_pole"`` and is implicitly
associated with the dimension coordinate variables ``rlat`` and ``rlon``.


Since version `1.8 of the CF Conventions
<https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections>`_
, there has been support for a more explicit version of the ``grid_mapping``
attribute. This allows for **multiple coordinate systems** to be defined for
a data variable and individual coordinates to be explicitly associated with
a coordinate system. This is achieved by use of an **extended syntax** in the
``grid_mapping`` variable of a data variable:


.. code-block:: text

  <grid_mapping_var>: <coord_var> [<coord_var>] [<grid_mapping_var>: <coord_var> ...]

where each ``grid_mapping_var`` identifies a grid mapping variable followed by
the list of associated coordinate variables (``coord_var``). Note that with
this syntax it is possible to specify multiple coordinate systems for a
data variable.

For example, consider the following *air pressure* variable that is
defined on an *OSGB Transverse Mercator grid*:

.. code-block:: text

    float pres(y, x) ;
        pres:standard_name = "air_pressure" ;
        pres:units = "Pa" ;
        pres:coordinates = "lat lon" ;
        pres:grid_mapping = "crsOSGB: x y crsWGS84: lat lon" ;

    double x(x) ;
        x:standard_name = "projection_x_coordinate" ;
        x:units = "m" ;

    double y(y) ;
        y:standard_name = "projection_y_coordinate" ;
        y:units = "m" ;

    double lat(y, x) ;
        lat:standard_name = "latitude" ;
        lat:units = "degrees_north" ;

    double lon(y, x) ;
        lon:standard_name = "longitude" ;
        lon:units = "degrees_east" ;

    int crsOSGB ;
        crsOSGB:grid_mapping_name = "transverse_mercator" ;
        crsOSGB:semi_major_axis = 6377563.396 ;
        crsOSGB:inverse_flattening = 299.3249646 ;
        <snip>

    int crsWGS84 ;
        crsWGS84:grid_mapping_name = "latitude_longitude" ;
        crsWGS84:longitude_of_prime_meridian = 0. ;
        <snip>


The dimension coordinates ``x`` and ``y`` are explicitly defined on
an a *transverse mercator* grid via the ``crsOSGB`` variable.

However, with the extended grid syntax, it is also possible to define
a second coordinate system on a standard **latitude_longitude** grid
and associate it with the auxiliary ``lat`` and ``lon`` coordinates:

::

    pres:grid_mapping = "crsOSGB: x y crsWGS84: lat lon" ;


Note, the *order* of the axes in the extended grid mapping specification is
significant, but only when used in conjunction with a
`CRS Well Known Text (WKT)`_ representation of the coordinate system where it
should be consistent with the ``AXES ORDER`` specified in the ``crs_wkt``
attribute.


Effect on loading
^^^^^^^^^^^^^^^^^

When Iris loads a NetCDF file that uses the extended grid mapping syntax
it will generate an :class:`iris.coord_systems.CoordSystem` for each
coordinate system listed and attempt to attach it to the associated
:class:`iris.coords.Coord` instances on the cube. Currently, Iris considers
the ``crs_wkt`` supplementary and builds coordinate systems exclusively
from the ``grid_mapping`` attribute.

The :attr:`iris.cube.Cube.extended_grid_mapping` property will be set to
``True`` for cubes loaded from NetCDF data variables utilising the extended
``grid_mapping`` syntax.

Effect on saving
^^^^^^^^^^^^^^^^

To maintain existing behaviour, saving an :class:`iris.cube.Cube` to
a netCDF file will default to the "simple" grid mapping syntax, unless
the cube was loaded from a file using the extended grid mapping syntax.
If the cube contains multiple coordinate systems, only the coordinate
system of the dimension coordinate(s) will be specified.

To enable saving of multiple coordinate systems with ordered axes,
set the :attr:`iris.cube.Cube.extended_grid_mapping` to ``True``.
This will generate a ``grid_mapping`` attribute using the extended syntax
to specify all coordinate systems on the cube. The axes ordering of the
associated coordinate variables will be consistent with that of the
generated ``crs_wkt`` attribute.

Note, the ``crs_wkt`` attribute will only be generated when the
extended grid mapping is also written, i.e. when
``Cube.extended_grid_mapping=True``.


.. _CRS Well Known Text (WKT): https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#use-of-the-crs-well-known-text-format
.. explanation:: NetCDF I/O Handling in Iris
   :tags: topic_load_save

   Read about how Iris loads and saves NetCDF files.

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
    dask.config.set({"array.chunk-size": "500KiB"})


.. testcleanup:: chunk_control

    dask.config.set({"array.chunk-size": old_dask})
    shutil.rmtree(tmp_dir)

.. _netcdf_io:

=============================
NetCDF I/O Handling in Iris
=============================

.. readingtime::

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
    ...
    >>>
    >>> print(cube.core_data().chunksize)
    (180, 37, 25)

Note that ``var_name`` is optional, and that you don't need to specify every dimension. If you
specify only one dimension, the rest will be optimised using Iris' default behaviour.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.set(longitude=25):
    ...     cube = iris.load_cube(tmp_filepath)
    ...
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
    ...
    >>>
    >>> print(cube.core_data().chunksize)
    (120, 37, 49)

Custom Chunking: As Dask
^^^^^^^^^^^^^^^^^^^^^^^^

The final context manager, :meth:`~iris.fileformats.netcdf.loader.ChunkControl.as_dask`, bypasses
Iris' optimisation all together, and will take its chunksizes from Dask's behaviour.

.. doctest:: chunk_control

    >>> with CHUNK_CONTROL.as_dask():
    ...     cube = iris.load_cube(tmp_filepath)
    ...
    >>>
    >>> print(cube.core_data().chunksize)
    (70, 37, 49)

.. _char_and_string_datatypes:

Character and String datatypes
------------------------------
Text can be present  in NetCDF in a variety of ways
(see : :ref:`cf_strings` for details).

The main aspect to be explained here is the storage of bulk text data in *variables*.

String data in Iris
^^^^^^^^^^^^^^^^^^^
Iris objects can store strings in their data arrays, such as a cube ``.data`` or
coordinate ``.points``.

These are always stored as arrays of numpy dtype "U<xx>", where <xx> is a maximum
string width (either numpy or dask: see :ref:`real_and_lazy_arrays`).

This data is currently **only** read and written to NetCDF files as
``char`` type variables (i.e. byte arrays).

.. note::

    In Iris, the NetCDF ``string`` datatype is **not supported at present**, though this
    is planned for future releases.
    See : `issue #7092 <https://github.com/SciTools/iris/issues/7092>`_.
    See the following section `Variable-length datatypes`_
    for an interim solution enabling you at least to *load* variable-length string data.

Encodings
~~~~~~~~~
String support is fairly simple when strings contain only ASCII characters.
When strings may include non-ascii characters, this requires a specific encoding to be
adopted when translating to and from bytes, and rules for determining what the encoding
is or was.

In some cases a definite record of the byte encoding is needed (though usually a default
can be assumed) : An encoding name can appear in the ``_Encoding`` attribute of a file
variable, and likewise as an attribute of the corresponding Iris component object
(e.g. cube or coordinate) :  This is loaded and saved as a normal attribute without
modification, but it can also control both loading and saving behaviour.

Iris supports only certain specific encodings :

* "ascii"
* "utf8"
* "utf16"
* "utf32"

(Though, common aliases are also allowed : those recognised by the Python ``codecs``
module).

When loading
~~~~~~~~~~~~
If there is a valid ``_Encoding`` attribute this is used to decode the
data, otherwise a default encoding of "utf8" is applied:  This works transparently when
only ascii characters are present, and also allows the ``_Encoding`` attribute to be
omitted as long as utf8 was used to write the data.

An invalid or unsupported encoding name will be ignored, with a warning, but the
attribute will still be added to the Iris component object.

When saving
~~~~~~~~~~~
To save string data **does not require an ``_Encoding`` attribute**, since UTF-8 is
applied by default -- which, for ascii data, is also equivalent to ``"ascii"``.

An ``_Encoding`` attribute can however be provided : either for clarity, or to specify a
non-default encoding (e.g. UTF-32).  This will be saved to the file.

If there are characters which can not be encoded then an error will be raised.
At present, the only *supported* encoding which this applies to is ``"ascii"``, but in
theory it could happen with other encodings, like "ISO-8859-1".

An invalid or unsupported encoding name will be ignored, with a warning, but the
attribute will still be stored to the file.

So effectively,

*   the **default encoding is 'utf8'** for both load and for save
*   no data ever actually **requires** an ``_Encoding`` to save correctly
*   if there **is** an ``_Encoding`` attribute, saving checks the actual data for
    compliance


String widths and string dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each supported encoding, Iris defines a specific function relating the string
dimension length in a NetCDF file (i.e. the "maximum byte width"), to the maximum
number of characters in the array dtype, aka string width
(i.e. the "<xx>" in the dtype "U<xx>").

On write, string dimensions are created with the **minimum number of bytes** which would
be needed to store ascii-only data of the given width in the given encoding.

These are:

*   ascii : n-bytes = n-characters
*   utf8 : n-bytes = n-characters
*   utf16 : n-bytes = 2 * (n-characters + 1)
*   utf32 : n-bytes = 4 * (n-characters + 1)

For 'ascii' and 'utf32' this character-to-byte relationship is simple + fixed.

For 'utf8' and 'utf16', however, the number of encoded bytes depends on the actual
characters present **and can exceed the numbers given above**.

String widths in Saving
#######################
If any string in an actual data array encodes to *more* bytes than the above-calculated
string dimension, when written, then Iris will raise an
:class:`iris.exceptions.TranslationError`.  In this case, the user should **explicitly
specify** a longer string dimension, by converting the data to a longer "U<xx>" dtype :
for example, ``cube.data = cube.core_data().astype("U20")``.

For example:

*   "U12" data with encoding of "utf8", "ascii", or none, will be written with a string
    dimension of 12 bytes.

*   "U7" data with an encoding of "utf16" will be written with a string dimension of
    16 bytes.

.. warning::

    When processing string arrays, Numpy does not routinely preserve the "<xx>" width part
    of "U<xx>" type data : instead, some operations will reduce it to the maximum width
    occurring.  So in these cases also, it may be necessary to explicitly re-assert the
    desired string width before saving -- use ``.astype()``,  as above.

String widths in Loading
########################
On reading, the returned data has a '"U<xx>"' dtype of which the <xx> string width is
determined by **inverting the above relations**.

For example:

*   A string dimension of 9 with an encoding of "utf8", "ascii", or none, will read in
    as a string array of dtype "U9".

*   A string dimension of 24 with an encoding of "utf32" will read in as a
    string array of dtype "U5".

The actual maximum number of characters in the data cannot exceed this dtype width,
since the maximum possible string length is achieved when all characters are plain
ascii characters -- i.e. the bytes contain no multi-byte sequences for
extended characters.

The dtype width created by reading will always round-trip correctly, i.e. the dimension
length will be unchanged if data is read and then written back.

Background: NetCDF strings in Iris' dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The relevant supporting code libraries and standards provide various facilities for
translating between bytes and Python/numpy strings, but not all possibilities are
supported.  The facilities and conventions for this have changed over time, and
obsolete methods persist in archive datasets, which must therefore be taken into
consideration.

The above documentation explains how Iris handles the different cases, and this section
details relevant aspects of its supporting projects, which in practice affect its design.
These are:

*   the NetCDF file format;
*   the CF conventions;
*   the ``numpy`` Python module; and
*   the ``netCDF4`` Python module.


.. _cf_strings:

String data in NetCDF
~~~~~~~~~~~~~~~~~~~~~
In the NetCDF v4 implementation, there are three specific areas where the datatype and
storage characteristics of character data are relevant:

*   **The names of file components (variables, dimensions, and attributes) :** are
    natively unicode-capable strings of arbitrary (variable) length.

*   **Attributes with string content :** are likewise "natively" unicode.  However, the
    actual storage datatype of the attribute may vary, being either ``char`` or ``string``.

*   **The content of variables :** can be either ``char`` or ``string``.

    *   ``string`` type variables contain a variable-length unicode string at each array element.

    *   ``char`` type variables contain one-byte characters, and generally have a fixed-length
        "string dimension".  If they contain *only* ascii character values, this is
        uncomplicated, but they may also be used to contain non-ascii data (i.e.
        including unicode characters).  There is no universally defined agreement for
        how to indicate that bytes are encoded non-ascii data, but many older datasets
        have used a variable attribute ``_Encoding`` indicating the encoding name.

.. note::

    Nearly everything here is written assuming NetCDF version 4 files, which is the newer
    NetCDF storage format based on HDF5.  The older NetCDF3 format did not provide the
    ``string`` datatype, or support unicode in names and attributes.

The NetCDF documentation does also briefly mention that an ``_Encoding`` attribute may be
used to represent non-ascii strings, but only to state that it is "reserved for future use",
and its valid values and effects are not explicitly defined.
See : `here in the NetCDF v3 description <https://docs.unidata.ucar.edu/n ug/current/file_format_specifications.html>`_
: "The variable attribute '_Encoding' is reserved ...".

However, it is also notable that the standard ``ncgen`` and ``ncdump`` tools *do*
correctly interpret an ``_Encoding`` attribute in most cases, despite this not being an
"official" solution.


String data in the NetCDF CF Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `CF Conventions <https://cfconventions.org/>`_ define a subset of
"allowed" datatypes, and various types of data elements represented by variables
-- such as data variables, auxiliary coordinates, cell methods, etc.

CF currently supports the use of either NetCDF ``string`` or ``char`` type arrays for
**any** variables.
However, *historically*, CF had more limited support, and also "unofficial conventions"
have been used for string data encoded as bytes, which may be encountered
in older datasets, as follows ...

Prior to v1.8
#############
CF required to use ``char`` type only, and provided
**no official means** of representing non-ascii data.

Since v1.8
##########
CF has allowed the use of ``string`` data in all variables.
However, up to v1.12 there was still no official way of encoding non-ascii data in
``char`` arrays.

Since v1.12
###########
CF now mandates a *default* assumption of utf-8 encoding to store
non-ascii data in ``char`` form.  It does also note that some data in the past has used an
``_Encoding`` attribute -- though this was never an official CF usage.

Characteristics of CF string storage
####################################
Where strings are stored as ``char`` datatype, which is the more common traditional approach,
the array must have a "string dimension", which is a normal file dimension.  Thus, these
strings always have a *fixed byte width*.  However, that is not the same as a fixed
*string* width, since in most encodings non-ascii characters require more bytes to
store.

CF states that a string dimension is **always the last dimension of the array**.

Although the variable-length ``string`` datatype is now supported in CF, the use of
fixed-width ``char`` arrays is obviously more efficient for storage and access, and it is
still the most common approach in practice.

String data in numpy
~~~~~~~~~~~~~~~~~~~~
Numpy provides a number of dtypes which may be used to store string data.  Relevant here
are the dtype kinds "U" and "S" : these contain elements which read and write as
Python ``str`` or ``bytes`` objects.

See `Data Types for Strings and Bytes <https://numpy.org/doc/2.5/user/basics.types.html#data-types-for-strings-and-bytes>`_.


String data in the netCDF4 Python module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attributes with string content
##############################
These always appear as Python 'str' (i.e. unicode strings).

It is not possible to distinguish or control the ``char`` and ``string`` datatype in the file :
This is hidden from the user by the Python implementation.

Variables of type ``string``
############################
Are presented (read and written) as a variable with a ``.dtype`` of ``str``
-- that is, the actual Python ``str`` class.

N.B. this is **not a valid numpy dtype** : the corresponding variable ``.datatype`` is
``netCDF4.VLType``, an internal class used to represent variable-length strings.

The variable array has a numpy dtype of "O" -- i.e. "Python objects"", and its individual
elements are Python ``str`` objects.

Variables of type ``char``
##########################
Are presented (read and written) as a variable of dtype "S1".
That is, each element is a single byte, which reads as a Python "bytes" object.

Any non-blank character reads as a length-1 byte string, but a blank character
(zero byte) reads as a zero-length ``b''``.  A blank can be *written* as either b'' or
b'\00'.

    .. note::

        The netCDF4 package can also automatically translate byte arrays into string
        arrays of dtype "U<xx>" on load, if the variable has an ``_Encoding`` attribute.
        See in netCDF4 python documentation :
        `Dealing with strings <https://unidata.github.io/netcdf4-python/#dealing-with-strings>`_.
        **However,** Iris turns this feature *off*, in order to implement its own
        wider-ranging encoding support (as described above).

    .. note::

        The netCDF4 package does not allow variables of 'S<xx>' dtype **other than** "S1".
        If you try to create one, it treats it as the equivalent "U" type, so it has
        the variable-length NetCDF ``string`` datatype, as detailed above.


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
    >>> print(cube.coord("experiment_version").has_lazy_points())
    True
    >>> with CHUNK_CONTROL.set("expver", _vl_hint=5):
    ...     cube = iris.load_cube(sample_file)
    ...
    >>> print(cube.coord("experiment_version").has_lazy_points())
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
    ...     points=[1000, 1010, 1020],
    ...     long_name="pressure_threshold",
    ...     units="hPa",
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

    float press(y, x) ;
        press:standard_name = "air_pressure" ;
        press:units = "Pa" ;
        press:coordinates = "lat lon" ;
        press:grid_mapping = "crsOSGB: x y crsWGS84: lat lon" ;

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

    press:grid_mapping = "crsOSGB: x y crsWGS84: lat lon" ;


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

=============================
Missing Data Handling in Iris
=============================

This document provides a brief overview of how Iris handles missing data values
when datasets are loaded as cubes, and when cubes are saved or modified.

A missing data value, or fill-value, defines the value used within a dataset to
indicate that data point is missing or not set.
This value is included as part of a dataset's metadata.

For example, in a gridded global ocean dataset, no data values will be recorded
over land, so land points will be missing data.
In such a case, land points could be indicated by being set to the dataset's
missing data value.


Loading
-------

On load, any fill-value or missing data value defined in the loaded dataset
should be used as the ``fill_value`` of the NumPy masked array data attribute of the
:class:`~iris.cube.Cube`. This will only appear when the cube's data is realised.


Saving
------

On save, the fill-value of a cube's masked data array is **not** used in saving data.
Instead, Iris always uses the default fill-value for the fileformat, *except*
when a fill-value is specified by the user via a fileformat-specific saver.

For example::

    >>> iris.save(my_cube, 'my_file.nc', fill_value=-99999)

.. note::
    Not all savers accept the ``fill_value`` keyword argument.

Iris will check for and issue warnings of fill-value 'collisions'.
This basically means that whenever there are unmasked values that would read back
as masked, we issue a warning and suggest a workaround.

This will occur in the following cases:

* where masked data contains *unmasked* points matching the fill-value, or
* where unmasked data contains the fill-value (either the format-specific default fill-value,
  or a fill-value specified by the user in the save call).


NetCDF
~~~~~~

NetCDF is a special case, because all ordinary variable data is "potentially masked",
owing to the use of default fill values. The default fill-value used depends on the type
of the variable data.

The exceptions to this are:

* One-byte values are not masked unless the variable has an explicit ``_FillValue`` attribute.
  That is, there is no default fill-value for ``byte`` types in NetCDF.
* Data may be tagged with a ``_NoFill`` attribute. This is not currently officially
  documented or widely implemented.
* Small integers create problems by *not* having the exemption applied to byte data.
  Thus, in principle, ``int32`` data cannot use the full range of 2**16 valid values.


Merging
-------

Merged data should have a fill-value equal to that of the components, if they
all have the same fill-value. If the components have differing fill-values, a
default fill-value will be used instead.


Other operations
----------------

Other operations, such as :class:`~iris.cube.Cube` arithmetic operations,
generally produce output with a default (NumPy) fill-value. That is, these operations
ignore the fill-values of the input(s) to the operation.

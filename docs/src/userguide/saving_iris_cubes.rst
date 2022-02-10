.. _saving_iris_cubes:

==================
Saving Iris Cubes
==================

Iris supports the saving of cubes and cube lists to:

* CF netCDF (version 1.7)
* GRIB edition 2  (if `iris-grib  <https://github.com/SciTools/iris-grib>`_ is installed)
* Met Office PP


The :py:func:`iris.save` function saves one or more cubes to a file.

If the filename includes a supported suffix then Iris will use the correct saver
and the keyword argument `saver` is not required.

    >>> import iris
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cubes = iris.load(filename)
    >>> iris.save(cubes, '/tmp/uk_hires.nc')

.. warning::

    Saving a cube whose data has been loaded lazily
    (if `cube.has_lazy_data()` returns `True`) to the same file it expects
    to load data from will cause both the data in-memory and the data on
    disk to be lost.

    .. code-block:: python

        cube = iris.load_cube("somefile.nc")
        # The next line causes data loss in 'somefile.nc' and the cube.
        iris.save(cube, "somefile.nc")

    In general, overwriting a file which is the source for any lazily loaded
    data can result in corruption. Users should proceed with caution when
    attempting to overwrite an existing file.


Controlling the Save Process
----------------------------

The :py:func:`iris.save` function passes all other keywords through to the saver function defined, or automatically set from the file extension.  This enables saver specific functionality to be called.

.. doctest::

    >>> # Save a cube to PP
    >>> iris.save(cubes[0], "myfile.pp")
    >>> # Save a cube list to a PP file, appending to the contents of the file
    >>> # if it already exists
    >>> iris.save(cubes, "myfile.pp", append=True)
    >>> # Save a cube to netCDF, defaults to NETCDF4 file format
    >>> iris.save(cubes[0], "myfile.nc")
    >>> # Save a cube list to netCDF, using the NETCDF3_CLASSIC storage option
    >>> iris.save(cubes, "myfile.nc", netcdf_format="NETCDF3_CLASSIC")

.. testcleanup::

    import pathlib
    p = pathlib.Path("myfile.pp")
    if p.exists():
        p.unlink()
    p = pathlib.Path("myfile.nc")
    if p.exists():
        p.unlink()

See 

* :py:func:`iris.fileformats.netcdf.save`
* :py:func:`iris.fileformats.pp.save`

for more details on supported arguments for the individual savers.

Customising the Save Process
----------------------------

When saving to GRIB or PP, the save process may be intercepted between the translation step and the file writing.  This enables customisation of the output messages, based on Cube metadata if required, over and above the translations supplied by Iris.

For example, a GRIB2 message with a particular known long_name may need to be saved to a specific parameter code and type of statistical process.  This can be achieved by::

        def tweaked_messages(cube):
            for cube, grib_message in iris_grib.save_pairs_from_cube(cube):
                # post process the GRIB2 message, prior to saving
                if cube.name() == 'carefully_customised_precipitation_amount':
                    gribapi.grib_set_long(grib_message, "typeOfStatisticalProcess", 1)
                    gribapi.grib_set_long(grib_message, "parameterCategory", 1)
                    gribapi.grib_set_long(grib_message, "parameterNumber", 1)
                yield grib_message
        iris_grib.save_messages(tweaked_messages(cubes[0]), '/tmp/agrib2.grib2')

Similarly a PP field may need to be written out with a specific value for LBEXP.  This can be achieved by::

        def tweaked_fields(cube):
            for cube, field in iris.fileformats.pp.save_pairs_from_cube(cube):
                # post process the PP field, prior to saving
                if cube.name() == 'air_pressure':
                    field.lbexp = 'meaxp'
                elif cube.name() == 'air_density':
                    field.lbexp = 'meaxr'
                yield field
        iris.fileformats.pp.save_fields(tweaked_fields(cubes[0]), '/tmp/app.pp')


NetCDF
^^^^^^

NetCDF is a flexible container for metadata and cube metadata is closely related to the CF for netCDF semantics.  This means that cube metadata is well represented in netCDF files, closely resembling the in memory metadata representation.
Thus there is no provision for similar save customisation functionality for netCDF saving, all customisations should be applied to the cube prior to saving to netCDF.

Bespoke Saver
-------------

A bespoke saver may be written to support an alternative file format.  This can be provided to the :py:func:`iris.save`  function, enabling Iris to write to a different file format.
Such a custom saver will need be written to meet the needs of the file format and to handle the metadata translation from cube metadata effectively. 

Implementing a bespoke saver is out of scope for the user guide.


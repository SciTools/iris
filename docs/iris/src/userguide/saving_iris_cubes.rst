.. _saving_iris_cubes:

==================
Saving Iris cubes
==================

Iris supports the saving of cubes and cube lists to:

* CF netCDF (1.5)
* GRIB (edition 2)
* Met Office PP


The :py:func:`iris.save` function saves one or more cubes to a file.

If the filename includes a supported suffix then Iris will use the correct saver
and the keyword argument `saver` is not required.

    >>> import iris
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cubes = iris.load(filename)
    >>> iris.save(cubes, '/tmp/uk_hires.nc')


Controlling the save process
-----------------------------

The :py:func:`iris.save` function passes all other keywords through to the saver function defined, or automatically set from the file extension.  This enables saver specific functionality to be called.

    >>> # Save a cube to PP
    >>> iris.save(my_cube, "myfile.pp")
    >>> # Save a cube list to a PP file, appending to the contents of the file
    >>> # if it already exists
    >>> iris.save(my_cube_list, "myfile.pp", append=True)
    >>> # Save a cube to netCDF, defaults to NETCDF4 file format
    >>> iris.save(my_cube, "myfile.nc")
    >>> # Save a cube list to netCDF, using the NETCDF4_CLASSIC storage option
    >>> iris.save(my_cube_list, "myfile.nc", netcdf_format="NETCDF3_CLASSIC")

Customising the save process
-----------------------------

When saving to GRIB or PP, the save process may be intercepted between the translation step and the file writing.  This enables customisation of the output messages, based on Cube metadata if required, over and above the translations supplied by Iris.

For example, a GRIB2 message with a particular known long_name may need to be saved to a specific parameter code and type of statistical process.  This can be achieved by::

        def tweaked_messages(cube):
            for message in iris.fileformats.grib.as_messages(cube):
                # post process the GRIB2 message, prior to saving
                if cube.name() == 'carefully_customised_precipitation_amount':
		    gribapi.grib_set_long(grib, "typeOfStatisticalProcess", 1)
                    gribapi.grib_set_long(grib, "parameterCategory", 1)
                    gribapi.grib_set_long(grib, "parameterNumber", 1)
                yield message
        iris.fileformats.grib.save(tweaked_messages(cube))



netCDF
^^^^^^^

As the cube is closely related to the CF for netCDF semantics, there is no similar functionality for netCDF saving, cube metadata is closely related to netCDF metadata, so custom cube changes are generally written our cleanly to netCDF files.

Custom Saver
-------------

A custom saver can be provided to the function to write to a different file format; the custom saver has to be written to meet the needs of the file format, but this is out of scope for the user guide.


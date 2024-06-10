.. include:: ../common_links.inc

======================
Iris ❤️ :term:`Xarray`
======================

There is a lot of overlap between Iris and :term:`Xarray`, but some important
differences too. Below is a summary of the most important differences, so that
you can be prepared, and to help you choose the best package for your use case.
See :doc:`phrasebook` for a broad comparison of terminology.

Overall Experience
------------------

Iris is the more specialised package, focused on making it as easy
as possible to work with meteorological and climatological data. Iris
is built to natively handle many key concepts, such as the CF conventions,
coordinate systems and bounded coordinates. Iris offers a smaller toolkit of
operations compared to Xarray, particularly around API for sophisticated
computation such as array manipulation and multi-processing.

Xarray's more generic data model and community-driven development give it a
richer range of operations and broader possible uses. Using Xarray
specifically for meteorology/climatology may require deeper knowledge
compared to using Iris, and you may prefer to add Xarray plugins
such as :ref:`cfxarray` to get the best experience. Advanced users can likely
achieve better performance with Xarray than with Iris.

Conversion
----------
There are multiple ways to convert between Iris and Xarray objects.

* Xarray includes the :meth:`~xarray.DataArray.to_iris` and
  :meth:`~xarray.DataArray.from_iris` methods - detailed in the
  `Xarray IO notes on Iris`_. Since Iris evolves independently of Xarray, be
  vigilant for concepts that may be lost during the conversion.
* Because both packages are closely linked to the :term:`NetCDF Format`, it is
  feasible to save a NetCDF file using one package then load that file using
  the other package. This will be lossy in places, as both Iris and Xarray
  are opinionated on how certain NetCDF concepts relate to their data models.
* `ncdata <https://github.com/pp-mo/ncdata/blob/main/README.md>`_ is a package which
  the Iris development team have developed to manage netcdf data, which can act as an
  improved 'bridge' between Iris and Xarray :

Ncdata can convert Iris cubes to an Xarray dataset, or vice versa, with minimal
overhead and as lossless as possible.

For example :

.. code-block:: python

      from ncdata.iris_xarray import cubes_from_xarray, cubes_to_xarray
      cubes = cubes_from_xarray(dataset)
      xrds = cubes_to_xarray(cubes)

Ncdata avoids the feature limitations previously mentioned regarding Xarray's
:meth:`~xarray.DataArray.to_iris` and :meth:`~xarray.DataArray.from_iris`,
because it doesn't replicate any logic of either Xarray or Iris.
Instead, it uses the netcdf file interfaces of both to exchange data
"as if" via a netcdf file.  So, these conversions *behave* just like exchanging data
via a file, but are far more efficient because they can transfer data without copying
arrays or fetching lazy data.

Regridding
----------
Iris and Xarray offer a range of regridding methods - both natively and via
additional packages such as `iris-esmf-regrid`_ and `xESMF`_ - which overlap
in places
but tend to cover a different set of use cases (e.g. Iris handles unstructured
meshes but offers access to fewer ESMF methods). The behaviour of these
regridders also differs slightly (even between different regridders attached to
the same package) so the appropriate package to use depends highly on the
particulars of the use case.

Plotting
--------
Xarray and Iris have a large overlap of functionality when creating
:term:`Matplotlib` plots and both support the plotting of multidimensional
coordinates. This means the experience is largely similar using either package.

Xarray supports further plotting backends through external packages (e.g. Bokeh through `hvPlot`_)
and, if a user is already familiar with `pandas`_, the interface should be
familiar. It also supports some different plot types to Iris, and therefore can
be used for a wider variety of plots. It also has benefits regarding "out of
the box", quick customisations to plots. However, if further customisation is
required, knowledge of matplotlib is still required.

In both cases, :term:`Cartopy` is/can be used. Iris does more work
automatically for the user here, creating Cartopy
:class:`~cartopy.mpl.geoaxes.GeoAxes` for latitude and longitude coordinates,
whereas the user has to do this manually in Xarray.

Statistics
----------
Both libraries are quite comparable with generally similar capabilities,
performance and laziness. Iris offers more specificity in some cases, such as
some more specific unique functions and masked tolerance in most statistics.
Xarray seems more approachable however, with some less unique but more
convenient solutions (these tend to be wrappers to :term:`Dask` functions).

Laziness and Multi-Processing with :term:`Dask`
-----------------------------------------------
Iris and Xarray both support lazy data and out-of-core processing through
utilisation of Dask.

While both Iris and Xarray expose :term:`NumPy` conveniences at the API level
(e.g. the `ndim()` method), only Xarray exposes Dask conveniences. For example
:attr:`xarray.DataArray.chunks`, which gives the user direct control
over the underlying Dask array chunks. The Iris API instead takes control of
such concepts and user control is only possible by manipulating the underlying
Dask array directly (accessed via :meth:`iris.cube.Cube.core_data`).

:class:`xarray.DataArray`\ s comply with `NEP-18`_, allowing NumPy arrays to be
based on them, and they also include the necessary extra members for Dask
arrays to be based on them too. Neither of these is currently possible with
Iris :class:`~iris.cube.Cube`\ s, although an ambition for the future.

NetCDF File Control
-------------------
(More info: :ref:`netcdf_io`)

Unlike Iris, Xarray generally provides full control of major file structures,
i.e. dimensions + variables, including their order in the file.  It mostly
respects these in a file input, and can reproduce them on output.
However, attribute handling is not so complete: like Iris, it interprets and
modifies some recognised aspects, and can add some extra attributes not in the
input.

Whereas Iris is primarily designed to handle netCDF data encoded according to
`CF Conventions <https://cfconventions.org/>`_ , this is not so important to Xarray,
which therefore may make it harder to correctly manage this type of data.
While Xarray CF support is not complete, it may improve, and obviously
:ref:`cfxarray` may be relevant here.
There is also relevant documentation
`at this page <https://docs.xarray.dev/en/stable/user-guide/weather-climate.html#weather-and-climate-data>`_.

In some particular aspects, CF data is not loaded well (or at all), and in many cases
output is not fully CF compliant (as-per `the cf checker <https://cfchecker.ncas.ac.uk/>`_).

* xarray has it's own interpretation of coordinates, which is different from the CF-based
  approach in Iris, and means that the use of the "coordinates" attribute in output is
  often not CF compliant.
* dates are converted to datetime-like objects internally.  There are special features
  providing `support for  non-standard calendars <https://docs.xarray.dev/en/stable/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-nanosecond-precision-range>`_,
  however date units may not always be saved correctly.
* CF-style coordinate bounds variables are not fully understood.  The CF approach
  where bounds variables do not usually define their units or standard_names can cause
  problems.  Certain files containing bounds variables with more than 2 bounds (e.g.
  unstructured data) may not load at all.
* missing points are always represented as NaNs, as-per Pandas usage.
  (See :ref:`xarray_missing_data` ).
  This means that fill values are not preserved, and that masked integer data is
  converted to floats.
  The netCDF default fill-values are not supported, so that variables with no
  "_FillValue" attribute will have missing points equal to the fill-value
  in place of NaNs.  By default, output variables generally have ``_FillValue = NaN``.

Ultimately, however, nearly everything wanted in a particular desired result file
**can** be achieved in Xarray, via provided override mechanisms (`loading keywords`_
and the '`encoding`_' dictionaries).

.. _xarray_missing_data:

Missing Data
------------
Xarray uses :data:`numpy.nan` to represent missing values and this will support
many simple use cases assuming the data are floats. Iris enables more
sophisticated missing data handling by representing missing values as masks
(:class:`numpy.ma.MaskedArray` for real data and :class:`dask.array.Array`
for lazy data) which allows data to be any data type and to include either/both
a mask and :data:`~numpy.nan`\ s.

.. _cfxarray:

`cf-xarray`_
-------------
Iris has a data model entirely based on :term:`CF Conventions`. Xarray has a
data model based on :term:`NetCDF Format` with cf-xarray acting as translation
into CF. Xarray/cf-xarray methods can be
called and data accessed with CF like arguments (e.g. axis, standard name) and
there are some CF specific utilities (similar
to Iris utilities). Iris tends to cover more of and be stricter about CF.


.. seealso::

    * `Xarray IO notes on Iris`_
    * `Xarray notes on other NetCDF libraries`_

.. _Xarray IO notes on Iris: https://docs.xarray.dev/en/stable/user-guide/io.html#iris
.. _Xarray notes on other NetCDF libraries: https://docs.xarray.dev/en/stable/getting-started-guide/faq.html#what-other-netcdf-related-python-libraries-should-i-know-about
.. _loading keywords: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray.open_dataset
.. _encoding: https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data
.. _xESMF: https://github.com/pangeo-data/xESMF/
.. _seaborn: https://seaborn.pydata.org/
.. _hvPlot: https://hvplot.holoviz.org/
.. _pandas: https://pandas.pydata.org/
.. _NEP-18: https://numpy.org/neps/nep-0018-array-function-protocol.html
.. _cf-xarray: https://github.com/xarray-contrib/cf-xarray
.. _iris#4994: https://github.com/SciTools/iris/issues/4994

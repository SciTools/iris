=====================
Iris ❤️ :term:`Xarray`
=====================

There is a lot of overlap between Iris and :term:`Xarray`, but some important
differences too. Below is a summary of the most important differences, so that
you can be prepared, and to help you choose the best package for your use case.

Regridding
----------
Iris and xarray offer a range of regridding methods - both natively and via
additional packages such as `iris-esmf-regrid`_ and `xESMF`_ - which overlap
in places
but tend to cover a different set of use cases (e.g. Iris handles unstructured
meshes but offers access to fewer ESMF methods). The behaviour of these
regridders also differ slightly (even between different regridders attached to
the same package) so the appropriate package to use depends highly on the
particulars of the use case.

Plotting
--------
xarray and Iris have a large overlap of functionality when creating matplotlib
plots and both support the plotting of multidimensional coordinates. This means
the experience is largely similar using either package.

xarray supports further plotting backends (such as seaborn, hvplot, etc) and,
if a user is already familiar with pandas, the interface should be familiar.
It also supports some different plot types to Iris, and therefore can be used
for a wider variety of plots. It also has benefits regarding "out of the box",
quick customisations to plots. However, if further customisation is required,
knowledge of matplotlib is still required.

In both cases, cartopy is/can be used. Iris does more work automatically for
the user here, creating cartopy.GeoAxes for latitude and longitude coordinates,
whereas the user has to do this manually in xarray.


Statistics
----------
Both libraries are quite comparable with generally similar capabilities,
performance and laziness. Iris offers more specificity in some cases, such as
some more specific unique functions and masked tolerance in most statistics.
Xarray seems more approachable however, with some less unique but more
convenient solutions (these tend to be wrappers to Dask functions).

Dask (Multi-Processing)
-----------------------
Iris and Xarray both support lazy data and out-of-core processing through
utilisation of Dask.

In general, Xarray exposes Dask and Numpy conveniences at the API level,
whereas Iris takes an opinionated approach with no explicit API controls.

Xarray has a level of interoperability with Dask and Numpy that is not
available with Iris, specifically with regards to NEP-18 and passing
DataArrays to Dask.

NetCDF File Control
-------------------
Unlike Iris, xarray generally provides full control of major file structures,
i.e. dimensions + variables, including their order in the file.  It mostly
respects these in a file input, and can reproduce them on output.
However, attribute handling is not so complete: like Iris, it interprets +
modifies some recognised aspects, and can add some extra attributes not in the
input.

Handling of dates and fill values have some special problems here.

Ultimately, nearly everything wanted in a particular desired result file can
be achieved in Xarray, via provided override mechanisms (`loading keywords`_
and
the '`encoding`_' dictionaries).

Missing Data
------------
Xarray uses NaNs to represent missing values and this will support many simple
use cases assuming the data are floats. Iris enables more sophisticated missing
data handling by representing missing values as masks (Numpy masked arrays for
real data and Dask arrays for lazy data) which allows data to be any data type
and to include either/both a mask and NaNs.

cf-xarray
---------
Iris has a data model entirely based on CF conventions. Xarray has a data model
based on
NetCDF with cf-xarray acting as translation into CF. Xarray/cf-xarray methods
can be
called and data accessed with CF like arguments (e.g. axis, standard name) and
there are some CF specific utilities (similar
to Iris utilities). Iris tends to cover more of and be stricter about CF.


.. seealso::

    * `xarray IO notes on Iris`_
    * `xarray notes on other NetCDF libraries`_

.. _xarray IO notes on Iris: https://docs.xarray.dev/en/stable/user-guide/io.html#iris
.. _xarray notes on other NetCDF libraries: https://docs.xarray.dev/en/stable/getting-started-guide/faq.html#what-other-netcdf-related-python-libraries-should-i-know-about
.. _loading keywords: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html?highlight=open_dataset#xarray.open_dataset
.. _encoding: https://docs.xarray.dev/en/stable/user-guide/io.html?highlight=encoding#writing-encoded-data

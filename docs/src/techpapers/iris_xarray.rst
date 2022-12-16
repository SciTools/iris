.. include:: ../common_links.inc

======================
Iris ❤️ :term:`Xarray`
======================

There is a lot of overlap between Iris and :term:`Xarray`, but some important
differences too. Below is a summary of the most important differences, so that
you can be prepared, and to help you choose the best package for your use case.

Overall Experience
------------------

Iris is the more specialised package, focussed on making it as easy
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

Xarray supports further plotting backends (such as `seaborn`_, `hvPlot`_, etc)
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

In general, Xarray exposes Dask and :term:`NumPy` conveniences at the API
level, whereas Iris takes an opinionated approach with no explicit API
controls.

Xarray has a level of interoperability with Dask and NumPy that is not
available with Iris, specifically with regards to `NEP-18`_ and passing
:class:`xarray.DataArray`\ s to Dask.

NetCDF File Control
-------------------
(More info: :term:`NetCDF Format`)

Unlike Iris, Xarray generally provides full control of major file structures,
i.e. dimensions + variables, including their order in the file.  It mostly
respects these in a file input, and can reproduce them on output.
However, attribute handling is not so complete: like Iris, it interprets +
modifies some recognised aspects, and can add some extra attributes not in the
input.

Handling of dates and fill values have some special problems here.

Ultimately, nearly everything wanted in a particular desired result file can
be achieved in Xarray, via provided override mechanisms (`loading keywords`_
and the '`encoding`_' dictionaries).

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

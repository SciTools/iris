.. _why_iris:

Why Iris
========

**A powerful, format-agnostic, community-driven Python package for analysing
and visualising Earth science data.**

Iris implements a data model based on the `CF conventions <http://cfconventions.org>`_
giving you a powerful, format-agnostic interface for working with your data.
It excels when working with multi-dimensional Earth Science data, where tabular
representations become unwieldy and inefficient.

`CF Standard names <http://cfconventions.org/standard-names.html>`_,
`units <https://github.com/SciTools/cf_units>`_, and coordinate metadata
are built into Iris, giving you a rich and expressive interface for maintaining
an accurate representation of your data. Its treatment of data and
associated metadata as first-class objects includes:

.. rst-class:: squarelist

* visualisation interface based on `matplotlib <https://matplotlib.org/>`_ and
  `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_,
* unit conversion,
* subsetting and extraction,
* merge and concatenate,
* aggregations and reductions (including min, max, mean and weighted averages),
* interpolation and regridding (including nearest-neighbor, linear and
  area-weighted), and
* operator overloads (``+``, ``-``, ``*``, ``/``, etc.).

A number of file formats are recognised by Iris, including CF-compliant NetCDF,
GRIB, and PP, and it has a plugin architecture to allow other formats to be
added seamlessly.

Building upon `NumPy <http://www.numpy.org/>`_ and
`dask <https://dask.pydata.org/en/latest/>`_, Iris scales from efficient
single-machine workflows right through to multi-core clusters and HPC.
Interoperability with packages from the wider scientific Python ecosystem comes
from Iris' use of standard NumPy/dask arrays as its underlying data storage.

Iris is part of SciTools, for more information see https://scitools.org.uk/.
For **Iris 2.4** and earlier documentation please see the
:link-badge:`https://scitools.org.uk/iris/docs/v2.4.0/,"legacy documentation",cls=badge-info text-white`.

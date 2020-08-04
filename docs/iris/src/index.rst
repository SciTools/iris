Iris Documentation
==================

.. todolist:: 

**A powerful, format-agnostic, community-driven Python library for analysing and
visualising Earth science data.**

Iris implements a data model based on the `CF conventions <http://cfconventions.org>`_ 
giving you a powerful, format-agnostic interface for working with your data.
It excels when working with multi-dimensional Earth Science data, where tabular
representations become unwieldy and inefficient.

`CF Standard names <http://cfconventions.org/standard-names.html>`_,
`units <https://github.com/SciTools/cf_units>`_, and coordinate metadata
are built into Iris, giving you a rich and expressive interface for maintaining
an accurate representation of your data. Its treatment of data and
associated metadata as first-class objects includes:

* visualisation interface based on `matplotlib <https://matplotlib.org/>`_ and
  `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_,
* unit conversion,
* subsetting and extraction,
* merge and concatenate,
* aggregations and reductions (including min, max, mean and weighted averages),
* interpolation and regridding (including nearest-neighbor, linear and area-weighted), and
* operator overloads (``+``, ``-``, ``*``, ``/``, etc.).

A number of file formats are recognised by Iris, including CF-compliant NetCDF, GRIB,
and PP, and it has a plugin architecture to allow other formats to be added seamlessly.

Building upon `NumPy <http://www.numpy.org/>`_ and
`dask <https://dask.pydata.org/en/latest/>`_,
Iris scales from efficient single-machine workflows right through to multi-core
clusters and HPC.
Interoperability with packages from the wider scientific Python ecosystem comes from Iris'
use of standard NumPy/dask arrays as its underlying data storage.


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installing
   generated/gallery/index


.. toctree::
   :maxdepth: 1
   :caption: User Guide

   userguide/index
   userguide/iris_cubes
   userguide/loading_iris_cubes
   userguide/saving_iris_cubes
   userguide/navigating_a_cube
   userguide/subsetting_a_cube
   userguide/real_and_lazy_data
   userguide/plotting_a_cube
   userguide/interpolation_and_regridding
   userguide/merge_and_concat
   userguide/cube_statistics
   userguide/cube_maths
   userguide/citation
   userguide/code_maintenance


.. toctree::
   :maxdepth: 1
   :caption: Developers Guide

   developers_guide/contributing_documentation
   developers_guide/documenting/index
   developers_guide/gitwash/index
   developers_guide/code_format
   developers_guide/pulls
   developers_guide/tests
   developers_guide/deprecations
   developers_guide/release
   generated/api/iris


.. toctree::
   :maxdepth: 1
   :caption: Reference

   whatsnew/index   
   techpapers/index   
   copyright

.. note:: For **Iris 2.4** and earlier documentation please see the
          `legacy documentation`_

.. _legacy documentation: https://scitools.org.uk/iris/docs/v2.4.0/


Iris Documentation
==================

.. todolist::

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


Panel Experiment
~~~~~~~~~~~~~~~~

https://sphinx-panels.readthedocs.io/en/latest/


Play #1
"""""""

.. panels::

    .. link-button:: Iris
        :type: ref
        :text: Iris API 1

    ---

    A package for handling multi-dimensional data and associated metadata.

    +++

    .. link-button:: Iris
        :type: ref
        :text: Iris API 2
        :classes: btn-outline-primary btn-block stretched-link


Play #2
"""""""

and more....

.. panels::
    :container: container-lg pb-3
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

    this is some text

    .. link-button:: generated/api/iris.html
        :text: Iris API 1
        :classes: btn-outline-primary
    ---
    .. link-button:: generated/api/iris.html
        :text: Iris API 2
        :classes: btn-outline-primary btn-block btn-sm
    ---
    .. link-button:: generated/api/iris.html
        :text: Iris API 3
        :classes: btn-outline-info stretched-link font-weight-bold btn-lg
    ---
    :column: col-lg-12 p-2
    panel4


Play #3
"""""""

More links....

.. panels::
    :container: container-lg pb-3
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

    Find out what's new in Iris

    +++

    .. link-button:: Iris
        :type: ref
        :text: Iris API
        :classes: btn-outline-primary
    ---

    Find out what has recently been added to Iris, or what is soon about to be.

    +++

    .. link-button:: iris_whatsnew
        :type: ref
        :text: What's new
        :classes: btn-outline-primary
    ---

    As a developer you can contribute to Iris

    +++

    .. link-button::_development_where_to_start
        :type: ref
        :text: Getting Involved
        :classes: btn-outline-primary
    ---



Play #4
"""""""

More consistent layout, showing some button variations.

.. panels::
    :container: container-lg pb-3
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

    Install to use Iris or for develeopment.
    +++
    .. link-button:: installing_iris
        :type: ref
        :text: Installing Iris
        :classes: btn-outline-primary
    ---
    View the gallery with python code used to create it.
    +++
    .. link-button:: sphx_glr_generated_gallery
        :type: ref
        :text: Gallery
        :classes: btn-primary
    ---
    Find out what has recently, or soon to be added to Iris.
    +++
    .. link-button:: iris_whatsnew
        :type: ref
        :text: What's new
        :classes: btn-outline-success
    ---
    Learn how to use Iris.
    +++
    .. link-button:: user_guide_index
        :type: ref
        :text: User Guide
        :classes: btn-outline-info
    ---
    Extensive programming API documentation.
    +++
    .. link-button:: Iris
        :type: ref
        :text: Iris API
        :classes: btn-info
    ---
    As a developer you can contribute to Iris.
    +++
    .. link-button:: development_where_to_start
        :type: ref
        :text: Getting Involved
        :classes: btn-success


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installing
   generated/gallery/index


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :name: userguide_index

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


.. _developers_guide:

.. toctree::
   :maxdepth: 1
   :caption: Developers Guide
   :name: development_index

   developers_guide/contributing_getting_involved
   developers_guide/gitwash/index
   developers_guide/contributing_documentation
   developers_guide/contributing_codebase_index
   developers_guide/contributing_changes
   developers_guide/release
   generated/api/iris


.. toctree::
   :maxdepth: 1
   :caption: Reference

   whatsnew/index
   techpapers/index
   copyright

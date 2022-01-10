.. include:: ../common_links.inc

.. _developer_running_tests:

Running the Tests
*****************

Using setuptools for Testing Iris
=================================

.. warning:: The `setuptools`_ ``test`` command was deprecated in `v41.5.0`_. See :ref:`using nox`.

A prerequisite of running the tests is to have the Python environment
setup.  For more information on this see :ref:`installing_from_source`.

Many Iris tests will use data that may be defined in the test itself, however
this is not always the case as sometimes example files may be used.  Due to
the size of some of the files used these are not kept in the Iris repository.
A separate repository under the `SciTools`_ organisation is used, see
https://github.com/SciTools/iris-test-data.

In order to run the tests with **all** the test data you must clone the
``iris-test-data`` repository and then ensure the Iris tests can access
``iris-test-data/test_data``, using one of two methods:

* Store the path in a shell environment variable named **OVERRIDE_TEST_DATA_REPOSITORY**.
* Store the path in ``lib/iris/etc/site.cfg`` (see :mod:`iris.config` for more).

The example command below uses ``~/projects`` as the parent directory::

   cd ~/projects
   git clone git@github.com:SciTools/iris-test-data.git
   export OVERRIDE_TEST_DATA_REPOSITORY=~/projects/iris-test-data/test_data

All the Iris tests may be run from the root ``iris`` project directory via::

   python setup.py test

You can also run a specific test, the example below runs the tests for
mapping::

   cd lib/iris/tests
   python test_mapping.py

When running the test directly as above you can view the command line options
using the commands ``python test_mapping.py -h`` or
``python test_mapping.py --help``.

.. tip::  A useful command line option to use is ``-d``.  This will display
          matplotlib_ figures as the tests are run.  For example::

             python test_mapping.py -d

          You can also use the ``-d`` command line option when running all
          the tests but this will take a while to run and will require the
          manual closing of each of the figures for the tests to continue.

The output from running the tests is verbose as it will run ~5000 separate
tests.  Below is a trimmed example of the output::

   running test
   Running test suite(s): default

   Running test discovery on iris.tests with 2 processors.
   test_circular_subset (iris.tests.experimental.regrid.test_regrid_area_weighted_rectilinear_src_and_grid.TestAreaWeightedRegrid) ... ok
   test_cross_section (iris.tests.experimental.regrid.test_regrid_area_weighted_rectilinear_src_and_grid.TestAreaWeightedRegrid) ... ok
   test_different_cs (iris.tests.experimental.regrid.test_regrid_area_weighted_rectilinear_src_and_grid.TestAreaWeightedRegrid) ... ok
   ...
   ...
   test_ellipsoid (iris.tests.unit.experimental.raster.test_export_geotiff.TestProjection) ... SKIP: Test requires 'gdal'.
   test_no_ellipsoid (iris.tests.unit.experimental.raster.test_export_geotiff.TestProjection) ... SKIP: Test requires 'gdal'.
   ...
   ...
   test_slice (iris.tests.test_util.TestAsCompatibleShape) ... ok
   test_slice_and_transpose (iris.tests.test_util.TestAsCompatibleShape) ... ok
   test_transpose (iris.tests.test_util.TestAsCompatibleShape) ... ok

   ----------------------------------------------------------------------
   Ran 4762 tests in 238.649s

   OK (SKIP=22)

There may be some tests that have been **skipped**.  This is due to a Python
decorator being present in the test script that will intentionally skip a test
if a certain condition is not met.  In the example output above there are
**22** skipped tests, at the point in time when this was run this was primarily
due to an experimental dependency not being present.


.. tip::

   The most common reason for tests to be skipped is when the directory for the
   ``iris-test-data`` has not been set which would shows output such as::

      test_coord_coord_map (iris.tests.test_plot.Test1dScatter) ... SKIP: Test(s) require external data.
      test_coord_coord (iris.tests.test_plot.Test1dScatter) ... SKIP: Test(s) require external data.
      test_coord_cube (iris.tests.test_plot.Test1dScatter) ... SKIP: Test(s) require external data.

   All Python decorators that skip tests will be defined in
   ``lib/iris/tests/__init__.py`` with a function name with a prefix of
   ``skip_``.


.. _using nox:

Using Nox for Testing Iris
==========================

Iris has adopted the use of the `nox`_ tool for automated testing on `cirrus-ci`_
and also locally on the command-line for developers.

`nox`_ is similar to `tox`_, but instead leverages the expressiveness and power of a Python
configuration file rather than an `.ini` style file. As with `tox`_, `nox`_ can use `virtualenv`_
to create isolated Python environments, but in addition also supports `conda`_ as a testing
environment backend.


Where is Nox Used?
------------------

Iris uses `nox`_ as a convenience to fully automate the process of executing the Iris tests, but also
automates the process of:

* building the documentation and executing the doc-tests
* building the documentation gallery
* running the documentation URL link check
* linting the code-base
* ensuring the code-base style conforms to the `black`_ standard


You can perform all of these tasks manually yourself, however the onus is on you to first ensure
that all of the required package dependencies are installed and available in the testing environment.

`Nox`_ has been configured to automatically do this for you, and provides a means to easily replicate
the remote testing behaviour of `cirrus-ci`_ locally for the developer.


Installing Nox
--------------

We recommend installing `nox`_ using `conda`_. To install `nox`_ in a separate `conda`_ environment::

  conda create -n nox -c conda-forge nox
  conda activate nox

To install `nox`_ in an existing active `conda`_ environment::

  conda install -c conda-forge nox

The `nox`_ package is also available on PyPI, however `nox`_ has been configured to use the `conda`_
backend for Iris, so an installation of `conda`_ must always be available.


Testing with Nox
----------------

The `nox`_ configuration file `noxfile.py` is available in the root ``iris`` project directory, and
defines all the `nox`_ sessions (i.e., tasks) that may be performed. `nox`_ must always be executed
from the ``iris`` root directory.

To list the configured `nox`_ sessions for Iris::

  nox --list

To run the Iris tests for all configured versions of Python::

  nox --session tests

To build the Iris documentation specifically for Python 3.7::

  nox --session doctest-3.7

To run all the Iris `nox`_ sessions::

  nox

For further `nox`_ command-line options::

  nox --help

.. tip::
    For `nox`_ sessions that use the `conda`_ backend, you can use the ``-v`` or ``--verbose``
    flag to display the `nox`_ `conda`_ environment package details and environment info.
    For example::

        nox --session tests -- --verbose


.. note:: `nox`_ will cache its testing environments in the `.nox` root ``iris`` project directory.


.. _setuptools: https://setuptools.readthedocs.io/en/latest/
.. _tox: https://tox.readthedocs.io/en/latest/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _PyPI: https://pypi.org/project/nox/
.. _v41.5.0: https://setuptools.readthedocs.io/en/latest/history.html#v41-5-0

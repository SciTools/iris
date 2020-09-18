.. include:: ../common_links.inc

.. _developer_running_tests:

Running the tests
*****************

A prerequisite of running the tests is to have the Python environment
setup.  For more information on this see :ref:`installing_from_source`.

Many Iris tests will use data that may be defined in the test itself, however
this is not always the case as sometimes example files may be used.  Due to
the size of some of the files used these are not kept in the Iris repository.
A separate repository under the `SciTools`_ organisation is used, see
https://github.com/SciTools/iris-test-data.

In order to run the tests with **all** the test data you must clone the
``iris-test-data`` repository and then configure your shell to ensure the Iris
tests can find it by using the shell environment variable named
**OVERRIDE_TEST_DATA_REPOSITORY**.  The example command below uses
``~/projects`` as the parent directory::

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
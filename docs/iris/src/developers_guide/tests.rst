Testing
*******

There are three categories of tests within Iris:
 - Unit tests
 - Integration tests
 - Legacy tests

All code changes should be accompanied by one or more unit tests, and by
zero or more integration tests. No new tests should be added to the
legacy tests.

Unit tests
==========

All code changes should be accompanied by enough unit tests to give a
high degree of confidence that the change works as expected. In
addition, the unit tests can help describe the intent behind a change.

The docstring for each test module should state the unit under test.
For example:
    `"""Unit tests for the `iris.experimental.raster.export_geotiff` function."""`

All unit tests must be placed and named according to the following
structure:

Classes
-------
When testing a class all the tests must reside in the module:
    `lib/iris/tests/unit/<fully/qualified/module>/test_<ClassName>.py`

For example, unit tests for :py:class:`iris.cube.Cube` must reside in:
    `lib/iris/tests/unit/cube/test_Cube.py`

Within this test module each tested method must have one or more
corresponding test classes:
 - Either: `Test_name_of_public_method`
 - Or: `Test_name_of_public_method__aspect_of_method`

For example, unit tests for :py:meth:`iris.fileformats.netcdf.Saver.write`
must reside in classes named:
 - `Test_write`
 - or something like `Test_write__data_types`

Functions
---------
When testing a function all the tests must reside in the module:
    `lib/iris/tests/unit/<fully/qualified/module>/test_<function_name>.py`

For example, unit tests for `iris.util.file_is_newer_than` must reside in:
    `lib/iris/tests/unit/util/test_file_is_newer_than.py`

Within this test module there must be one or more test classes:
 - `Test*AspectOfFunction*`


Integration tests
=================

Some code changes may require tests which exercise several units in
order to demonstrate an important consequence of their interaction which
may not be apparent when considering the units in isolation.

These tests should be placed in the `lib/iris/tests/integration` folder.
Unlike unit tests, there is no fixed naming scheme for integration
tests. But it is recommended they are named according to the relevant
capabilities under test, e.g. `test_pp_metadata_preservation.py`, and
not named according to the module(s) under test.

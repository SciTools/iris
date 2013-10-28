Testing
*******

There are three categories of tests within Iris:
 - Unit tests
 - Integration tests
 - Legacy tests

All code changes should be accompanied by one or more unit tests, and by
zero or more integration tests. New tests should not be added to the
legacy tests.

Unit tests
==========

All code changes should be accompanied by enough unit tests to give a
high degree of confidence that the change works as expected. In
addition, the unit tests can help describe the intent behind a change.

The docstring for each test module must state the unit under test.
For example:
    `"""Unit tests for the `iris.experimental.raster.export_geotiff` function."""`

All unit tests must be placed and named according to the following
structure:

Classes
-------
When testing a class all the tests must reside in the module:
    `lib/iris/tests/unit/<fully/qualified/module>/test_<ClassName>.py`

Within this test module each tested method must have one or more
corresponding test classes:
 - Either: `Test_name_of_public_method`
 - Or: `Test_name_of_public_method__aspect_of_method`

And within those test classes, the test methods must be named according
to the aspect of the tested method which they address.

**Examples**:

All unit tests for :py:class:`iris.cube.Cube` must reside in:
    `lib/iris/tests/unit/cube/test_Cube.py`

All unit tests for :py:meth:`iris.cube.Cube.xml` must reside in the file
above with test classes named:
    `Test_xml`

    or something like `Test_xml__checksum`

So a unit test of :py:meth:`iris.cube.Cube.xml` which examines the handling
of masked data might be named:
    `Test_write.test_checksum_ignores_masked_values`.

Functions
---------
When testing a function all the tests must reside in the module:
    `lib/iris/tests/unit/<fully/qualified/module>/test_<function_name>.py`

Within this test module there must be one or more test classes:
 - Either: `TestAll`
 - Or: `TestAspectOfFunction`

And within those test classes, the test methods must be named according
to the aspect of the tested function which they address.

**Examples**:

All unit tests for :py:func:`iris.experimental.raster.export_geotiff`
must reside in:
    `lib/iris/tests/unit/experimental/raster/test_export_geotiff.py`

With tests classes named:
    `TestAll`

    or something like `TestGeoTransform`

So a unit test of :py:func:`iris.experimental.raster.export_geotiff`
which examines the handling of 16-bit integer data might be named:
    `TestDtypeAndValues.test_int16`


Integration tests
=================

Some code changes may require tests which exercise several units in
order to demonstrate an important consequence of their interaction which
may not be apparent when considering the units in isolation.

These tests must be placed in the `lib/iris/tests/integration` folder.
Unlike unit tests, there is no fixed naming scheme for integration
tests. But folders and files must be created as required to help
developers locate relevant tests. It is recommended they are named
according to the capabilities under test, e.g.
`metadata/test_pp_preservation.py`, and not named according to the
module(s) under test.

.. _developer_tests:

Testing
*******

There are three categories of tests within Iris:
 - Unit tests
 - Integration tests
 - Legacy tests

Ideally, all code changes should be accompanied by one or more unit
tests, and by zero or more integration tests. And where possible, new
tests should not be added to the legacy tests.

But if in any doubt about what tests to add or how to write them please
feel free to submit a pull-request in any state and ask for assistance.


Unit tests
==========

Code changes should be accompanied by enough unit tests to give a
high degree of confidence that the change works as expected. In
addition, the unit tests can help describe the intent behind a change.

The docstring for each test module must state the unit under test.
For example:

    :literal:`"""Unit tests for the \`iris.experimental.raster.export_geotiff\` function."""`

All unit tests must be placed and named according to the following
structure:

Classes
-------
When testing a class all the tests must reside in the module:

    :literal:`lib/iris/tests/unit/<fully/qualified/module>/test_<ClassName>.py`

Within this test module each tested method must have one or more
corresponding test classes:
- Either: `Test_name_of_public_method`
- Or: `Test_name_of_public_method__aspect_of_method`

And within those test classes, the test methods must be named according
to the aspect of the tested method which they address.

**Examples**:

All unit tests for :py:class:`iris.cube.Cube` must reside in:

    :literal:`lib/iris/tests/unit/cube/test_Cube.py`

Within that file the tests might look something like:

.. code-block:: python

    # Tests for the Cube.xml() method.
    class Test_xml(tests.IrisTest):
        def test_some_general_stuff(self):
            ...

    # Tests for the Cube.xml() method, focussing on the behaviour of
    # the checksums.
    class Test_xml__checksum(tests.IrisTest):
        def test_checksum_ignores_masked_values(self):
            ...

    # Tests for the Cube.add_dim_coord() method.
    class Test_add_dim_coord(tests.IrisTest):
        def test_normal_usage(self):
            ...

        def test_coord_already_present(self):
            ...


Functions
---------
When testing a function all the tests must reside in the module:

    :literal:`lib/iris/tests/unit/<fully/qualified/module>/test_<function_name>.py`

Within this test module there must be one or more test classes:
- Either: `Test`
- Or: `TestAspectOfFunction`

And within those test classes, the test methods must be named according
to the aspect of the tested function which they address.

**Examples**:

All unit tests for :py:func:`iris.experimental.raster.export_geotiff`
must reside in:

    :literal:`lib/iris/tests/unit/experimental/raster/test_export_geotiff.py`

Within that file the tests might look something like:

.. code-block:: python

    # Tests focussing on the handling of different data types.
    class TestDtypeAndValues(tests.IrisTest):
        def test_int16(self):
            ...

        def test_int16_big_endian(self):
            ...

    # Tests focussing on the handling of different projections.
    class TestProjection(tests.IrisTest):
        def test_no_ellipsoid(self):
            ...


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


Graphics tests
=================
Certain Iris tests are based on checking plotted images.
This the only way of testing the modules :mod:`iris.plot` and
:mod:`iris.quickplot`, but is also used for some other legacy and integration-
style testcases.

Prior to Iris version 1.10, a single reference image for each testcase was
stored in the main Iris repository, and a 'tolerant' comparison was performed
against this.

From version 1.11 onwards, graphics testcase outputs are compared against
possibly *multiple* known-good images, of which only the signature is stored.
This uses a sophisticated perceptual "image hashing" scheme (see: 
<https://github.com/JohannesBuchner/imagehash>).
Only imagehash signatures are stored in the Iris repo itself, thus freeing up
valuable space.  Meanwhile, the actual reference *images* -- which are required
for human-eyes evaluation of proposed new "good results" -- are all stored
elsewhere in a separate public repository.
See :ref:`developer_graphics_tests`.

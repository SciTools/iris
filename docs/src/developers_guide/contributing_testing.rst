.. include:: ../common_links.inc

.. _developer_test_categories:


Test Categories
***************

There are two main categories of tests within Iris:

 - :ref:`testing.unit_test`
 - :ref:`testing.integration`

Ideally, all code changes should be accompanied by one or more unit
tests, and by zero or more integration tests.

But if in any doubt about what tests to add or how to write them please
feel free to submit a pull-request in any state and ask for assistance.


.. _testing.unit_test:

Unit Tests
==========

Code changes should be accompanied by enough unit tests to give a
high degree of confidence that the change works as expected. In
addition, the unit tests can help describe the intent behind a change.

The docstring for each test module must state the unit under test.
For example:

    :literal:`"""Unit tests for the \`iris.experimental.raster.export_geotiff\` function."""`

All unit tests must be placed and named according to the following
structure:


.. _testing.classes:

Classes
-------

When testing a class all the tests must reside in the module:

    :literal:`lib/iris/tests/unit/<fully/qualified/module>/test_<ClassName>.py`

Within this test module each tested method must have one or more
corresponding test classes, for example:

* ``Test_<name of public method>``
* ``Test_<name of public method>__<aspect of method>``

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


.. _testing.functions:

Functions
---------

When testing a function all the tests must reside in the module:

    :literal:`lib/iris/tests/unit/<fully/qualified/module>/test_<function_name>.py`

Within this test module there must be one or more test classes, for example:

* ``Test``
* ``TestAspectOfFunction``

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


.. _testing.integration:

Integration Tests
=================

Some code changes may require tests which exercise several units in
order to demonstrate an important consequence of their interaction which
may not be apparent when considering the units in isolation.

These tests must be placed in the ``lib/iris/tests/integration`` folder.
Unlike unit tests, there is no fixed naming scheme for integration
tests. But folders and files must be created as required to help
developers locate relevant tests. It is recommended they are named
according to the capabilities under test, e.g.
``metadata/test_pp_preservation.py``, and not named according to the
module(s) under test.

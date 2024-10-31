.. include:: ../common_links.inc

.. _contributing_tests:

***********************
Contributing Iris Tests
***********************

.. note::
    If you're converting UnitTest tests to PyTest, check out :ref:`converting_tests`.

.. _developer_pytest_categories:

Test Categories
===============

There are two main categories of tests within Iris:

- :ref:`testing.unit_test`
- :ref:`testing.integration`

Ideally, all code changes should be accompanied by one or more unit
tests, and by zero or more integration tests.

But if in any doubt about what tests to add or how to write them please
feel free to submit a pull-request in any state and ask for assistance.


.. _pytesting.unit_test:

Unit Tests
----------

Code changes should be accompanied by enough unit tests to give a
high degree of confidence that the change works as expected. In
addition, the unit tests can help describe the intent behind a change.

The docstring for each test module must state the unit under test.
For example:

    :literal:`"""Unit tests for the \`iris.experimental.raster.export_geotiff\` function."""`

All unit tests must be placed and named according to the following
structure:


.. _pytesting.classes:

Classes
-------

When testing a class all the tests must reside in the module:

    :literal:`lib/iris/tests/unit/<fully/qualified/module>/test_<ClassName>.py`


Within this test module each tested method may corresponding test classes,
for example:

* ``Test_<name of public method>``
* ``Test_<name of public method>__<aspect of method>``

Within test classes, the test methods must be named according
to the aspect of the tested method which they address.

**Examples**:

All unit tests for :py:class:`iris.cube.Cube` reside in:

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


.. _pytesting.functions:

Functions
---------

When testing a function all the tests must reside in the module:

    :literal:`lib/iris/tests/unit/<fully/qualified/module>/test_<function_name>.py`

Within this test module there may be test classes, for example:

* ``Test``
* ``TestAspectOfFunction``

Within those test classes, the test methods must be named according
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


.. _pytesting.integration:

Integration Tests
-----------------

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

.. _pytesting_style_guide:

PyTest Style Guide
==================

This style guide should be approached pragmatically. Most of the guidelines laid out
below will not be practical in every scenario, and as such should not be considered
firm rules.

At time of writing, some existing tests have already been written in PyTest,
so might not be abiding by these guidelines.

`conftest.py <https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files>`_
----------------------------------------------------------------------------------------------------------------------------

There should be a ``conftest.py`` file in the ``root/unit`` and ``root/integration``
folders. Additional lower level ``conftest``s can be added if it is agreed there
is a need.

`Fixtures <https://docs.pytest.org/en/stable/how-to/fixtures.html#how-to-fixtures>`_
------------------------------------------------------------------------------------

As far as is possible, the actual test function should do little else but the
actual assertion. Separating off preparation into fixtures may make the code
harder to follow, so compromises are acceptable. For example, setting up a test
``Cube`` should be a fixture, whereas creating a simple string
(``expected = "foo"``), or a single use setup, should *not* be a fixture.


New fixtures should always be considered for ``conftest`` when added. If it is
decided that they are not suitably reusable, they can be placed within the
local test file.

`Parameterisation <https://docs.pytest.org/en/stable/example/parametrize.html>`_
--------------------------------------------------------------------------------

Though it is a useful tool, we should not be complicating tests to work around
parameters; they should only be used when it is simple and apparent to implement.

Where you are parameterising multiple tests with the same parameters, it is
usually prudent to use the `parameterisation within fixtures
<https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures>`_.
When doing this, ensure within the tests that it is apparent that they are being
parameterised, either within the fixture name or with comments.

All parameterisation benefits from
`ids <https://docs.pytest.org/en/stable/example/parametrize.html#different-options-for-test-ids>`_,
and so should be used where possible.

`Classes <https://docs.pytest.org/en/stable/getting-started.html#group-multiple-tests-in-a-class>`_
---------------------------------------------------------------------------------------------------

How and when to group tests within classes can be based on personal opinion,
we do not deem consistency on this a vital concern.

`Mocks <https://docs.pytest.org/en/stable/how-to/monkeypatch.html>`_
--------------------------------------------------------------------

Any mocking should be done with ``pytest.mock``, and monkeypatching where suitable.

.. note::
    If you think we're missing anything important here, please consider creating an
    issue or discussion and share your ideas with the team!

.. _pytesting_tools:

==============
Testing tools
==============

.. note::
    :class:`iris.tests.IrisTest` has been deprecated, and replaced with
    the :mod:`iris.tests._shared_utils` module.

Iris has various internal convenience functions and utilities available to
support writing tests. Using these makes tests quicker and easier to write, and
also consistent with the rest of Iris (which makes it easier to work with the
code). Most of these conveniences are accessed through the
:mod:`iris.tests._shared_utils` module.

.. tip::

    All functions listed on this page are defined within
    :mod:`iris.tests._shared_utils`. They can be accessed within a test using
    ``_shared_utils.example_function``.

Custom assertions
-----------------

:mod:`iris.tests._shared_utils` supports a variety of custom pytest-style
assertions, such as :func:`~iris.tests._shared_utils.assert_array_equal`, and
:func:`~iris.tests._shared_utils.assert_array_almost_equal`.

.. _pycreate-missing:

Saving results
--------------

Some tests compare the generated output to the expected result contained in a
file. Custom assertions for this include
:func:`~iris.tests._shared_utils.assert_CML_approx_data`
:func:`~iris.tests._shared_utils.assert_CDL`
:func:`~iris.tests._shared_utils.assert_CML` and
:func:`~iris.tests._shared_utils.assert_text_file`. See docstrings for more
information.

.. note::

    Sometimes code changes alter the results expected from a test containing the
    above methods. These can be updated by removing the existing result files
    and then running the file containing the test with a ``--create-missing``
    command line argument, or setting the ``IRIS_TEST_CREATE_MISSING``
    environment variable to anything non-zero. This will create the files rather
    than erroring, allowing you to commit the updated results.

=================
Context managers
=================

Capturing exceptions and logging
--------------------------------

:mod:`~iris.tests._shared_utils` includes several context managers that can be used
to make test code tidier and easier to read. These include
:meth:`~iris.tests._shared_utils.assert_no_warnings_regexp` and
:meth:`~iris.tests._shared_utils.assert_logs`.

Patching
--------

After the change from ``unittest`` to ``pytest``, ``IrisTest.patch`` has been
converted into :meth:`~iris.tests._shared_utils.patch`.

This is currently not implemented, and will raise an error if called.

Graphic tests
-------------

As a package capable of generating graphical outputs, Iris has utilities for
creating and updating graphical tests - see :ref:`testing.graphics` for more
information.
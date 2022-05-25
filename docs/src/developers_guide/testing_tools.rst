.. include:: ../common_links.inc

.. _testing_tools:

Testing tools
*************

Iris has various internal convenience functions and utilities available to
support writing tests. Using these makes tests quicker and easier to write, and
also consistent with the rest of Iris (which makes it easier to work with the
code). Most of these conveniences are accessed through the
:class:`iris.tests.IrisTest` class, from
which Iris' test classes then inherit.

.. tip::

    All functions listed on this page are defined within
    :mod:`iris.tests.__init__.py` as methods of
    :class:`iris.tests.IrisTest_nometa` (which :class:`iris.tests.IrisTest`
    inherits from). They can be accessed within a test using
    ``self.exampleFunction``.

Custom assertions
=================

:class:`iris.tests.IrisTest` supports a variety of custom unittest-style
assertions, such as :meth:`~iris.tests.IrisTest_nometa.assertStringEqual`,
:meth:`~iris.tests.IrisTest_nometa.assertArrayEqual`,
:meth:`~iris.tests.IrisTest_nometa.assertArrayAlmostEqual`.

Saving results
--------------

Some tests compare the generated output to the expected result contained in a
file. Custom assertions for this include
:meth:`~iris.tests.IrisTest_nometa.assertCMLApproxData`
:meth:`~iris.tests.IrisTest_nometa.assertCDL`
:meth:`~iris.tests.IrisTest_nometa.assertCML` and
:meth:`~iris.tests.IrisTest_nometa.assertTextFile`. See docstrings for more
information.

.. note::

    Sometimes code changes alter the results expected from a test containing the
    above methods. These can be updated by removing the existing result files
    and then running the file containing the test with a ``--create-missing``
    command line argument, or setting the ``IRIS_TEST_CREATE_MISSING``
    environment variable to anything non-zero. This will create the files rather
    than erroring, allowing you to commit the updated results.

Context managers
================

Capturing exceptions and logging
--------------------------------

:class:`iris.tests.IrisTest` includes several context managers that can be used
to make test code tidier and easier to read. These include
:meth:`~iris.tests.IrisTest_nometa.assertWarnsRegexp` and
:meth:`~iris.tests.IrisTest_nometa.assertLogs`.

Temporary files
---------------

It's also possible to generate temporary files in a concise fashion with
:meth:`~iris.tests.IrisTest_nometa.temp_filename`.

Patching
========

:meth:`~iris.tests.IrisTest_nometa.patch` is a wrapper around ``unittest.patch``
that will be automatically cleaned up at the end of the test.

Graphic tests
=============

As a package capable of generating graphical outputs, Iris has utilities for
creating and updating graphical tests - see :ref:`testing.graphics` for more
information.
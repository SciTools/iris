.. include:: ../common_links.inc

.. _testing_tools:

Testing tools
*************

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
    ``_shared_utils.exampleFunction``.

Custom assertions
=================

:mod:`iris.tests._shared_utils` supports a variety of custom pytest-style
assertions, such as :meth:`~iris.tests._shared_utils.assert_array_equal`, and
:meth:`~iris.tests._shared_utils.assert_array_almost_equal`.

.. _create-missing:

Saving results
--------------

Some tests compare the generated output to the expected result contained in a
file. Custom assertions for this include
:meth:`~iris.tests._shared_utils.assert_CML_approx_data`
:meth:`~iris.tests._shared_utils.assert_CDL`
:meth:`~iris.tests._shared_utils.assert_CML` and
:meth:`~iris.tests._shared_utils.assert_text_file`. See docstrings for more
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

:mod:`~iris.tests._shared_utils` includes several context managers that can be used
to make test code tidier and easier to read. These include
:meth:`~iris.tests.IrisTest_nometa.assert_no_warnings_regexp` and
:meth:`~iris.tests.IrisTest_nometa.assert_logs`.

Patching
========

After the change from ``unittest`` to ``pytest``, ``IrisTest.patch`` has been
converted into :meth:`~iris.tests._shared_utils.patch`.

This is currently not implemented, and will raise an error if called.

Graphic tests
=============

As a package capable of generating graphical outputs, Iris has utilities for
creating and updating graphical tests - see :ref:`testing.graphics` for more
information.
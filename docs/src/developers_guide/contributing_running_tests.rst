.. include:: ../common_links.inc

.. _developer_running_tests:

Running the Tests
*****************

There are two options for running the tests:

* Use an environment you created yourself.  This requires more manual steps to
  set up, but gives you more flexibility.  For example, you can run a subset of
  the tests or use ``python`` interactively to investigate any issues.  See
  :ref:`test manual env`.

* Use ``nox``.  This will automatically generate an environment and run test
  sessions consistent with our GitHub continuous integration.  See :ref:`using nox`.

.. _test manual env:

Testing Iris in a Manually Created Environment
==============================================

To create a suitable environment for running the tests, see :ref:`installing_from_source`.

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

All the Iris tests may be run from the root ``iris`` project directory using
``pytest``.  For example::

   pytest -n 2

will run the tests across two processes.  For more options, use the command
``pytest -h``.  Below is a trimmed example of the output::

   ============================= test session starts ==============================
   platform linux -- Python 3.10.5, pytest-7.1.2, pluggy-1.0.0
   rootdir: /path/to/git/clone/iris, configfile: pyproject.toml, testpaths: lib/iris
   plugins: xdist-2.5.0, forked-1.4.0
   gw0 I / gw1 I
   gw0 [6361] / gw1 [6361] 

   ........................................................................ [  1%]
   ........................................................................ [  2%]
   ........................................................................ [  3%]
   ...
   .......................ssssssssssssssssss............................... [ 99%]
   ........................                                                 [100%]
   =============================== warnings summary ===============================
   ...
   -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
   =========================== short test summary info ============================
   SKIPPED [1] lib/iris/tests/experimental/test_raster.py:152: Test requires 'gdal'.
   SKIPPED [1] lib/iris/tests/experimental/test_raster.py:155: Test requires 'gdal'.
   ...
   ========= 6340 passed, 21 skipped, 1659 warnings in 193.57s (0:03:13) ==========

There may be some tests that have been **skipped**.  This is due to a Python
decorator being present in the test script that will intentionally skip a test
if a certain condition is not met.  In the example output above there are
**21** skipped tests. At the point in time when this was run this was due to an
experimental dependency not being present.

.. tip::

   The most common reason for tests to be skipped is when the directory for the
   ``iris-test-data`` has not been set which would shows output such as::

      SKIPPED [1] lib/iris/tests/unit/fileformats/test_rules.py:157: Test(s) require external data.
      SKIPPED [1] lib/iris/tests/unit/fileformats/pp/test__interpret_field.py:97: Test(s) require external data.
      SKIPPED [1] lib/iris/tests/unit/util/test_demote_dim_coord_to_aux_coord.py:29: Test(s) require external data.
      
   All Python decorators that skip tests will be defined in
   ``lib/iris/tests/__init__.py`` with a function name with a prefix of
   ``skip_``.

You can also run a specific test module.  The example below runs the tests for
mapping::

   cd lib/iris/tests
   python test_mapping.py

When running the test directly as above you can view the command line options
using the commands ``python test_mapping.py -h`` or
``python test_mapping.py --help``.

.. tip::  A useful command line option to use is ``-d``.  This will display
          matplotlib_ figures as the tests are run.  For example::

             python test_mapping.py -d

.. _using nox:

Using Nox for Testing Iris
==========================

The `nox`_ tool has for adopted for automated testing on `Iris GitHub Actions`_
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

You can perform all of these tasks manually yourself, however the onus is on you to first ensure
that all of the required package dependencies are installed and available in the testing environment.

`Nox`_ has been configured to automatically do this for you, and provides a means to easily replicate
the remote testing behaviour of `Iris GitHub Actions`_ locally for the developer.


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

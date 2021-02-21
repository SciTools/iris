.. _developer_testing_ci:

.. include:: ../common_links.inc

Continuous Integration (CI) Testing
===================================

The `Iris`_ GitHub repository is configured to run checks against all its
branches automatically whenever a pull request is created, updated or merged.
The checks performed are:

* :ref:`testing_cla`
* :ref:`testing_cirrus`


.. _testing_cla:

SciTools CLA Checker
********************

A bot which checks that the GitHub author of the pull request has signed the
**SciTools Contributor's License Agreement (CLA)**.  For more information on
this please see https://scitools.org.uk/organisation.html#governance.


.. _testing_cirrus:

Cirrus-CI
*********

Iris unit and integration tests are an essential mechanism to ensure
that the Iris code base is working as expected.  :ref:`developer_running_tests`
may be performed manually by a developer locally. However Iris is configured to
use the `cirrus-ci`_ service for automated Continuous Integration (CI) testing.

The `cirrus-ci`_ configuration file  `.cirrus.yml`_ in the root of the Iris repository
defines the tasks to be performed by `cirrus-ci`_. For further details
refer to the `Cirrus-CI Documentation`_. The tasks performed during CI include:

* linting the code base and ensuring it adheres to the `black`_ format
* running the system, integration and unit tests for Iris
* ensuring the documentation gallery builds successfully
* performing all doc-tests within the code base
* checking all URL references within the code base and documentation are valid

The above `cirrus-ci`_ tasks are run automatically against all `Iris`_ branches
on GitHub whenever a pull request is submitted, updated or merged. See the
`Cirrus-CI Dashboard`_ for details of recent past and active Iris jobs.

.. _skipping Cirrus-CI tasks:

Skipping Cirrus-CI Tasks
------------------------

As a developer you may wish to not run all the CI tasks when you are actively
developing e.g., you are writing documentation and there is no need for linting,
or long running compute intensive testing tasks to be executed.

As a convenience, it is possible to easily skip one or more tasks by setting
the appropriate environment variable within the `.cirrus.yml`_ file to a
**non-empty** string:

* ``SKIP_LINT_TASK`` to skip `flake8`_ linting and `black`_ formatting
* ``SKIP_TEST_MINIMAL_TASK`` to skip restricted unit and integration testing
* ``SKIP_TEST_FULL_TASK`` to skip full unit and integration testing
* ``SKIP_GALLERY_TASK`` to skip building the documentation gallery
* ``SKIP_DOCTEST_TASK`` to skip running the documentation doc-tests
* ``SKIP_LINKCHECK_TASK`` to skip checking for broken documentation URL references
* ``SKIP_ALL_TEST_TASKS`` which is equivalent to setting ``SKIP_TEST_MINIMAL_TASK`` and ``SKIP_TEST_FULL_TASK``
* ``SKIP_ALL_DOC_TASKS`` which is equivalent to setting ``SKIP_GALLERY_TASK``, ``SKIP_DOCTEST_TASK``, and ``SKIP_LINKCHECK_TASK``

e.g., to skip the linting task, the following are all equivalent::

   SKIP_LINT_TASK: "1"
   SKIP_LINT_TASK: "true"
   SKIP_LINT_TASK: "false"
   SKIP_LINT_TASK: "skip"
   SKIP_LINT_TASK: "unicorn"


GitHub Checklist
****************

An example snapshot from a successful GitHub pull request shows all tests
passing:

.. image:: ci_checks.png

If any CI tasks fail, then the pull request is unlikely to be merged to the
Iris target branch by a core developer.


.. _Cirrus-CI Dashboard: https://cirrus-ci.com/github/SciTools/iris
.. _Cirrus-CI Documentation: https://cirrus-ci.org/guide/writing-tasks/


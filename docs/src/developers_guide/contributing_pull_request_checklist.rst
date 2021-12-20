.. include:: ../common_links.inc

.. _pr_check:

Pull Request Checklist
======================

All pull request will be reviewed by a core developer who will manage the
process of merging. It is the responsibility of a developer submitting a
pull request to do their best to deliver a pull request which meets the
requirements of the project it is submitted to.

The check list summarises criteria which will be checked before a pull request
is merged.  Before submitting a pull request please consider this list.


#. **Provide a helpful description** of the Pull Request.  This should include:

   * The aim of the change / the problem addressed / a link to the issue.
   * How the change has been delivered.

#. **Include a "What's New" entry**, if appropriate.
   See :ref:`whats_new_contributions`.

#. **Check all tests pass**.  This includes existing tests and any new tests
   added for any new functionality.  For more information see
   :ref:`developer_running_tests`.

#. **Check all modified and new source files conform to the required**
   :ref:`code_formatting`.

#. **Check all new dependencies added to the** `requirements/ci/`_ **yaml
   files.**  If dependencies have been added then new nox testing lockfiles
   should be generated too, see :ref:`cirrus_test_env`.

#. **Check the source documentation been updated to explain all new or changed
   features**.  See :ref:`docstrings`.

#. **Include code examples inside the docstrings where appropriate**.  See
   :ref:`contributing.documentation.testing`.

#. **Check the documentation builds without warnings or errors**.  See
   :ref:`contributing.documentation.building`

#. **Check for any new dependencies in the** `.cirrus.yml`_ **config file.**

#. **Check for any new dependencies in the** `readthedocs.yml`_ **file**.  This
   file is used to build the documentation that is served from
   https://scitools-iris.readthedocs.io/en/latest/

#. **Check for updates needed for supporting projects for test or example
   data**.  For example:

    * `iris-test-data`_ is a github project containing all the data to support
      the tests.
    * `iris-sample-data`_ is a github project containing all the data to support
      the gallery and examples.
    * `test-iris-imagehash`_ is a github project containing reference plot
      images to support Iris :ref:`testing.graphics`.

   If new files are required by tests or code examples, they must be added to
   the appropriate supporting project via a suitable pull-request.  This pull
   request should be referenced in the main Iris pull request and must be
   accepted and merged before the Iris one can be.

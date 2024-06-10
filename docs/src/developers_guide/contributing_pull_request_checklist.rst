.. include:: ../common_links.inc

.. _pr_check:

Pull Request Checklist
======================

All pull request will be reviewed by a core developer who will manage the
process of merging. It is the responsibility of the contributor submitting a
pull request to do their best to deliver a pull request which meets the
requirements of the project it is submitted to.

This check list summarises criteria which will be checked before a pull request
is merged.  Before submitting a pull request please consider the following:


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

#. **Check all new dependencies added to the** `requirements`_ **yaml
   files.**  If dependencies have been added then new nox testing lockfiles
   should be generated too, see :ref:`gha_test_env`.

#. **Check the source documentation been updated to explain all new or changed
   features**.  Note, we now use numpydoc strings.  Any touched code should
   be updated to use the docstrings formatting. See :ref:`docstrings`.

#. **Include code examples inside the docstrings where appropriate**.  See
   :ref:`contributing.documentation.testing`.

#. **Check the documentation builds without warnings or errors**.  See
   :ref:`contributing.documentation.building`

#. **Check for any new dependencies in the** `readthedocs.yml`_ **file**.  This
   file is used to build the documentation that is served from
   https://scitools-iris.readthedocs.io/en/latest/

#. **Check for updates needed for supporting projects for test or example
   data**.  For example:

   * `iris-test-data`_ is a github project containing all the data to support
     the tests.
   * `iris-sample-data`_ is a github project containing all the data to support
     the gallery and examples.

   If new files are required by tests or code examples, they must be added to
   the appropriate supporting project via a suitable pull-request.  This pull
   request should be referenced in the main Iris pull request and must be
   accepted and merged before the Iris one can be.

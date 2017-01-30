Pull Request Check List
***********************

A pull request to a SciTools project master should be ready to merge into the
master branch.

All pull request will be reviewed by a core developer who will manage the
process of merging. It is the responsibility of a developer submitting a
pull request to do their best to deliver a pull request which meets the
requirements of the project it is submitted to. 

The check list summarises criteria which will be checked before a pull request
is merged.  Before submitting a pull request please consider this list.


The Iris Check List
====================

* Have you provided a helpful description of the Pull Request?
  I.E. what has changed and why.  This should include:
 * the aim of the change ; the problem addressed ; a link to the issue.
 * how the change has been delivered.
 * a "What's New" entry, submitted as a new file added in the pull request.
   See `Contributing a "What's New" entry`_.

* Do all the tests pass locally?
 * The Iris tests may be run with ``python setup.py test`` which has a command 
   line utility included.

* Have new tests been provided for all additional functionality?

* Do all modified and new sourcefiles pass PEP8?
 * PEP8_ is the Python source code style guide.
 * There is a python module for checking pep8 compliance: python-pep8_
 * a standard Iris test checks that all sourcefiles meet PEP8 compliance
   (see "iris.tests.test_coding_standards.TestCodeFormat").

* Do all modified and new sourcefiles have a correct, up-to-date copyright
  header?
 * a standard Iris test checks that all sourcefiles include a copyright
   message, including the correct year of the latest change
   (see "iris.tests.test_coding_standards.TestLicenseHeaders").

* Has the documentation been updated to explain all new or changed features?
 * refer to the developer guide on docstrings_

* Have code examples been provided inside docstrings, where relevant?
 * these are strongly recommended as concrete (working) examples always
   considerably enhance the documentation.
  * live test code can be included in docstrings.
   * See for example :data:`iris.cube.Cube.data`
   * Details at http://www.sphinx-doc.org/en/stable/ext/doctest.html
  * The documentation tests may be run with ``make doctest``, from within the
    ``./docs/iris`` subdirectory.

* Have you provided a 'whats new' contribution?
 * this should be done for all changes that affect API or behaviour.
   See :ref:`whats_new_contributions`

* Does the documentation build without errors?
 * The documentation is built using ``make html`` in ``./docs/iris``.

* Do the documentation and code-example tests pass?
 * Run with ``make doctest`` and ``make extest``, from within the subdirectory
   ``./docs/iris``.
 * note that code examples must *not* raise deprecations.  This is now checked
   and will result in an error.
   When an existing code example encounters a deprecation, it must be fixed.

* Has the travis file been updated to reflect any dependency updates?
 * ``./.travis.yml`` is used to manage the continuous integration testing.
 * the files ``./conda-requirements.yml`` and
    ``./minimal-conda-requirements.yml`` are used to define the software
    environments used, using the conda_ package manager.

* Have you provided updates to supporting projects for test or example data?
 * the following separate repos are used to manage larger files used by tests
   and code examples :
  * iris-test-data_ is a github project containing all the data to support the
    tests.
  * iris-sample-data_ is a github project containing all the data to support
    the gallery and examples.
  * test-images-scitools_ is a github project containing reference plot images
    to support iris graphics tests : see :ref:`test graphics images`.
 * If new files are required by tests or code examples, they must be added to
   the appropriate supporting project via a suitable pull-request.
   This new 'supporting pull request' should be referenced in the main Iris
   pull request, and must be accepted and merged before the Iris one can be.


.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _python-pep8: https://pypi.python.org/pypi/pep8
.. _conda: http://conda.readthedocs.io/en/latest/
.. _iris-test-data: https://github.com/SciTools/iris-test-data
.. _iris-sample-data: https://github.com/SciTools/iris-sample-data
.. _test-images-scitools https://github.com/SciTools/test-images-scitools
.. _docstrings: http://scitools.org.uk/iris/docs/latest/developers_guide/documenting/docstrings.html
.. _Contributing a "What's New" entry: http://scitools.org.uk/iris/docs/latest/developers_guide/documenting/whats_new_contributions.html

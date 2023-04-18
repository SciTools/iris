.. include:: ../common_links.inc

.. _testing.graphics:

Adding or Updating Graphics Tests
=================================

.. note::

  If a large number of images tests are failing due to an update to the
  libraries used for image hashing, follow the instructions on
  :ref:`refresh-imagerepo`.

Generating New Results
----------------------

When you find that a graphics test in the Iris testing suite has failed,
following changes in Iris or the run dependencies, this is the process
you should follow:

#. Create a new, empty directory to store temporary image results, at the path
   ``lib/iris/tests/result_image_comparison`` in your Iris repository checkout.

#. Run the relevant (failing) tests directly as python scripts, or using
   ``pytest``.

The results of the failing image tests will now be available in
``lib/iris/tests/result_image_comparison``.

.. note::

  The ``result_image_comparison`` folder is covered by a project
  ``.gitignore`` setting, so those files *will not show up* in a
  ``git status`` check.

Reviewing Failing Tests
-----------------------

#. Run ``iris/lib/iris/tests/graphics/idiff.py`` with python, e.g.::

     python idiff.py

   This will open a window for you to visually inspect
   side-by-side **old**, **new** and **difference** images for each failed
   graphics test.  Hit a button to either :guilabel:`accept`,
   :guilabel:`reject` or :guilabel:`skip` each new result.

   If the change is **accepted**:

   * the imagehash value of the new result image is added into the relevant
     set of 'valid result hashes' in the image result database file,
     ``tests/results/imagerepo.json``

   * the relevant output file in ``tests/result_image_comparison`` is renamed
     according to the test name. A copy of this new PNG file must then be added
     into the ``iris-test-data`` repository, at
     https://github.com/SciTools/iris-test-data (See below).

   If a change is **skipped**:

   * no further changes are made in the repo.

   * when you run ``iris/tests/idiff.py`` again, the skipped choice will be
     presented again.

   If a change is **rejected**:

   * the output image is deleted from ``result_image_comparison``.

   * when you run ``iris/tests/idiff.py`` again, the skipped choice will not
     appear, unless the relevant failing test is re-run.

#. **Now re-run the tests**. The **new** result should now be recognised and the
   relevant test should pass.  However, some tests can perform *multiple*
   graphics checks within a single test case function.  In those cases, any
   failing check will prevent the following ones from being run, so a test
   re-run may encounter further (new) graphical test failures.  If that
   happens, simply repeat the check-and-accept process until all tests pass.

#. You're now ready to :ref:`add-graphics-test-changes`


Adding a New Image Test
-----------------------

If you attempt to run ``idiff.py`` when there are new graphical tests for which
no baseline yet exists, you will get a warning that ``idiff.py`` is ``Ignoring
unregistered test result...``. In this case,

#. rename the relevant images from ``iris/tests/result_image_comparison`` by

   * removing the ``result-`` prefix

   * fully qualifying the test name if it isn't already (i.e. it should start
     ``iris.tests...``or ``gallery_tests...``)

#. run the tests in the mode that lets them create missing data (see
   :ref:`create-missing`). This will update ``imagerepo.json`` with the new
   test name and image hash.

#. and then add them to the Iris test data as covered in
   :ref:`add-graphics-test-changes`.


.. _refresh-imagerepo:

Refreshing the Stored Hashes
----------------------------

From time to time, a new version of the image hashing library will cause all
image hashes to change. The image hashes stored in
``tests/results/imagerepo.json`` can be refreshed using the baseline images
stored in the ``iris-test-data`` repository (at
https://github.com/SciTools/iris-test-data) using the script
``tests/graphics/recreate_imagerepo.py``. Use the ``--help`` argument for the
command line arguments.


.. _add-graphics-test-changes:

Add Your Changes to Iris
------------------------

To add your changes to Iris, you need to make two pull requests (PR).

#. The first PR is made in the ``iris-test-data`` repository, at
   https://github.com/SciTools/iris-test-data.

   * Add all the newly-generated referenced PNG files into the
     ``test_data/images`` directory.  In your Iris repo, these files are to be found
     in the temporary results folder ``iris/tests/result_image_comparison``.

   * Create a PR proposing these changes, in the usual way.

#. The second PR is the one that makes the changes you intend to the Iris_ repository.
   The description box of this pull request should contain a reference to
   the matching one in ``iris-test-data``.

   * This PR should include updating the version of the test data in
     ``.github/workflows/ci-tests.yml`` and
     ``.github/workflows/ci-docs-tests.yml`` to the new version created by the
     merging of your ``iris-test-data`` PR.

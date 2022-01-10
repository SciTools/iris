.. include:: ../common_links.inc

.. _testing.graphics:

Graphics Tests
**************

Iris may be used to create various forms of graphical output; to ensure
the output is consistent, there are automated tests to check against
known acceptable graphical output.  See :ref:`developer_running_tests` for
more information.

At present graphical tests are used in the following areas of Iris:

* Module ``iris.tests.test_plot``
* Module ``iris.tests.test_quickplot``
* :ref:`sphx_glr_generated_gallery` plots contained in
  ``docs/gallery_tests``.


Challenges
==========

Iris uses many dependencies that provide functionality, an example that
applies here is matplotlib_.  For more information on the dependences, see
:ref:`installing_iris`.  When there are updates to the matplotlib_ or a
dependency of matplotlib, this may result in a change in the rendered graphical
output.  This means that there may be no changes to Iris_, but due to an
updated dependency any automated tests that compare a graphical output to a
known acceptable output may fail.  The failure may also not be visually
perceived as it may be a simple pixel shift.


Testing Strategy
================

The `Iris Cirrus-CI matrix`_ defines multiple test runs that use
different versions of Python to ensure Iris is working as expected.

To make this manageable, the ``iris.tests.IrisTest_nometa.check_graphic`` test
routine tests against multiple alternative **acceptable** results.  It does
this using an image **hash** comparison technique which avoids storing
reference images in the Iris repository itself.

This consists of:

 * The ``iris.tests.IrisTest_nometa.check_graphic`` function uses a perceptual
   **image hash** of the outputs (see https://github.com/JohannesBuchner/imagehash)
   as the basis for checking test results.

 * The hashes of known **acceptable** results for each test are stored in a
   lookup dictionary, saved to the repo file
   ``lib/iris/tests/results/imagerepo.json``
   (`link <https://github.com/SciTools/iris/blob/main/lib/iris/tests/results/imagerepo.json>`_) .

 * An actual reference image for each hash value is stored in a *separate*
   public repository https://github.com/SciTools/test-iris-imagehash.

 * The reference images allow human-eye assessment of whether a new output is
   judged to be close enough to the older ones, or not.

 * The utility script ``iris/tests/idiff.py`` automates checking, enabling the
   developer to easily compare proposed new **acceptable** result images
   against the existing accepted reference images, for each failing test.

The acceptable images for each test can be viewed online. The :ref:`testing.imagehash_index` lists all the graphical tests in the test suite and
shows the known acceptable result images for comparison.

Reviewing Failing Tests
=======================

When you find that a graphics test in the Iris testing suite has failed,
following changes in Iris or the run dependencies, this is the process
you should follow:

#. Create a new, empty directory to store temporary image results, at the path
   ``lib/iris/tests/result_image_comparison`` in your Iris repository checkout.

#. **In your Iris repo root directory**, run the relevant (failing) tests
   directly as python scripts, or by using a command such as::

     python -m unittest discover paths/to/test/files

#. In the ``iris/lib/iris/tests`` folder, run the command::

     python idiff.py

   This will open a window for you to visually inspect
   side-by-side **old**, **new** and **difference** images for each failed
   graphics test.  Hit a button to either :guilabel:`accept`,
   :guilabel:`reject` or :guilabel:`skip` each new result.

   If the change is **accepted**:

     * the imagehash value of the new result image is added into the relevant
       set of 'valid result hashes' in the image result database file,
       ``tests/results/imagerepo.json``

     * the relevant output file in ``tests/result_image_comparison`` is
       renamed according to the image hash value, as ``<hash>.png``.
       A copy of this new PNG file must then be added into the reference image
       repository at https://github.com/SciTools/test-iris-imagehash
       (See below).

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


Add Your Changes to Iris
========================

To add your changes to Iris, you need to make two pull requests (PR).

#. The first PR is made in the ``test-iris-imagehash`` repository, at
   https://github.com/SciTools/test-iris-imagehash.

    * First, add all the newly-generated referenced PNG files into the
      ``images/v4`` directory.  In your Iris repo, these files are to be found
      in the temporary results folder ``iris/tests/result_image_comparison``.

    * Then, to update the file which lists available images,
      ``v4_files_listing.txt``, run from the project root directory::

         python recreate_v4_files_listing.py

    * Create a PR proposing these changes, in the usual way.

#. The second PR is created in the Iris_ repository, and
   should only include the change to the image results database,
   ``tests/results/imagerepo.json``.
   The description box of this pull request should contain a reference to
   the matching one in ``test-iris-imagehash``.

.. note::

  The ``result_image_comparison`` folder is covered by a project
  ``.gitignore`` setting, so those files *will not show up* in a
  ``git status`` check.

.. important::

  The Iris pull-request will not test successfully in Cirrus-CI until the
  ``test-iris-imagehash`` pull request has been merged.  This is because there
  is an Iris_ test which ensures the existence of the reference images (uris)
  for all the targets in the image results database.  It will also fail
  if you forgot to run ``recreate_v4_files_listing.py`` to update the
  image-listing file in ``test-iris-imagehash``.


.. _Iris Cirrus-CI matrix: https://github.com/scitools/iris/blob/main/.cirrus.yml

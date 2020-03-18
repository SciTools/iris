.. _developer_graphics_tests:

Graphics tests
**************

The only practical way of testing plotting functionality is to check actual
output plots.
For this, a basic 'graphics test' assertion operation is provided in the method
:meth:`iris.tests.IrisTest.check_graphic` :  This tests plotted output for a
match against a stored reference.
A "graphics test" is any test which employs this.

At present, such tests include the testing for modules `iris.tests.test_plot`
and `iris.tests.test_quickplot`, all output plots from the gallery examples
(contained in `docs/iris/example_tests`), and a few  other 'legacy' style tests
(as described in :ref:`developer_tests`).
It is conceivable that new 'graphics tests' of this sort can still be added.
However, as graphics tests are inherently "integration" style rather than true
unit tests, results can differ with the installed versions of dependent
libraries (see below), so this is not recommended except where no alternative
is practical.

Testing actual plot results introduces some significant difficulties :
 * Graphics tests are inherently 'integration' style tests, so results will
   often vary with the versions of key dependencies, i.e. the exact versions of
   third-party modules which are installed :  Obviously, results will depend on
   the matplotlib version, but they can also depend on numpy and other
   installed packages.
 * Although it seems possible in principle to accommodate 'small' result changes
   by distinguishing plots which are 'nearly the same' from those which are
   'significantly different', in practice no *automatic* scheme for this can be
   perfect :  That is, any calculated tolerance in output matching will allow
   some changes which a human would judge as a significant error.
 * Storing a variety of alternative 'acceptable' results as reference images
   can easily lead to uncontrolled increases in the size of the repository,
   given multiple independent sources of variation.


Graphics Testing Strategy
=========================

In the Iris Travis matrix, and over time, graphics tests must run with
multiple versions of Python, and of key dependencies such as matplotlib.
To make this manageable, the "check_graphic" test routine tests against
multiple alternative 'acceptable' results.  It does this using an image "hash"
comparison technique which avoids storing reference images in the Iris
repository itself, to avoid space problems.

This consists of :

 * The 'check_graphic' function uses a perceptual 'image hash' of the outputs
   (see https://github.com/JohannesBuchner/imagehash) as the basis for checking
   test results.
 * The hashes of known 'acceptable' results for each test are stored in a
   lookup dictionary, saved to the repo file
   ``lib/iris/tests/results/imagerepo.json`` .
 * An actual reference image for each hash value is stored in a *separate*
   public repository : https://github.com/SciTools/test-iris-imagehash .
 * The reference images allow human-eye assessment of whether a new output is
   judged to be 'close enough' to the older ones, or not.
 * The utility script ``iris/tests/idiff.py`` automates checking, enabling the
   developer to easily compare proposed new 'acceptable' result images against the
   existing accepted reference images, for each failing test.


How to Add New 'Acceptable' Result Images to Existing Tests
========================================

When you find that a graphics test in the Iris testing suite has failed,
following changes in Iris or the run dependencies, this is the process
you should follow:

#. Create a new, empty directory to store temporary image results, at the path
   ``lib/iris/tests/result_image_comparison`` in your Iris repository checkout.

#. **In your Iris repo root directory**, run the relevant (failing) tests
   directly as python scripts, or by using a command such as
   ``python -m unittest discover paths/to/test/files``.

#. **In the** ``iris/lib/iris/tests`` **folder**,  run the command: ``python idiff.py``.
   This will open a window for you to visually inspect side-by-side 'old', 'new'
   and 'difference' images for each failed graphics test.
   Hit a button to either "accept", "reject" or "skip" each new result ...

   * If the change is *"accepted"* :

     * the imagehash value of the new result image is added into the relevant
       set of 'valid result hashes' in the image result database file,
       ``tests/results/imagerepo.json`` ;

     * the relevant output file in ``tests/result_image_comparison`` is
       renamed according to the image hash value, as ``<hash>.png``.
       A copy of this new PNG file must then be added into the reference image
       repository at https://github.com/SciTools/test-iris-imagehash.
       (See below).

   * If a change is *"skipped"* :

     * no further changes are made in the repo.

     * when you run idiff again, the skipped choice will be presented again.

   * If a change is *"rejected"* :

     * the output image is deleted from ``result_image_comparison``.

     * when you run idiff again, the skipped choice will not appear, unless
       and until the relevant failing test is re-run.

#. Now re-run the tests.  The 'new' result should now be recognised and the
   relevant test should pass.  However, some tests can perform *multiple* graphics
   checks within a single testcase function : In those cases, any failing
   check will prevent the following ones from being run, so a test re-run may
   encounter further (new) graphical test failures.  If that happens, simply
   repeat the check-and-accept process until all tests pass.

#. To add your changes to Iris, you need to make two pull requests :

   * (1) The first PR is made in the test-iris-imagehash repository, at
     https://github.com/SciTools/test-iris-imagehash.

     *  First, add all the newly-generated referenced PNG files into the
        ``images/v4`` directory.  In your Iris repo, these files are to be found
        in the temporary results folder ``iris/tests/result_image_comparison``.

        .. Note::

           The ``result_image_comparison`` folder is covered by a project
           ``.gitignore`` setting, so those files *will not show up* in a
           ``git status`` check.

     *  Then, run ``python recreate_v4_files_listing.py``, to update the file
        which lists available images, ``v4_files_listing.txt``.

     *  Create a PR proposing these changes, in the usual way.

   * (2) The second PR is created in the Iris repository, and
     should only include the change to the image results database,
     ``tests/results/imagerepo.json`` :
     The description box of this pull request should contain a reference to
     the matching one in test-iris-imagehash.

Note: the Iris pull-request will not test out successfully in Travis until the
test-iris-imagehash pull request has been merged :  This is because there is
an Iris test which ensures the existence of the reference images (uris) for all
the targets in the image results database.  N.B. likewise, it will *also* fail
if you forgot to run ``recreate_v4_files_listing.py`` to update the image-listing
file in test-iris-imagehash.

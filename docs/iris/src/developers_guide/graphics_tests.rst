.. _developer_graphics_tests:

Graphics tests
**************

The only practical way of testing plotting functionality is to check actual
output plots.
For this, a basic 'graphics test' assertion operation is provided in the method
:method:`iris.tests.IrisTest.check_graphic` :  This tests plotted output for a
match against a stored reference.
A "graphics test" is any test which employs this.

At present (Iris version 1.10), such tests include the testing for modules
`iris.tests.test_plot` and `iris.tests.test_quickplot`, and also some other
'legacy' style tests (as described in :ref:`developer_tests`).
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

Prior to Iris 1.10, all graphics tests compared against a stored reference
image with a small tolerance on pixel values.

From Iris v1.11 onward, we want to support testing Iris against multiple
versions of matplotlib (and some other dependencies).  
To make this manageable, we have now rewritten "check_graphic" to allow
multiple alternative 'correct' results without including many more images in
the Iris repository.  
This consists of :

 * using a perceptual 'image hash' of the outputs (see
   <<https://github.com/JohannesBuchner/imagehash>) as the basis for checking
   test results.
 * storing the hashes of 'known accepted results' for each test in a
   database in the repo (which is actually stored in 
   ``lib/iris/tests/results/imagerepo.json``).
 * storing associated reference images for each hash value in a separate public
   repository, currently in https://github.com/SciTools/test-images-scitools ,
   allowing human-eye judgement of 'valid equivalent' results.
 * a new version of the 'iris/tests/idiff.py' assists in comparing proposed
   new 'correct' result images with the existing accepted ones.

BRIEF...
There should be sufficient work-flow detail here to allow an iris developer to:
    * understand the new check graphic test process
    * understand the steps to take and tools to use to add a new graphic test
    * understand the steps to take and tools to use to diagnose and fix an graphic test failure


Basic workflow
==============
#   If you notice that a graphics test in the Iris testing suite has failed
    following changes in Iris or any of its dependencies, this is the process
    you now need to follow:

#1. Create a directory in iris/lib/iris/tests called 'result_image_comparison'.
#2. From your Iris root directory, run the tests by using the command:
    ``python setup.py test``.
#3. Navigate to iris/lib/iris/tests and run the command: ``python idiff.py``.
    This will open a window for you to visually inspect the changes to the
    graphic and then either accept or reject the new result.
#4. Upon acceptance of a change or a new image, a copy of the output PNG file
    is added to the reference image repository in
    https://github.com/SciTools/test-images-scitools.  The file is named
    according to the image hash value, as ``<hash>.png``.
#5. The hash value of the new result is added into the relevant set of 'valid
    result hashes' in the image result database file,
    ``tests/results/imagerepo.json``.
#6. The tests must now be re-run, and the 'new' result should be accepted.
    Occasionally there are several graphics checks in a single test, only the
    first of which will be run should it fail.  If this is the case, then you
    may well encounter further graphical test failures in your next runs, and
    you must repeat the process until all the graphical tests pass.
#7. To add your changes to Iris, you need to make two pull requests.  The first
    should be made to the test-images-scitools repository, and this should
    contain all the newly-generated png files copied into the folder named
    'image_files'.
#8. The second pull request should be created in the Iris repository, and should
    only include the change to the image results database
    (``tests/results/imagerepo.json``) :
    This pull request must contain a reference to the matching one in
    test-images-scitools.

Note: the Iris pull-request will not test out successfully in Travis until the
test-images-scitools pull request has been merged :  This is because there is
an Iris test which ensures the existence of the reference images (uris) for all
the targets in the image results database.


Fixing a failing graphics test
==============================


Adding a new graphics test
==========================

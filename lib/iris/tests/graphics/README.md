# Graphics Tests

Iris may be used to create various forms of graphical output; to ensure
the output is consistent, there are automated tests to check against
known acceptable graphical output.

At present graphical tests are used in the following areas of Iris:

* Module `iris.tests.test_plot`
* Module `iris.tests.test_quickplot`
* Gallery plots contained in `docs/gallery_tests`.


## Challenges

Iris uses many dependencies that provide functionality, an example that
applies here is `matplotlib`. When there are updates to `matplotlib` or a
dependency of it, this may result in a change in the rendered graphical
output. This means that there may be no changes to `Iris`, but due to an
updated dependency any automated tests that compare a graphical output to a
known acceptable output may fail.  The failure may also not be visually
perceived as it may be a simple pixel shift.


## Testing Strategy

The `iris.tests.IrisTest.check_graphic` test routine calls out to
`iris.tests.graphics.check_graphic` which tests against the **acceptable**
result. It does this using an image **hash** comparison technique which allows
us to be robust against minor variations based on underlying library updates.

This consists of:

* The `graphics.check_graphic` function uses a perceptual
  **image hash** of the outputs (see https://github.com/JohannesBuchner/imagehash)
  as the basis for checking test results.

* The hashes of known **acceptable** results for each test are stored in a
  lookup dictionary, saved to the repo file
  `lib/iris/tests/results/imagerepo.json`
  (`link <https://github.com/SciTools/iris/blob/main/lib/iris/tests/results/imagerepo.json>`_) .

* An actual baseline image for each hash value is stored in the test data
  repository (`link <https://github.com/SciTools/iris-test-data>`_).

* The baseline images allow human-eye assessment of whether a new output is
  judged to be close enough to the older ones, or not.

* The utility script `iris/tests/idiff.py` automates checking, enabling the
  developer to easily compare the proposed new **acceptable** result image
  against the existing accepted baseline image, for each failing test.
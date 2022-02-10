.. include:: ../common_links.inc

.. _testing.imagehash_index:

Graphical Test Hash Index
*************************

The iris test suite produces plots of data using matplotlib and cartopy.
The images produced are compared to known "good" output, the images for
which are kept in `scitools/test-iris-imagehash <https://github.com/scitools/test-iris-imagehash>`_.

For an overview of iris' graphics tests, see :ref:`testing.graphics`

Typically running the iris test suite will output the rendered
images to ``$PROJECT_DIR/iris_image_test_output``.
The known good output for each test can be seen at the links below
for comparison.


.. imagetest-list::
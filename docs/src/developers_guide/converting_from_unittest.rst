.. include:: ../common_links.inc

.. _converting_tests:

*******************************************
Converting From ``unittest`` to ``pytest``
*******************************************

Before making any manual changes, we recommend running [pytestify](https://github.com/dannysepler/pytestify)
on the file. This does a lot of the brunt work for you!
Additionally, once you've made your changes, we recommend checking them with ruff,
``pip install ruff`` -> ``ruff check --select PT <file>``.


Conversion Checklist
--------------------
.. note::
    Please bear in mind the following checklist is for general use; there may be
    some cases which require extra context or thought before implementing these changes.

#. Check for references to :class:`iris.tests.IrisTest`. If a class inherits
   from this, remove it. :class:`iris.tests.IrisTest` has been deprecated, and
   replaced with the :mod:`iris.tests._shared_utils` module.
   Some :class:`iris.tests.IrisTest` modules have not been converted into
   :mod:`iris.tests._shared_utils`, as they were deemed easily done without a
   specialised function.
#. Check for references  to ``unittest``. Many of the functions within unittest
   are also in pytest, so often you can just change where the function is imported
   from.
#. Check for references to `self.assert`. Pytest has a lighter-weight syntax for
   assertions, e.g. ``assert x == 2`` instead of ``assertEqual(x, 2)``.
#. Check for references to ``setUp()``. Pytest recognises a specific method called
   ``_setup()`` instead. Ensure that this is decorated with
   ``@pytest.fixture(autouse=True)``.
#. Check for references to ``@tests``. These should be changed to ``@_shared_utils``.
#. Check for references to ``with mock.patch("...")``. These should be replaced with
   ``mocker.patch("...")``. Note, ``mocker.patch("...")`` is NOT a context manager.
#. Check for ``np.testing.assert...``. This can usually be swapped for
   ``_shared_utils.assert...``.
#. Check for references to ``super()``. Most test classes used to inherit from
   :class:`iris.tests.IrisTest`, so references to this should be removed.
#. Check for references to ``self.tmp_dir``. In pytest, ``tmp_path`` is used instead,
   and can be passed into functions as a fixture.
#. Check for ``if __name__ == 'main'``. This is no longer needed with pytest.


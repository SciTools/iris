.. include:: ../common_links.inc

.. _contributing_pytest_conversions:

*******************************************
Converting From ``unittest`` to ``pytest``
*******************************************

Conversion Checklist
--------------------
.. note::
    Please bear in mind the following checklist is for general use; there may be
    some cases which require extra context or thought before implementing these changes.

#. Before making any manual changes, run https://github.com/dannysepler/pytestify
   on the file. This does a lot of the brunt work for you!
#. Check for references to :class:`iris.tests.IrisTest`. If a class inherits
   from this, remove the inheritance. Inheritance is unnecessary for
   pytest tests, so :class:`iris.tests.IrisTest` has been deprecated
   and its convenience methods have been moved to the
   :mod:`iris.tests._shared_utils` module.
#. Check for references  to ``unittest``. Many of the functions within unittest
   are also in pytest, so often you can just change where the function is imported
   from.
#. Check for references to ``self.assert``. Pytest has a lighter-weight syntax for
   assertions, e.g. ``assert x == 2`` instead of ``assertEqual(x, 2)``. In the
   case of custom :class:`~iris.tests.IrisTest` assertions, the majority of these
   have been replicated in
   :mod:`iris.tests._shared_utils`, but with snake_case instead of camelCase.
   Some :class:`iris.tests.IrisTest` assertions have not been converted into
   :mod:`iris.tests._shared_utils`, as these were deemed easy to achieve via
   simple ``assert ...`` statements.
#. Check for references to ``setUp()``. Replace this with ``_setup()`` instead.
   Ensure that this is decorated with ``@pytest.fixture(autouse=True)``.

   .. code-block:: python

      @pytest.fixture(autouse=True)
      def _setup(self):
         ...

#. Check for references to ``@tests``. These should be changed to ``@_shared_utils``.
#. Check for references to ``with mock.patch("...")``. These should be replaced with
   ``with mocker.patch("...")``. ``mocker`` is a fixture, should be passed into
   relevant functions as a parameter.
#. Check for ``np.testing.assert...``. This can usually be swapped for
   ``_shared_utils.assert...``.
#. Check for references to ``super()``. Most test classes used to inherit from
   :class:`iris.tests.IrisTest`, so references to this should be removed.
#. Check for references to ``self.tmp_dir``. In pytest, ``tmp_path`` is used instead,
   and can be passed into functions as a fixture.
#. Check for ``if __name__ == 'main'``. This is no longer needed with pytest.
#. Check for ``mock.patch("warnings.warn")``. This can be replaced with
   ``pytest.warns(match=message)``.
#. Check the file against https://github.com/astral-sh/ruff , using ``pip install ruff`` ->
   ``ruff check --select PT <file>``.

Common Translations
-------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ``unittest`` method
     - ``pytest`` equivalent
   * - ``assertTrue(x)``
     - ``assert x``
   * - ``assertFalse(x)``
     - ``assert not x``
   * - ``assertRegex(x, y)``
     - ``assert re.match(y, x)``
   * - ``assertRaisesRegex(cls, msg_re)``
     - ``with pytest.raises(cls, match=msg_re):``
   * - ``mock.patch(...)``
     - ``mocker.patch(...)``
   * - ``with mocker.patch.object(...) as x:``
     - ``x = mocker.patch.object(...)``


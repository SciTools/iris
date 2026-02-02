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
#. Check for references to :class:`IrisTest`. If a class inherits
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
#. Check for references to ``setup_method()``. Replace this with ``_setup()`` instead.
   Ensure that this is decorated with ``@pytest.fixture(autouse=True)``.

   .. code-block:: python

      @pytest.fixture(autouse=True)
      def _setup(self): ...

#. Check for references to ``super()``. Most test classes used to inherit from
   :class:`iris.tests.IrisTest`, so references to this should be removed. Any
   inheritance patterns in ``_setup()`` (see above) can be achieved by
   'chaining' fixtures; note that each setup fixture needs a unique name:

   .. code-block:: python

      class TestFoo:
          @pytest.fixture(autouse=True)
          def _setup_foo(self): ...


      class TestBar(TestFoo):
          @pytest.fixture(autouse=True)
          def _setup(self, _setup_foo): ...

#. Check for references to ``@tests``. These should be changed to ``@_shared_utils``.
#. Check for ``mock.patch("warnings.warn")``. This can be replaced with
   ``pytest.warns(match=message)``.
#. Check for references to ``mock`` or ``self.patch``. These should be changed to use
   the ``mocker`` fixture - see the `pytest-mock docs`_. Note that pytest-mock's
   ``patch`` does not support the context-manager syntax; in most cases this is made
   unnecessary (see `Usage as context manager`_), in advanced cases consider using
   the `monkeypatch`_ fixture to provide a context-manager.
#. Check for ``np.testing.assert...``. This can usually be swapped for
   ``_shared_utils.assert...``.
#. Check for ``np.allclose``. This should be swapped for
   ``_shared_utils.assert_array_all_close``.
#. Check for references to ``self.tmp_dir`` and ``self.temp_filename``. In
   pytest, ``tmp_path`` is used instead, and can be passed into functions as a
   fixture.
#. Check for ``if __name__ == 'main'``. This is no longer needed with pytest.
#. Remove the top-level import of :mod:`iris.tests` (usually ``import iris.tests as tests``).
   Having followed the above steps, any remaining calls
   (e.g. :func:`iris.tests.get_data_path`) should be easily replacable with calls to
   :mod:`iris.tests._shared_utils` (e.g. :func:`iris.tests._shared_utils.get_data_path`).
#. Ensure that all test classes start with ``Test``. Tests will not run in pytest without it.
#. Check the file against https://github.com/astral-sh/ruff , using ``pip install ruff`` ->
   ``ruff check --select PT <file>``.
#. Ensure that all the tests are *passing*. Some tests are set to skip if certain packages
   aren't installed in your environment. These are often also skipped in the Iris CI also,
   so make sure that they run and pass locally.

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
   * - ``with mock.patch.object(...) as x:``
     - ``x = mocker.patch.object(...)``


.. _pytest-mock docs: https://pytest-mock.readthedocs.io/en/latest/index.html
.. _Usage as context manager: https://pytest-mock.readthedocs.io/en/latest/usage.html#usage-as-context-manager
.. _monkeypatch: https://docs.pytest.org/en/stable/how-to/monkeypatch.html

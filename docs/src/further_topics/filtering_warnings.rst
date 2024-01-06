.. _filtering-warnings:

==================
Filtering Warnings
==================

Since Iris cannot predict your specific needs, it by default raises Warnings
for anything that might be a problem for **any** user, and is designed to work with
you to ``ignore`` Warnings which you do not find helpful.

.. testsetup:: filtering_warnings

    from pathlib import Path
    import sys
    import warnings

    import iris
    import iris.coord_systems
    import iris.exceptions

    # Hack to ensure doctests actually see Warnings that are raised, and that
    #  they have a relative path (so a test pass is not machine-dependent).
    warnings.filterwarnings("default")
    IRIS_FILE = Path(iris.__file__)
    def custom_warn(message, category, filename, lineno, file=None, line=None):
        filepath = Path(filename)
        filename = str(filepath.relative_to(IRIS_FILE.parents[1]))
        sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
    warnings.showwarning = custom_warn

    geog_cs_globe = iris.coord_systems.GeogCS(6400000)
    orthographic_coord_system = iris.coord_systems.Orthographic(
        longitude_of_projection_origin=0,
        latitude_of_projection_origin=0,
        ellipsoid=geog_cs_globe,
    )


    def my_operation():
        geog_cs_globe.inverse_flattening = 0.1
        _ = orthographic_coord_system.as_cartopy_crs()

Here is a hypothetical operation - ``my_operation()`` - which raises two
Warnings:

.. doctest:: filtering_warnings

    >>> my_operation()
    ...
    iris/coord_systems.py:432: IrisUserWarning: Setting inverse_flattening does not affect other properties of the GeogCS object. To change other properties set them explicitly or create a new GeogCS instance.
      warnings.warn(wmsg, category=iris.exceptions.IrisUserWarning)
    iris/coord_systems.py:770: IrisDefaultingWarning: Discarding false_easting and false_northing that are not used by Cartopy.
      warnings.warn(

Warnings can be suppressed using the Python warnings filter with the ``ignore``
action. Detailed information is available in the Python documentation:
:external+python:mod:`warnings`.

The key points are:

- :ref:`When<warning-filter-application>`: a warnings filter can be applied
  either from the command line or from within Python.
- :ref:`What<warning-filter-specificity>`: a warnings filter accepts
  various arguments to specify which Warnings are being filtered. Both broad
  and narrow filters are possible.

.. _warning-filter-application:

**When** a Warnings Filter can be Applied
-----------------------------------------

- **Command line:** setting the :external+python:envvar:`PYTHONWARNINGS`
  environment variable.
- **Command line:** the `python -W <https://docs.python.org/3/using/cmdline.html#cmdoption-W>`_
  command line argument.
- **Within Python:** use :func:`warnings.filterwarnings` .

The :ref:`warning-filter-specificity` section demonstrates using
:func:`warnings.filterwarnings`, and shows the equivalent **command line**
approaches.


.. _warning-filter-specificity:

**What** Warnings will be Filtered
----------------------------------

.. note::

    For all of these examples we are using the
    :class:`~warnings.catch_warnings` context manager to ensure any changes to
    settings are temporary.

    This should always work fine for the ``ignore``
    warning filter action, but note that some of the other actions
    may not behave correctly with all Iris operations, as
    :class:`~warnings.catch_warnings` is not thread-safe (e.g. using the
    ``once`` action may cause 1 warning per chunk of lazy data).

Specific Warnings
~~~~~~~~~~~~~~~~~

**When you do not want a specific warning, but still want all others.**

You can target specific Warning messages, e.g.

.. doctest:: filtering_warnings

    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", message="Discarding false_easting")
    ...     my_operation()
    ...
    iris/coord_systems.py:432: IrisUserWarning: Setting inverse_flattening does not affect other properties of the GeogCS object. To change other properties set them explicitly or create a new GeogCS instance.
      warnings.warn(wmsg, category=iris.exceptions.IrisUserWarning)

::

    python -W ignore:"Discarding false_easting"
    export PYTHONWARNINGS=ignore:"Discarding false_easting"

----

Or you can target Warnings raised by specific lines of specific modules, e.g.

.. doctest:: filtering_warnings

    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", module="iris.coord_systems", lineno=449)
    ...     my_operation()
    ...
    iris/coord_systems.py:432: IrisUserWarning: Setting inverse_flattening does not affect other properties of the GeogCS object. To change other properties set them explicitly or create a new GeogCS instance.
      warnings.warn(wmsg, category=iris.exceptions.IrisUserWarning)
    iris/coord_systems.py:770: IrisDefaultingWarning: Discarding false_easting and false_northing that are not used by Cartopy.
      warnings.warn(

::

    python -W ignore:::iris.coord_systems:453
    export PYTHONWARNINGS=ignore:::iris.coord_systems:453

Warnings from a Common Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When you do not want ANY warnings raised by a module, or collection of
modules.**

E.g. filtering the ``coord_systems`` module:

.. doctest:: filtering_warnings

    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", module="iris.coord_systems")
    ...     my_operation()

::

    python -W ignore:::iris.coord_systems
    export PYTHONWARNINGS=ignore:::iris.coord_systems

----

If using :func:`warnings.filterwarnings` , you can also use partial
definitions. The below example will ``ignore`` all Warnings from ``iris`` as a
whole.

.. doctest:: filtering_warnings

    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", module="iris")
    ...     my_operation()

The above 'partial' filter is not available with the command line approaches.

Warnings of a Common Type
~~~~~~~~~~~~~~~~~~~~~~~~~

**When you do not want any Warnings of the same nature, from anywhere in the
code you are calling.**

The below example will ``ignore`` any
:class:`~iris.exceptions.IrisDefaultingWarning` that gets raised by *any*
module during execution:

.. doctest:: filtering_warnings

    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings(
    ...         "ignore",
    ...         category=iris.exceptions.IrisDefaultingWarning
    ...     )
    ...     my_operation()
    ...
    iris/coord_systems.py:432: IrisUserWarning: Setting inverse_flattening does not affect other properties of the GeogCS object. To change other properties set them explicitly or create a new GeogCS instance.
      warnings.warn(wmsg, category=iris.exceptions.IrisUserWarning)

----

Using :class:`~iris.exceptions.IrisUserWarning` in the filter will ``ignore``
both Warnings, since :class:`~iris.exceptions.IrisDefaultingWarning` subclasses
:class:`~iris.exceptions.IrisUserWarning` :

.. doctest:: filtering_warnings

    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings(
    ...         "ignore",
    ...         category=iris.exceptions.IrisUserWarning
    ...     )
    ...     my_operation()

----

The command line approaches can only handle the built-in Warning
categories (`cpython#66733`_)::

    python -W ignore::UserWarning
    export PYTHONWARNINGS=ignore::UserWarning

----

There are several built-in Python warning categories that can be used here
(:class:`DeprecationWarning` being a popular example, see
:external+python:mod:`warnings` for more). Since Iris has
so many different warnings that might be raised, Iris subclasses
:class:`UserWarning` to :class:`~iris.exceptions.IrisUserWarning`, which itself
has **many** specialised subclasses. These subclasses exist to give you more
granularity in your warning filtering; you can see the full list by
searching the :mod:`iris.exceptions` page for ``warning`` .

.. attention::

    If you have ideas for adding/altering Iris' warning categories, please
    :ref:`get in touch<development_where_to_start>`! The categories exist to
    make your life easier, and it is simple to make modifications.


More Detail
-----------

Different people use Iris for very different purposes, from quick file
visualisation to extract-transform-load to statistical analysis. These
contrasting priorities mean disagreement on which Iris problems can be ignored
and which are critically important.

For problems that prevent Iris functioning: **Concrete Exceptions** are raised, which
stop code from running any further - no debate here. For less catastrophic
problems: **Warnings** are raised,
which notify you (in ``stderr``) but allow code to continue running. The Warnings are
there because Iris may **OR may not** function in the way you expect,
depending on what you need - e.g. a problem might prevent data being saved to
NetCDF, but statistical analysis will still work fine.

Examples of Iris Warnings
~~~~~~~~~~~~~~~~~~~~~~~~~

- If you attempt to plot un-bounded point data as a ``pcolormesh``: Iris will
  guess appropriate bounds around each point so that quadrilaterals can be
  plotted. This permanently modifies the relevant coordinates, so the you are
  warned in case downstream operations assume un-bounded coordinates.
- If you load a NetCDF file where a CF variable references another variable -
  e.g. ``my_var:coordinates = "depth_var" ;`` - but the referenced variable
  (``depth_var``) is not in the file: Iris will still construct
  its data model, but without this reference relationship. You are warned since
  the file includes an error and the loaded result might therefore not be as
  expected.


.. testcleanup:: filtering_warnings

    warnings.filterwarnings("ignore")


.. _cpython#66733: https://github.com/python/cpython/issues/66733

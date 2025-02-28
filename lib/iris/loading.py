# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iris general file loading mechanism."""

from dataclasses import dataclass
import itertools
from pathlib import Path
from traceback import TracebackException
from typing import Any, Iterable

from iris.common import CFVariableMixin, LimitedAttributeDict


def _generate_cubes(uris, callback, constraints):
    import iris.io

    """Return a generator of cubes given the URIs and a callback."""
    if isinstance(uris, str) or not isinstance(uris, Iterable):
        # Make a string, or other single item, into an iterable.
        uris = [uris]

    # Group collections of uris by their iris handler
    # Create list of tuples relating schemes to part names
    uri_tuples = sorted(iris.io.decode_uri(uri) for uri in uris)

    for scheme, groups in itertools.groupby(uri_tuples, key=lambda x: x[0]):
        # Call each scheme handler with the appropriate URIs
        if scheme == "file":
            part_names = [x[1] for x in groups]
            for cube in iris.io.load_files(part_names, callback, constraints):
                yield cube
        elif scheme in ["http", "https"]:
            urls = [":".join(x) for x in groups]
            for cube in iris.io.load_http(urls, callback):
                yield cube
        elif scheme == "data":
            data_objects = [x[1] for x in groups]
            for cube in iris.io.load_data_objects(data_objects, callback):
                yield cube
        else:
            raise ValueError("Iris cannot handle the URI scheme: %s" % scheme)


class _CubeFilter:
    """A constraint, paired with a list of cubes matching that constraint."""

    def __init__(self, constraint, cubes=None):
        from iris.cube import CubeList

        self.constraint = constraint
        if cubes is None:
            cubes = CubeList()
        self.cubes = cubes

    def __len__(self):
        return len(self.cubes)

    def add(self, cube):
        """Add the appropriate (sub)cube to the list of cubes where it matches the constraint."""
        sub_cube = self.constraint.extract(cube)
        if sub_cube is not None:
            self.cubes.append(sub_cube)

    def combined(self, unique=False):
        """Return a new :class:`_CubeFilter` by combining the list of cubes.

        Combines the list of cubes with :func:`~iris._combine_load_cubes`.

        Parameters
        ----------
        unique : bool, default=False
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        from iris._combine import _combine_load_cubes

        return _CubeFilter(
            self.constraint,
            _combine_load_cubes(self.cubes, merge_require_unique=unique),
        )


class _CubeFilterCollection:
    """A list of _CubeFilter instances."""

    @staticmethod
    def from_cubes(cubes, constraints=None):
        """Create a new collection from an iterable of cubes, and some optional constraints."""
        from iris._constraints import list_of_constraints

        constraints = list_of_constraints(constraints)
        pairs = [_CubeFilter(constraint) for constraint in constraints]
        collection = _CubeFilterCollection(pairs)
        for c in cubes:
            collection.add_cube(c)
        return collection

    def __init__(self, pairs):
        self.pairs = pairs

    def add_cube(self, cube):
        """Add the given :class:`~iris.cube.Cube` to all of the relevant constraint pairs."""
        for pair in self.pairs:
            pair.add(cube)

    def cubes(self):
        """Return all the cubes in this collection in a single :class:`CubeList`."""
        from iris.cube import CubeList

        result = CubeList()
        for pair in self.pairs:
            result.extend(pair.cubes)
        return result

    def combined(self, unique=False):
        """Return a new :class:`_CubeFilterCollection` by combining all the cube lists of this collection.

        Combines each list of cubes using :func:`~iris._combine_load_cubes`.

        Parameters
        ----------
        unique : bool, default=False
            If True, raises `iris.exceptions.DuplicateDataError` if
            duplicate cubes are detected.

        """
        return _CubeFilterCollection([pair.combined(unique) for pair in self.pairs])


def _load_collection(uris, constraints=None, callback=None):
    import iris.exceptions
    from iris.fileformats.rules import _MULTIREF_DETECTION

    try:
        # This routine is called once per iris load operation.
        # Control of the "multiple refs" handling is implicit in this routine
        # NOTE: detection of multiple reference fields, and it's enabling of post-load
        # concatenation, is triggered **per-load, not per-cube**
        # This behaves unexpectedly for "iris.load_cubes" : a post-concatenation is
        # triggered for all cubes or none, not per-cube (i.e. per constraint).
        _MULTIREF_DETECTION.found_multiple_refs = False

        cubes = _generate_cubes(uris, callback, constraints)
        result = _CubeFilterCollection.from_cubes(cubes, constraints)
    except EOFError as e:
        raise iris.exceptions.TranslationError(
            "The file appears empty or incomplete: {!r}".format(str(e))
        )
    return result


def load(uris, constraints=None, callback=None):
    """Load any number of Cubes for each constraint.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Parameters
    ----------
    uris : str or :class:`pathlib.PurePath`
        One or more filenames/URIs, as a string or :class:`pathlib.PurePath`.
        If supplying a URL, only OPeNDAP Data Sources are supported.
    constraints : optional
        One or more constraints.
    callback : optional
        A modifier/filter function.

    Returns
    -------
    :class:`iris.cube.CubeList`
        An :class:`iris.cube.CubeList`. Note that there is no inherent order
        to this :class:`iris.cube.CubeList` and it should be treated as if it
        were random.

    """
    cubes = _load_collection(uris, constraints, callback).combined().cubes()
    return cubes


def load_cube(uris, constraint=None, callback=None):
    """Load a single cube.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Parameters
    ----------
    uris :
        One or more filenames/URIs, as a string or :class:`pathlib.PurePath`.
        If supplying a URL, only OPeNDAP Data Sources are supported.
    constraints : optional
        A constraint.
    callback : optional
        A modifier/filter function.

    Returns
    -------
    :class:`iris.cube.Cube`

    """
    import iris._constraints
    import iris.exceptions

    constraints = iris._constraints.list_of_constraints(constraint)
    if len(constraints) != 1:
        raise ValueError("only a single constraint is allowed")

    cubes = _load_collection(uris, constraints, callback).combined(unique=False).cubes()

    try:
        # NOTE: this call currently retained to preserve the legacy exceptions
        # TODO: replace with simple testing to duplicate the relevant error cases
        cube = cubes.merge_cube()
    except iris.exceptions.MergeError as e:
        raise iris.exceptions.ConstraintMismatchError(str(e))
    except ValueError:
        raise iris.exceptions.ConstraintMismatchError("no cubes found")

    return cube


def load_cubes(uris, constraints=None, callback=None):
    """Load exactly one Cube for each constraint.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Parameters
    ----------
    uris :
        One or more filenames/URIs, as a string or :class:`pathlib.PurePath`.
        If supplying a URL, only OPeNDAP Data Sources are supported.
    constraints : optional
        One or more constraints.
    callback : optional
        A modifier/filter function.

    Returns
    -------
    :class:`iris.cube.CubeList`
        An :class:`iris.cube.CubeList`. Note that there is no inherent order
        to this :class:`iris.cube.CubeList` and it should be treated as if it
        were random.

    """
    import iris.exceptions

    # Merge the incoming cubes
    collection = _load_collection(uris, constraints, callback).combined()

    # Make sure we have exactly one merged cube per constraint
    bad_pairs = [pair for pair in collection.pairs if len(pair) != 1]
    if bad_pairs:
        fmt = "   {} -> {} cubes"
        bits = [fmt.format(pair.constraint, len(pair)) for pair in bad_pairs]
        msg = "\n" + "\n".join(bits)
        raise iris.exceptions.ConstraintMismatchError(msg)

    return collection.cubes()


def load_raw(uris, constraints=None, callback=None):
    """Load non-merged cubes.

    This function is provided for those occasions where the automatic
    combination of cubes into higher-dimensional cubes is undesirable.
    However, it is intended as a tool of last resort! If you experience
    a problem with the automatic combination process then please raise
    an issue with the Iris developers.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Parameters
    ----------
    uris :
        One or more filenames/URIs, as a string or :class:`pathlib.PurePath`.
        If supplying a URL, only OPeNDAP Data Sources are supported.
    constraints : optional
        One or more constraints.
    callback : optional
        A modifier/filter function.

    Returns
    -------
    :class:`iris.cube.CubeList`

    """
    from iris.fileformats.um._fast_load import _raw_structured_loading

    with _raw_structured_loading():
        return _load_collection(uris, constraints, callback).cubes()


from iris._combine import CombineOptions


class LoadPolicy(CombineOptions):
    """A control object for Iris loading options.

    Incorporates all the settings of a :class:`~iris.CombineOptions`.

    Examples
    --------
    >>> LOAD_POLICY.set("legacy")
    >>> print(LOAD_POLICY)
    LoadPolicy(support_multiple_references=False, merge_concat_sequence='m', repeat_until_unchanged=False)
    >>> LOAD_POLICY.support_multiple_references = True
    >>> print(LOAD_POLICY)
    LoadPolicy(support_multiple_references=True, merge_concat_sequence='m', repeat_until_unchanged=False)
    >>> LOAD_POLICY.set(merge_concat_sequence="cm")
    >>> print(LOAD_POLICY)
    LoadPolicy(support_multiple_references=True, merge_concat_sequence='cm', repeat_until_unchanged=False)
    >>> with LOAD_POLICY.context("comprehensive"):
    ...    print(LOAD_POLICY)
    LoadPolicy(support_multiple_references=True, merge_concat_sequence='mc', repeat_until_unchanged=True)
    >>> print(LOAD_POLICY)
    LoadPolicy(support_multiple_references=True, merge_concat_sequence='cm', repeat_until_unchanged=False)

    """

    pass


#: A control object containing the current file loading strategy options.
LOAD_POLICY = LoadPolicy()


@dataclass
class LoadProblemsEntry:
    loaded: CFVariableMixin | dict[str, Any] | None
    """The object that experienced loading problems.

    Four possible types:

    - :class:`~iris.cube.Cube`: if problems occurred while building a
      :class:`~iris.common.mixin.CFVariableMixin` -
      currently the only handled case is
      :class:`~iris.coords.DimCoord` - then the information will be stored in
      a 'bare bones' :class:`~iris.cube.Cube` containing only the
      :attr:`~iris.cube.Cube.data` array and the attributes. The attributes
      are un-parsed (they can still contain ``_FillValue`` etcetera), and are
      stored under a special key in the Cube :attr:`~iris.cube.Cube.attributes`
      dictionary: :attr:`~iris.common.mixin.LimitedAttributeDict.IRIS_RAW`.
    - :class:`dict`: if problems occurred while building objects from NetCDF
      attributes - currently the only handled cases are ``standard_name``,
      ``long_name``, ``var_name``. The
      dictionary key is the key of the attribute, and the value is the raw
      attribute returned by the ``netCDF4`` library.
    - Built objects, such as :class:`~iris.coords.DimCoord`: if the object was
      built successfully, but could not be added to the Cube being loaded.
    - ``None``: if a loading error occurred, but problems occurred while trying
      to store the problem object.
    """

    stack_trace: TracebackException
    """The traceback exception that was raised during loading.

    This instance contains rich information to support user-specific workflows,
    e.g:

    - ``"".join(stack_trace.format())``: the full stack trace as a string - the
      same way this would be seen at the command line.
    - ``stack_trace.exc_type``: the exception type e.g. :class:`ValueError`.
    """


# TODO: could this be a context manager in future?
# TODO: include in an __all__ somewhere
LOAD_PROBLEMS: dict[Path, list[LoadProblemsEntry]] = {}
"""Collections of cubes/coords/etcetera that could not be loaded correctly.

Structured as a dictionary of file paths. The dictionary values are lists of
:class:`LoadProblemsEntry` - see that docstring for more about what is stored.

Provided to increase transparency (problem objects are not simply discarded),
and to make it possible to fix loading problems without leaving the Iris API.

Examples
--------
.. dropdown:: (expand to see setup)

    ..
        Necessary as NumPy docstring doctests do not allow labelled
        testsetup/testcleanup, so this setup was clashing with other doctests in
        the same module.

    **This section is not necessary for understanding the examples.**

    >>> from pathlib import Path
    >>> from pprint import pprint
    >>> import sys
    >>> import warnings

    >>> import cf_units
    >>> import iris
    >>> import iris.common
    >>> import iris.coords
    >>> from iris.fileformats._nc_load_rules import helpers
    >>> import iris.loading
    >>> from iris import std_names

    >>> # Ensure doctests actually see Warnings that are raised, and that
    >>> #  they have a relative path (so a test pass is not machine-dependent).
    >>> showwarning_original = warnings.showwarning
    >>> warnings.filterwarnings("default")
    >>> IRIS_FILE = Path(iris.__file__)
    >>> def custom_warn(message, category, filename, lineno, file=None, line=None):
    ...     filepath = Path(filename)
    ...     filename = str(filepath.relative_to(IRIS_FILE.parents[1]))
    ...     sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
    >>> warnings.showwarning = custom_warn

    >>> build_dimension_coordinate_original = helpers.build_dimension_coordinate

    >>> def raise_example_error_dim(filename, cf_coord_var, coord_name, coord_system):
    ...     if cf_coord_var.cf_name == "time":
    ...         raise ValueError("Example dimension coordinate error")
    ...     else:
    ...         return build_dimension_coordinate_original(
    ...             filename, cf_coord_var, coord_name, coord_system
    ...         )

    >>> helpers.build_dimension_coordinate = raise_example_error_dim
    >>> air_temperature = std_names.STD_NAMES.pop("air_temperature")
    >>> iris.FUTURE.date_microseconds = True

For this example we have 'booby-trapped' the Iris loading process to force
errors to occur. When we load our first cube, we see the warning that
:class:`LOAD_PROBLEMS` has been added to:

>>> cube_a1b = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
iris/...IrisLoadWarning: Not all file objects were parsed correctly. See iris.loading.LOAD_PROBLEMS for details.
  warnings.warn(message, category=IrisLoadWarning)

Remember that Python by default suppresses duplicate warnings, so a second
load action does not raise another:

>>> cube_e1 = iris.load_cube(iris.sample_data_path("E1_north_america.nc"))

Examining the contents of :class:`LOAD_PROBLEMS` we can see that both files
experienced some problems:

>>> print([path.name for path in iris.loading.LOAD_PROBLEMS.keys()])
['A1B_north_america.nc', 'E1_north_america.nc']

Printing the A1B cube shows that the time dimension coordinate is missing:

>>> print(cube_a1b.summary(shorten=True))
air_temperature / (K)               (-- : 240; latitude: 37; longitude: 49)

Below demonstrates how to explore the captured stack traces in detail:

>>> (A1B,) = [
...     entries for path, entries in iris.loading.LOAD_PROBLEMS.items()
...     if path.name == "A1B_north_america.nc"
... ]
>>> for problem in A1B:
...     print(problem.stack_trace.exc_type)
<class 'ValueError'>
<class 'ValueError'>

>>> last_problem = A1B[-1]
>>> print("".join(last_problem.stack_trace.format()))
Traceback (most recent call last):
  File ..., in _add_or_capture
    built = build_func()
            ^^^^^^^^^^^^
  File ..., in raise_example_error_dim
ValueError: Example dimension coordinate error
<BLANKLINE>

:class:`LOAD_PROBLEMS` also captures the 'raw' information in the object that
could not be loaded - the time dimension coordinate. This is captured as a
:class:`~iris.cube.Cube`:

>>> print(last_problem.loaded)
unknown / (unknown)                 (-- : 240)
    Attributes:
        IRIS_RAW                    {'axis': 'T', ...}

Using ``last_problem.loaded``, we can manually reconstruct the missing
dimension coordinate:

>>> attributes = last_problem.loaded.attributes[
...     iris.common.LimitedAttributeDict.IRIS_RAW
... ]
>>> pprint(attributes)
{'axis': 'T',
 'bounds': 'time_bnds',
 'calendar': '360_day',
 'standard_name': 'time',
 'units': 'hours since 1970-01-01 00:00:00',
 'var_name': 'time'}

>>> units = cf_units.Unit(attributes["units"], calendar=attributes["calendar"])
>>> dim_coord = iris.coords.DimCoord(
...     points=last_problem.loaded.data,
...     standard_name=attributes["standard_name"],
...     units=units,
... )
>>> cube_a1b.add_dim_coord(dim_coord, 0)
>>> print(cube_a1b.summary(shorten=True))
air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)

Note that we were unable to reconstruct the missing bounds - ``time_bnds`` -
demonstrating that this error handling is a 'best effort' and not perfect. We
hope to continually improve it over time.

.. dropdown:: (expand to see cleanup)

    ..
        Necessary as NumPy docstring doctests do not allow labelled
        testsetup/testcleanup, so this cleanup was clashing with other doctests
        in the same module.

    **This section is not necessary for understanding the examples.**

    >>> warnings.showwarning = showwarning_original
    >>> warnings.filterwarnings("ignore")
    >>> helpers.build_dimension_coordinate = build_dimension_coordinate_original
    >>> std_names.STD_NAMES["air_temperature"] = air_temperature

"""


def _profile_load_problems() -> dict[Path, list[tuple[str | None, TracebackException]]]:
    """Return a simplified copy of the current :const:`LOAD_PROBLEMS`.

    Represent :class:`~iris.cube.Cube` s by their var_name, and :class:`dict` s
    by their keys. This avoids any large array payloads.
    """

    def _simplify_entry(
        entry: LoadProblemsEntry,
    ) -> tuple[str | None, TracebackException]:
        basic_value: str | None = None

        if hasattr(entry.loaded, "keys"):
            assert isinstance(entry.loaded, dict)
            (basic_value,) = entry.loaded.keys()

        elif hasattr(entry.loaded, "var_name"):
            assert isinstance(entry.loaded, CFVariableMixin)
            iris_raw = LimitedAttributeDict.IRIS_RAW
            raw_attributes = entry.loaded.attributes.get(iris_raw)
            # TODO: should "var_name" also be encoded as a constant?
            #  Is there a better way to store this info in the first place?
            if raw_attributes is not None:
                basic_value = raw_attributes["var_name"]
            else:
                basic_value = entry.loaded.name()

        else:
            assert entry.loaded is None

        return basic_value, entry.stack_trace

    return {
        path: [_simplify_entry(entry) for entry in entries]
        for path, entries in LOAD_PROBLEMS.items()
    }

# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iris general file loading mechanism."""

import contextlib
import itertools
from typing import Iterable


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

    def combined(self):
        """Return a new :class:`_CubeFilter` by combining the list of cubes.

        Combines the list of cubes with :func:`~iris._combine_load_cubes`.

        """
        from iris._combine import _combine_load_cubes

        return _CubeFilter(
            self.constraint,
            _combine_load_cubes(self.cubes),
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

    def combined(self):
        """Return a new :class:`_CubeFilterCollection` by combining all the cube lists of this collection.

        Combines each list of cubes using :func:`~iris._combine_load_cubes`.

        """
        return _CubeFilterCollection([pair.combined() for pair in self.pairs])


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

    cubes = _load_collection(uris, constraints, callback).combined().cubes()

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

    Incorporates all the settings of a :class:`~iris.CombineOptions`, and adds an
    additional ``support_multiple_references`` option.

    Also adds :meth:`~iris.LoadPolicy.context`, allowing loading behaviours to be
    modified for the duration of a code block.

    In addition to controlling the "combine" operations during loading, LoadPolicy also
    provides the ``support_multiple_references`` option, to manage the detection and
    handling of cases where a hybrid coordinate has multiple reference fields : for
    example, a UM file which contains a series of fields describing a time-varying
    orography.

    The ``support_multiple_references`` option takes the value ``True`` or ``False`` to
    enable or disable this.  The default is ``True`` for all standard settings except
    "legacy".

    .. note ::

        The default behaviour will "fix" loading for cases like the time-varying
        orography case described above.  However, this is not strictly
        backwards-compatible.  If this causes problems, you can force identical loading
        behaviour to earlier Iris versions with ``LOAD_POLICY.set("legacy")`` or
        equivalent.

    .. testsetup::

        from iris import LOAD_POLICY
        loadpolicy_old_settings = LOAD_POLICY.settings()

    .. testcleanup::

        # restore original settings, so as not to upset other tests
        LOAD_POLICY.set(loadpolicy_old_settings)

    Examples
    --------
    >>> LOAD_POLICY.set("legacy")
    >>> print(LOAD_POLICY)
    LoadPolicy(equalise_cubes_kwargs=None, merge_concat_sequence='m', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=False)
    >>> LOAD_POLICY.support_multiple_references = True
    >>> print(LOAD_POLICY)
    LoadPolicy(equalise_cubes_kwargs=None, merge_concat_sequence='m', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=True)
    >>> LOAD_POLICY.set(merge_concat_sequence="cm")
    >>> print(LOAD_POLICY)
    LoadPolicy(equalise_cubes_kwargs=None, merge_concat_sequence='cm', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=True)
    >>> with LOAD_POLICY.context("comprehensive"):
    ...    print(LOAD_POLICY)
    LoadPolicy(equalise_cubes_kwargs={'apply_all': True}, merge_concat_sequence='mc', merge_unique=False, repeat_until_unchanged=True, support_multiple_references=True)
    >>> print(LOAD_POLICY)
    LoadPolicy(equalise_cubes_kwargs=None, merge_concat_sequence='cm', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=True)
    """

    # Option keys are as for CombineOptions, plus the multiple-refs control
    OPTION_KEYS = CombineOptions.OPTION_KEYS + ["support_multiple_references"]

    # Allowed values are as for CombineOptions, plus boolean values for multiple-refs
    _OPTIONS_ALLOWED_VALUES = dict(
        list(CombineOptions._OPTIONS_ALLOWED_VALUES.items())
        + [("support_multiple_references", (True, False))]
    )

    # Settings dicts are as for CombineOptions, but with a multiple-load value added
    SETTINGS = {
        key: dict(
            list(settings.items()) + [("support_multiple_references", key != "legacy")]
        )
        for key, settings in CombineOptions.SETTINGS.items()
    }

    @contextlib.contextmanager
    def context(self, settings: str | dict | None = None, **kwargs):
        """Return a context manager applying given options changes during a scope.

        Parameters
        ----------
        settings : str or dict, optional
            A settings name or options dictionary, as for :meth:`~LoadPolicy.set`.
        kwargs : dict
            Option values, as for :meth:`~LoadPolicy.set`.

        Examples
        --------
        .. testsetup::

            import iris
            from iris import LOAD_POLICY, sample_data_path

        >>> # Show how a CombineOptions acts in the context of a load operation
        >>> path = sample_data_path("time_varying_hybrid_height", "*.pp")
        >>>
        >>> # Show that "legacy" load behaviour allows merge but not concatenate
        >>> with LOAD_POLICY.context("legacy"):
        ...     cubes = iris.load(path, "x_wind")
        >>> print(cubes)
        0: x_wind / (m s-1)                    (time: 2; model_level_number: 5; latitude: 144; longitude: 192)
        1: x_wind / (m s-1)                    (time: 12; model_level_number: 5; latitude: 144; longitude: 192)
        2: x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)
        >>>
        >>> # Show how "recommended" behaviour enables concatenation also
        >>> with LOAD_POLICY.context("recommended"):
        ...     cubes = iris.load(path, "x_wind")
        >>> print(cubes)
        0: x_wind / (m s-1)                    (model_level_number: 5; time: 15; latitude: 144; longitude: 192)
        """
        # Save the current state
        saved_settings = self.settings()

        # Apply the new options and execute the context
        try:
            self.set(settings, **kwargs)
            yield
        finally:
            # Re-establish the former state
            self.set(saved_settings)


#: A control object containing the current file loading strategy options.
LOAD_POLICY = LoadPolicy()

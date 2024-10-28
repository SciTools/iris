# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Loading mechanism and functions."""

import contextlib
import itertools
import threading
from typing import Iterable, Mapping

import iris
import iris.exceptions


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
        return _CubeFilter(
            self.constraint,
            _combine_load_cubes(self.cubes, merge_require_unique=unique),
        )


class _CubeFilterCollection:
    """A list of _CubeFilter instances."""

    @staticmethod
    def from_cubes(cubes, constraints=None):
        """Create a new collection from an iterable of cubes, and some optional constraints."""
        constraints = iris._constraints.list_of_constraints(constraints)
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


def _generate_cubes(uris, callback, constraints):
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


def _load_collection(uris, constraints=None, callback=None):
    from iris.fileformats.rules import _MULTIREF_DETECTION

    try:
        # This routine is called once per iris load operation.
        # Control of the "multiple refs" handling is implicit in this routine
        # NOTE: detection of multiple reference fields, and it's enabling of post-load
        # concatenation, is triggered **per-load, not per-cube**
        # This behaves unexpectefly for "iris.load_cubes" : a post-concatenation is
        # triggered for all cubes or none, not per-cube (i.e. per constraint).
        _MULTIREF_DETECTION.found_multiple_refs = False

        cubes = _generate_cubes(uris, callback, constraints)
        result = _CubeFilterCollection.from_cubes(cubes, constraints)
    except EOFError as e:
        raise iris.exceptions.TranslationError(
            "The file appears empty or incomplete: {!r}".format(str(e))
        )
    return result


class LoadPolicy(threading.local):
    """A container for loading strategy options.

    Controls merge/concatenate usage during loading.

    Also controls the detection and handling of cases where a hybrid coordinate
    uses multiple reference fields : for example, a UM file which contains a series of
    fields describing time-varying orography.

    Options can be set directly, or via :meth:`~iris.LoadPolicy.set`, or changed for
    the scope of a code block with :meth:`~iris.LoadPolicy.context`.

    .. note ::

        The default behaviour will "fix" loading for cases like the one just described.
        However this is not strictly backwards-compatible.  If this causes problems,
        you can force identical loading behaviour to earlier Iris versions with
        ``LOAD_POLICY.set("legacy")`` or equivalent.

    .. testsetup::

        from iris import LOAD_POLICY

    Notes
    -----
    The individual configurable options are :

    * ``support_multiple_references`` = True / False
        When enabled, the presence of multiple aux-factory reference cubes, which merge
        to define a extra dimension, will add that dimension to the loaded cubes.
        This is essential for correct support of time-dependent hybrid coordinates (i.e.
        aux factories) when loading from fields-based data (e.g. PP or GRIB).
        For example (notably) time-dependent orography in UM data on hybrid-heights.

        In addition, when such multiple references are detected, an extra concatenate
        step is added to the 'merge_concat_sequence' (see below), if none is already
        configured there.

    * ``merge_concat_sequence`` = "m" / "c" / "cm" / "mc"
        Specifies whether to merge, or concatenate, or both in either order.
        This is the "combine" operation which is applied to loaded data.

    * ``repeat_until_unchanged`` = True / False
        When enabled, the configured "combine" operation will be repeated until the
        result is stable (no more cubes are combined).

    Several common sets of options are provided in :data:`~iris.LOAD_POLICY.SETTINGS` :

    *  ``"legacy"``
        Produces results identical to Iris versions < 3.11, i.e. before the varying
        hybrid references were supported.

    * ``"default"``
        As "legacy" except that ``support_multiple_references=True``.  This differs
        from "legacy" only when multiple mergeable reference fields are encountered,
        in which case incoming cubes are extended into the extra dimension, and a
        concatenate step is added.

    * ``"recommended"``
        Enables multiple reference handling, and applies a merge step followed by
        a concatenate step.

    * ``"comprehensive"``
        Like "recommended", but will also *repeat* the merge+concatenate steps until no
        further change is produced.

        .. note ::

            The 'comprehensive' policy makes a maximum effort to reduce the number of
            cubes to a minimum.  However, it still cannot combine cubes with a mixture
            of matching dimension and scalar coordinates.  This may be supported at
            some later date, but for now is not possible without specific user actions.

    .. Note ::

        See also : :ref:`controlling_merge`.

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

    # Useful constants
    OPTION_KEYS = (
        "support_multiple_references",
        "merge_concat_sequence",
        "repeat_until_unchanged",
    )
    _OPTIONS_ALLOWED_VALUES = {
        "support_multiple_references": (False, True),
        "merge_concat_sequence": ("", "m", "c", "mc", "cm"),
        "repeat_until_unchanged": (False, True),
    }
    SETTINGS = {
        "legacy": dict(
            support_multiple_references=False,
            merge_concat_sequence="m",
            repeat_until_unchanged=False,
        ),
        "default": dict(
            support_multiple_references=True,
            merge_concat_sequence="m",
            repeat_until_unchanged=False,
        ),
        "recommended": dict(
            support_multiple_references=True,
            merge_concat_sequence="mc",
            repeat_until_unchanged=False,
        ),
        "comprehensive": dict(
            support_multiple_references=True,
            merge_concat_sequence="mc",
            repeat_until_unchanged=True,
        ),
    }

    def __init__(self, options: str | dict | None = None, **kwargs):
        """Create loading strategy control object."""
        self.set("default")
        self.set(options, **kwargs)

    def __setattr__(self, key, value):
        if key not in self.OPTION_KEYS:
            raise KeyError(f"LoadPolicy object has no property '{key}'.")

        allowed_values = self._OPTIONS_ALLOWED_VALUES[key]
        if value not in allowed_values:
            msg = (
                f"{value!r} is not a valid setting for LoadPolicy.{key} : "
                f"must be one of '{allowed_values}'."
            )
            raise ValueError(msg)

        self.__dict__[key] = value

    def set(self, options: str | dict | None = None, **kwargs):
        """Set new options.

        Parameters
        ----------
        * options : str or dict, optional
            A dictionary of options values, or the name of one of the
            :data:`~iris.LoadPolicy.SETTINGS` standard option sets,
            e.g. "legacy" or "comprehensive".
        * kwargs : dict
            Individual option settings, from :data:`~iris.LoadPolicy.OPTION_KEYS`.

        Note
        ----
        Keyword arguments are applied after the 'options' arg, and
        so will take precedence.

        """
        if options is None:
            options = {}
        elif isinstance(options, str) and options in self.SETTINGS:
            options = self.SETTINGS[options]
        elif not isinstance(options, Mapping):
            msg = (
                f"Invalid arg options={options!r} : "
                f"must be a dict, or one of {tuple(self.SETTINGS.keys())}"
            )
            raise TypeError(msg)

        # Override any options with keywords
        options.update(**kwargs)
        bad_keys = [key for key in options if key not in self.OPTION_KEYS]
        if bad_keys:
            msg = f"Unknown options {bad_keys} : valid options are {self.OPTION_KEYS}."
            raise ValueError(msg)

        # Implement all options by changing own content.
        for key, value in options.items():
            setattr(self, key, value)

    def settings(self):
        """Return an options dict containing the current settings."""
        return {key: getattr(self, key) for key in self.OPTION_KEYS}

    def __repr__(self):
        msg = f"{self.__class__.__name__}("
        msg += ", ".join(f"{key}={getattr(self, key)!r}" for key in self.OPTION_KEYS)
        msg += ")"
        return msg

    @contextlib.contextmanager
    def context(self, settings=None, **kwargs):
        """Return a context manager applying given options.

        Parameters
        ----------
        settings : str or dict
            Options dictionary or name, as for :meth:`~LoadPolicy.set`.
        kwargs : dict
            Option values, as for :meth:`~LoadPolicy.set`.

        Examples
        --------
        .. testsetup::

            import iris
            from iris import LOAD_POLICY, sample_data_path

        >>> path = sample_data_path("time_varying_hybrid_height", "*.pp")
        >>> with LOAD_POLICY.context("legacy"):
        ...     cubes = iris.load(path, "x_wind")
        >>> print(cubes)
        0: x_wind / (m s-1)                    (time: 2; model_level_number: 5; latitude: 144; longitude: 192)
        1: x_wind / (m s-1)                    (time: 12; model_level_number: 5; latitude: 144; longitude: 192)
        2: x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)
        >>>
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


#: A control object containing the current file loading options.
LOAD_POLICY = LoadPolicy()


def _combine_cubes(cubes, options, merge_require_unique):
    """Combine cubes as for load, according to "loading policy" options.

    Applies :meth:`~iris.cube.CubeList.merge`/:meth:`~iris.cube.CubeList.concatenate`
    steps to the given cubes, as determined by the 'settings'.

    Parameters
    ----------
    cubes : list of :class:`~iris.cube.Cube`
        A list of cubes to combine.
    options : dict
        Settings, as described for :meth:`iris.LOAD_POLICY.set`.
        Defaults to current :meth:`iris.LOAD_POLICY.settings`.
    merge_require_unique : bool
        Value for the 'unique' keyword in any merge operations.

    Returns
    -------
    :class:`~iris.cube.CubeList`

    .. Note::
        The ``support_multiple_references`` keyword/property has no effect on the
        :func:`_combine_cubes` operation : it only takes effect during a load operation.

    Notes
    -----
    TODO: make this public API in future.
    At that point, change the API to support (options=None, **kwargs) + add testing of
    those modes (notably arg type = None / str / dict).

    """
    from iris.cube import CubeList

    if not isinstance(cubes, CubeList):
        cubes = CubeList(cubes)

    while True:
        n_original_cubes = len(cubes)
        sequence = options["merge_concat_sequence"]

        if sequence[0] == "c":
            # concat if it comes first
            cubes = cubes.concatenate()
        if "m" in sequence:
            # merge if requested
            cubes = cubes.merge(unique=merge_require_unique)
        if sequence[-1] == "c":
            # concat if it comes last
            cubes = cubes.concatenate()

        # Repeat if requested, *and* this step reduced the number of cubes
        if not options["repeat_until_unchanged"] or len(cubes) >= n_original_cubes:
            break

    return cubes


def _combine_load_cubes(cubes, merge_require_unique=False):
    # A special version to call _combine_cubes while also implementing the
    # _MULTIREF_DETECTION behaviour
    options = LOAD_POLICY.settings()
    if (
        options["support_multiple_references"]
        and "c" not in options["merge_concat_sequence"]
    ):
        # Add a concatenate to implement the "multiref triggers concatenate" mechanism
        from iris.fileformats.rules import _MULTIREF_DETECTION

        if _MULTIREF_DETECTION.found_multiple_refs:
            options["merge_concat_sequence"] += "c"

    return _combine_cubes(cubes, options, merge_require_unique=merge_require_unique)


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

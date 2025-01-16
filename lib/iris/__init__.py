# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A package for handling multi-dimensional data and associated metadata.

.. note ::

    The Iris documentation has further usage information, including
    a :ref:`user guide <user_guide_index>` which should be the first port of
    call for new users.

The functions in this module provide the main way to load and/or save
your data.

The :func:`load` function provides a simple way to explore data from
the interactive Python prompt. It will convert the source data into
:class:`Cubes <iris.cube.Cube>`, and combine those cubes into
higher-dimensional cubes where possible.

The :func:`load_cube` and :func:`load_cubes` functions are similar to
:func:`load`, but they raise an exception if the number of cubes is not
what was expected. They are more useful in scripts, where they can
provide an early sanity check on incoming data.

The :func:`load_raw` function is provided for those occasions where the
automatic combination of cubes into higher-dimensional cubes is
undesirable. However, it is intended as a tool of last resort! If you
experience a problem with the automatic combination process then please
raise an issue with the Iris developers.

To persist a cube to the file-system, use the :func:`save` function.

All the load functions share very similar arguments:

* uris:
    Either a single filename/URI expressed as a string or
    :class:`pathlib.PurePath`, or an iterable of filenames/URIs.

    Filenames can contain `~` or `~user` abbreviations, and/or
    Unix shell-style wildcards (e.g. `*` and `?`). See the
    standard library function :func:`os.path.expanduser` and
    module :mod:`fnmatch` for more details.

    .. warning::

        If supplying a URL, only OPeNDAP Data Sources are supported.

* constraints:
    Either a single constraint, or an iterable of constraints.
    Each constraint can be either a string, an instance of
    :class:`iris.Constraint`, or an instance of
    :class:`iris.AttributeConstraint`.  If the constraint is a string
    it will be used to match against cube.name().

    .. _constraint_egs:

    For example::

        # Load air temperature data.
        load_cube(uri, 'air_temperature')

        # Load data with a specific model level number.
        load_cube(uri, iris.Constraint(model_level_number=1))

        # Load data with a specific STASH code.
        load_cube(uri, iris.AttributeConstraint(STASH='m01s00i004'))

* callback:
    A function to add metadata from the originating field and/or URI which
    obeys the following rules:

    1. Function signature must be: ``(cube, field, filename)``.
    2. Modifies the given cube inplace, unless a new cube is
       returned by the function.
    3. If the cube is to be rejected the callback must raise
       an :class:`iris.exceptions.IgnoreCubeException`.

    For example::

        def callback(cube, field, filename):
            # Extract ID from filenames given as: <prefix>__<exp_id>
            experiment_id = filename.split('__')[1]
            experiment_coord = iris.coords.AuxCoord(
                experiment_id, long_name='experiment_id')
            cube.add_aux_coord(experiment_coord)

"""

from collections.abc import Iterable
import contextlib
import glob
import importlib
import itertools
import os.path
import threading
from typing import Callable, Literal, Mapping

import iris._constraints
import iris.config
import iris.io

from ._deprecation import IrisDeprecation, warn_deprecated

try:
    from ._version import version as __version__  # noqa: F401
except ModuleNotFoundError:
    __version__ = "unknown"


try:
    import iris_sample_data
except ImportError:
    iris_sample_data = None


# Restrict the names imported when using "from iris import *"
__all__ = [
    "AttributeConstraint",
    "Constraint",
    "FUTURE",
    "Future",
    "IrisDeprecation",
    "LOAD_POLICY",
    "LoadPolicy",
    "NameConstraint",
    "load",
    "load_cube",
    "load_cubes",
    "load_raw",
    "sample_data_path",
    "save",
    "site_configuration",
    "use_plugin",
]


Constraint = iris._constraints.Constraint
AttributeConstraint = iris._constraints.AttributeConstraint
NameConstraint = iris._constraints.NameConstraint


class Future(threading.local):
    """Run-time configuration controller."""

    def __init__(
        self,
        datum_support=False,
        pandas_ndim=False,
        save_split_attrs=False,
        date_microseconds=False,
    ):
        """Container for run-time options controls.

        To adjust the values simply update the relevant attribute from
        within your code. For example::

            # example_future_flag is a fictional example.
            iris.FUTURE.example_future_flag = False

        If Iris code is executed with multiple threads, note the values of
        these options are thread-specific.

        Parameters
        ----------
        datum_support : bool, default=False
            Opts in to loading coordinate system datum information from NetCDF
            files into :class:`~iris.coord_systems.CoordSystem`, wherever
            this information is present.
        pandas_ndim : bool, default=False
            See :func:`iris.pandas.as_data_frame` for details - opts in to the
            newer n-dimensional behaviour.
        save_split_attrs : bool, default=False
            Save "global" and "local" cube attributes to netcdf in appropriately
            different ways :  "global" ones are saved as dataset attributes, where
            possible, while "local" ones are saved as data-variable attributes.
            See :func:`iris.fileformats.netcdf.saver.save`.
        date_microseconds : bool, default=False
            Newer versions of cftime and cf-units support microsecond precision
            for dates, compared to the legacy behaviour that only works with
            seconds. Enabling microsecond precision will alter core Iris
            behaviour, such as when using :class:`~iris.Constraint`, and you
            may need to defend against floating point precision issues where
            you didn't need to before.

        """
        # The flag 'example_future_flag' is provided as a reference for the
        # structure of this class.
        #
        # Note that self.__dict__ is used explicitly due to the manner in which
        # __setattr__ is overridden.
        #
        # self.__dict__['example_future_flag'] = example_future_flag
        self.__dict__["datum_support"] = datum_support
        self.__dict__["pandas_ndim"] = pandas_ndim
        self.__dict__["save_split_attrs"] = save_split_attrs
        self.__dict__["date_microseconds"] = date_microseconds

        # TODO: next major release: set IrisDeprecation to subclass
        #  DeprecationWarning instead of UserWarning.

    def __repr__(self):
        # msg = ('Future(example_future_flag={})')
        # return msg.format(self.example_future_flag)
        msg = "Future(datum_support={}, pandas_ndim={}, save_split_attrs={})"
        return msg.format(
            self.datum_support,
            self.pandas_ndim,
            self.save_split_attrs,
            self.date_microseconds,
        )

    # deprecated_options = {'example_future_flag': 'warning',}
    deprecated_options: dict[str, Literal["error", "warning"]] = {}

    def __setattr__(self, name, value):
        if name in self.deprecated_options:
            level = self.deprecated_options[name]
            if level == "error" and not value:
                emsg = (
                    "setting the 'Future' property {prop!r} has been "
                    "deprecated to be removed in a future release, and "
                    "deprecated {prop!r} behaviour has been removed. "
                    "Please remove code that sets this property."
                )
                raise AttributeError(emsg.format(prop=name))
            else:
                msg = (
                    "setting the 'Future' property {!r} is deprecated "
                    "and will be removed in a future release. "
                    "Please remove code that sets this property."
                )
                warn_deprecated(msg.format(name))
        if name not in self.__dict__:
            msg = "'Future' object has no attribute {!r}".format(name)
            raise AttributeError(msg)
        self.__dict__[name] = value

    @contextlib.contextmanager
    def context(self, **kwargs):
        """Return context manager for temp modification of option values for the active thread.

        On entry to the `with` statement, all keyword arguments are
        applied to the Future object. On exit from the `with`
        statement, the previous state is restored.

        For example::

            # example_future_flag is a fictional example.
            with iris.FUTURE.context(example_future_flag=False):
                # ... code that expects some past behaviour

        """
        # Save the current context
        current_state = self.__dict__.copy()
        # Update the state
        for name, value in kwargs.items():
            setattr(self, name, value)
        try:
            yield
        finally:
            # Return the state
            self.__dict__.clear()
            self.__dict__.update(current_state)


#: Object containing all the Iris run-time options.
FUTURE = Future()


# Initialise the site configuration dictionary.
#: Iris site configuration dictionary.
site_configuration: dict[
    Literal["cf_profile", "cf_patch", "cf_patch_conventions"],
    Callable | Literal[False] | None,
] = {}

try:
    from iris.site_config import update as _update
except ImportError:
    pass
else:
    _update(site_configuration)


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
    from iris.cube import _CubeFilterCollection
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


save = iris.io.save


def sample_data_path(*path_to_join):
    """Given the sample data resource, returns the full path to the file.

    .. note::

        This function is only for locating files in the iris sample data
        collection (installed separately from iris). It is not needed or
        appropriate for general file access.

    """
    target = os.path.join(*path_to_join)
    if os.path.isabs(target):
        raise ValueError(
            "Absolute paths, such as {!r}, are not supported.\n"
            "NB. This function is only for locating files in the "
            "iris sample data collection. It is not needed or "
            "appropriate for general file access.".format(target)
        )
    if iris_sample_data is not None:
        target = os.path.join(iris_sample_data.path, target)
    else:
        raise ImportError(
            "Please install the 'iris-sample-data' package to access sample data."
        )
    if not glob.glob(target):
        raise ValueError(
            "Sample data file(s) at {!r} not found.\n"
            "NB. This function is only for locating files in the "
            "iris sample data collection. It is not needed or "
            "appropriate for general file access.".format(target)
        )
    return target


def use_plugin(plugin_name):
    """Import a plugin.

    Parameters
    ----------
    plugin_name : str
        Name of plugin.

    Examples
    --------
    The following::

        use_plugin("my_plugin")

    is equivalent to::

        import iris.plugins.my_plugin

    This is useful for plugins that are not used directly, but instead do all
    their setup on import.  In this case, style checkers would not know the
    significance of the import statement and warn that it is an unused import.
    """
    importlib.import_module(f"iris.plugins.{plugin_name}")

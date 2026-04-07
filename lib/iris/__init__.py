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

.. note::

    User control of the 'combine' process is provided via a specific
    :class:`iris.CombineOptions` object called :data:`iris.COMBINE_POLICY`.
    See the :class:`iris.CombineOptions` class for details.

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

import contextlib
import glob
import importlib
import os.path
import threading
from typing import Callable, Literal

from iris._combine import COMBINE_POLICY as _COMBINE_POLICY
from iris._combine import CombineOptions
import iris._constraints
import iris.config
import iris.io
from iris.io import save
from iris.loading import (
    load,
    load_cube,
    load_cubes,
    load_raw,
)

# NOTE: we make an independent local 'LOAD_POLICY' definition here, just so that we
# can ensure an entry for it in our API documentation page.

#: An object to control default cube combination and loading options
COMBINE_POLICY = _COMBINE_POLICY

#: An alias for the :class:`~iris._combine.CombineOptions` class.
LoadPolicy = CombineOptions

#: An alias for the :data:`~iris.COMBINE_POLICY` object.
LOAD_POLICY = _COMBINE_POLICY


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
    "COMBINE_POLICY",
    "CombineOptions",
    "Constraint",
    "DATALESS",
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

#: To be used when copying a cube to make the new cube dataless.
DATALESS = "NONE"


class Future(threading.local):
    """Run-time configuration controller."""

    def __init__(
        self,
        datum_support=False,
        pandas_ndim=False,
        save_split_attrs=False,
        date_microseconds=False,
        derived_bounds=False,
        lam_pole_offset=False,
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
        derived_bounds : bool, default=False
            When ``True``, uses the correct CF rules for bounds of derived coordinates
            for both loading and saving NetCDF.  This requires that these must be linked
            via a separate "formula_terms" attribute on the bounds variable.
            If ``False``, bounds are only linked with a "bounds" attribute, though this
            is strictly incorrect for CF >= v1.7.
            See `here in CF <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#cell-boundaries>`_.
        lam_pole_offset : bool, default=False
            When True, saving a cube on a "Limited Area Model" (LAM) domain
            to a PP file will set the pole longitude (PP field ``bplon``) to
            180.0 degrees if the grid is defined on a standard pole. Does not
            affect global or rotated-pole domains.

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
        self.__dict__["derived_bounds"] = derived_bounds
        self.__dict__["lam_pole_offset"] = lam_pole_offset

        # TODO: next major release: set IrisDeprecation to subclass
        #  DeprecationWarning instead of UserWarning.

    def __repr__(self):
        content = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        msg = f"Future({content})"
        return msg

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

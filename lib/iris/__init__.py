# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A package for handling multi-dimensional data and associated metadata.

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
        Either a single filename/URI expressed as a string, or an
        iterable of filenames/URIs.

        Filenames can contain `~` or `~user` abbreviations, and/or
        Unix shell-style wildcards (e.g. `*` and `?`). See the
        standard library function :func:`os.path.expanduser` and
        module :mod:`fnmatch` for more details.

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
import itertools
import os.path
import threading

import iris._constraints
from iris._deprecation import IrisDeprecation, warn_deprecated
import iris.config
import iris.io

try:
    import iris_sample_data
except ImportError:
    iris_sample_data = None


# Iris revision.
__version__ = "3.2.dev0"

# Restrict the names imported when using "from iris import *"
__all__ = [
    "AttributeConstraint",
    "Constraint",
    "FUTURE",
    "Future",
    "IrisDeprecation",
    "NameConstraint",
    "load",
    "load_cube",
    "load_cubes",
    "load_raw",
    "sample_data_path",
    "save",
    "site_configuration",
]


Constraint = iris._constraints.Constraint
AttributeConstraint = iris._constraints.AttributeConstraint
NameConstraint = iris._constraints.NameConstraint


class Future(threading.local):
    """Run-time configuration controller."""

    def __init__(self):
        """
        A container for run-time options controls.

        To adjust the values simply update the relevant attribute from
        within your code. For example::

            iris.FUTURE.example_future_flag = False

        If Iris code is executed with multiple threads, note the values of
        these options are thread-specific.

        .. note::

            iris.FUTURE.example_future_flag does not exist. It is provided
            as an example because there are currently no flags in
            iris.Future.

        """
        # The flag 'example_future_flag' is provided as a future reference
        # for the structure of this class.
        #
        # self.__dict__['example_future_flag'] = example_future_flag
        pass

    def __repr__(self):

        # msg = ('Future(example_future_flag={})')
        # return msg.format(self.example_future_flag)
        msg = "Future()"
        return msg.format()

    # deprecated_options = {'example_future_flag': 'warning',}
    deprecated_options = {}

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
        """
        Return a context manager which allows temporary modification of
        the option values for the active thread.

        On entry to the `with` statement, all keyword arguments are
        applied to the Future object. On exit from the `with`
        statement, the previous state is restored.

        For example::
            with iris.FUTURE.context(example_future_flag=False):
                # ... code that expects some past behaviour

        .. note::

            iris.FUTURE.example_future_flag does not exist and is
            provided only as an example since there are currently no
            flags in Future.

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
site_configuration = {}

try:
    from iris.site_config import update as _update
except ImportError:
    pass
else:
    _update(site_configuration)


def _generate_cubes(uris, callback, constraints):
    """Returns a generator of cubes given the URIs and a callback."""
    if isinstance(uris, str):
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
        else:
            raise ValueError("Iris cannot handle the URI scheme: %s" % scheme)


def _load_collection(uris, constraints=None, callback=None):
    from iris.cube import _CubeFilterCollection

    try:
        cubes = _generate_cubes(uris, callback, constraints)
        result = _CubeFilterCollection.from_cubes(cubes, constraints)
    except EOFError as e:
        raise iris.exceptions.TranslationError(
            "The file appears empty or incomplete: {!r}".format(str(e))
        )
    return result


def load(uris, constraints=None, callback=None):
    """
    Loads any number of Cubes for each constraint.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Args:

    * uris:
        One or more filenames/URIs.

    Kwargs:

    * constraints:
        One or more constraints.
    * callback:
        A modifier/filter function.

    Returns:
        An :class:`iris.cube.CubeList`. Note that there is no inherent order
        to this :class:`iris.cube.CubeList` and it should be treated as if it
        were random.

    """
    return _load_collection(uris, constraints, callback).merged().cubes()


def load_cube(uris, constraint=None, callback=None):
    """
    Loads a single cube.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Args:

    * uris:
        One or more filenames/URIs.

    Kwargs:

    * constraints:
        A constraint.
    * callback:
        A modifier/filter function.

    Returns:
        An :class:`iris.cube.Cube`.

    """
    constraints = iris._constraints.list_of_constraints(constraint)
    if len(constraints) != 1:
        raise ValueError("only a single constraint is allowed")

    cubes = _load_collection(uris, constraints, callback).cubes()

    try:
        cube = cubes.merge_cube()
    except iris.exceptions.MergeError as e:
        raise iris.exceptions.ConstraintMismatchError(str(e))
    except ValueError:
        raise iris.exceptions.ConstraintMismatchError("no cubes found")

    return cube


def load_cubes(uris, constraints=None, callback=None):
    """
    Loads exactly one Cube for each constraint.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Args:

    * uris:
        One or more filenames/URIs.

    Kwargs:

    * constraints:
        One or more constraints.
    * callback:
        A modifier/filter function.

    Returns:
        An :class:`iris.cube.CubeList`. Note that there is no inherent order
        to this :class:`iris.cube.CubeList` and it should be treated as if it
        were random.

    """
    # Merge the incoming cubes
    collection = _load_collection(uris, constraints, callback).merged()

    # Make sure we have exactly one merged cube per constraint
    bad_pairs = [pair for pair in collection.pairs if len(pair) != 1]
    if bad_pairs:
        fmt = "   {} -> {} cubes"
        bits = [fmt.format(pair.constraint, len(pair)) for pair in bad_pairs]
        msg = "\n" + "\n".join(bits)
        raise iris.exceptions.ConstraintMismatchError(msg)

    return collection.cubes()


def load_raw(uris, constraints=None, callback=None):
    """
    Loads non-merged cubes.

    This function is provided for those occasions where the automatic
    combination of cubes into higher-dimensional cubes is undesirable.
    However, it is intended as a tool of last resort! If you experience
    a problem with the automatic combination process then please raise
    an issue with the Iris developers.

    For a full description of the arguments, please see the module
    documentation for :mod:`iris`.

    Args:

    * uris:
        One or more filenames/URIs.

    Kwargs:

    * constraints:
        One or more constraints.
    * callback:
        A modifier/filter function.

    Returns:
        An :class:`iris.cube.CubeList`.

    """
    from iris.fileformats.um._fast_load import _raw_structured_loading

    with _raw_structured_loading():
        return _load_collection(uris, constraints, callback).cubes()


save = iris.io.save


def sample_data_path(*path_to_join):
    """
    Given the sample data resource, returns the full path to the file.

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
            "Please install the 'iris-sample-data' package to "
            "access sample data."
        )
    if not glob.glob(target):
        raise ValueError(
            "Sample data file(s) at {!r} not found.\n"
            "NB. This function is only for locating files in the "
            "iris sample data collection. It is not needed or "
            "appropriate for general file access.".format(target)
        )
    return target

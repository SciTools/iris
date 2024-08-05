# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides access to Iris-specific configuration values.

The default configuration values can be overridden by creating the file
``iris/etc/site.cfg``. If it exists, this file must conform to the format
defined by :mod:`configparser`.

"""

import configparser
import contextlib
import logging
import os.path
import warnings

import iris.warnings


def get_logger(name, datefmt=None, fmt=None, level=None, propagate=None, handler=True):
    """Create a custom class for logging.

    Create a :class:`logging.Logger` with a :class:`logging.StreamHandler`
    and custom :class:`logging.Formatter`.

    Parameters
    ----------
    name :
        The name of the logger. Typically this is the module filename that
        owns the logger.
    datefmt : optional
        The date format string of the :class:`logging.Formatter`.
        Defaults to ``%d-%m-%Y %H:%M:%S``.
    fmt : optional
        The additional format string of the :class:`logging.Formatter`.
        This is appended to the default format string
        ``%(asctime)s %(name)s %(levelname)s - %(message)s``.
    level : optional
        The threshold level of the logger. Defaults to ``INFO``.
    propagate : optional
        Sets the ``propagate`` attribute of the :class:`logging.Logger`,
        which determines whether events logged to this logger will be
        passed to the handlers of higher level loggers. Defaults to
        ``False``.
    handler : bool, default=True
        Create and attach a :class:`logging.StreamHandler` to the
        logger. Defaults to ``True``.

    Returns
    -------
    :class:`logging.Logger`.

    """
    if level is None:
        # Default logging level.
        level = "INFO"

    if propagate is None:
        # Default logging propagate behaviour.
        propagate = False

    # Create the named logger.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Create and add the handler to the logger, if required.
    if handler:
        if datefmt is None:
            # Default date format string.
            datefmt = "%d-%m-%Y %H:%M:%S"

        # Default format string.
        _fmt = "%(asctime)s %(name)s %(levelname)s - %(message)s"
        # Append additional format string, if appropriate.
        fmt = _fmt if fmt is None else f"{_fmt} {fmt}"

        # Create a formatter.
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        # Create a logging handler.
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def get_option(section, option, default=None):
    """Return the option value for the given section.

    Returns the option value for the given section, or the default value
    if the section/option is not present.

    """
    value = default
    if config.has_option(section, option):
        value = config.get(section, option)
    return value


def get_dir_option(section, option, default=None):
    """Return the directory path from the given option and section.

    Returns the directory path from the given option and section, or
    returns the given default value if the section/option is not present
    or does not represent a valid directory.

    """
    path = default
    if config.has_option(section, option):
        c_path = config.get(section, option)
        if os.path.isdir(c_path):
            path = c_path
        else:
            msg = (
                "Ignoring config item {!r}:{!r} (section:option) as {!r}"
                " is not a valid directory path."
            )
            warnings.warn(
                msg.format(section, option, c_path),
                category=iris.warnings.IrisIgnoringWarning,
            )
    return path


def _get_test_data_dir():
    """Return the test data directory and overriding if approproiate."""
    override = os.environ.get("OVERRIDE_TEST_DATA_REPOSITORY")

    if override and os.path.isdir(os.path.expanduser(override)):
        test_data_dir = os.path.abspath(override)
    else:
        test_data_dir = get_dir_option(
            _RESOURCE_SECTION,
            "test_data_dir",
            default=os.path.join(os.path.dirname(__file__), "test_data"),
        )

    return test_data_dir


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
"""The full path to the "iris" package."""

CONFIG_PATH = os.path.join(ROOT_PATH, "etc")
"""The full path to the configuration directory of the active Iris instance."""

# Load the optional "site.cfg" file if it exists.
config = configparser.ConfigParser()
config.read([os.path.join(CONFIG_PATH, "site.cfg")])

##################
# Resource options
_RESOURCE_SECTION = "Resources"

TEST_DATA_DIR = _get_test_data_dir()
"""Local directory where test data exists.
   Defaults to "test_data" sub-directory of the Iris package install directory.
   The test data directory supports the subset of Iris unit tests that require data.
   Directory contents accessed via :func:`iris.tests.get_data_path`."""

PALETTE_PATH: str = get_dir_option(
    _RESOURCE_SECTION, "palette_path", os.path.join(CONFIG_PATH, "palette")
)
"""The full path to the Iris palette configuration directory."""


# Runtime options


class NetCDF:
    """Control Iris NetCDF options."""

    def __init__(self, conventions_override=None):
        """Set up NetCDF processing options for Iris.

        Parameters
        ----------
        conventions_override : bool, optional
            Define whether the CF Conventions version (e.g. `CF-1.6`) set when
            saving a cube to a NetCDF file should be defined by
            Iris (the default) or the cube being saved.  If `False`
            (the default), specifies that Iris should set the
            CF Conventions version when saving cubes as NetCDF files.
            If `True`, specifies that the cubes being saved to NetCDF should
            set the CF Conventions version for the saved NetCDF files.

        Examples
        --------
        * Specify, for the lifetime of the session, that we want all cubes
          written to NetCDF to define their own CF Conventions versions::

            iris.config.netcdf.conventions_override = True
            iris.save('my_cube', 'my_dataset.nc')
            iris.save('my_second_cube', 'my_second_dataset.nc')

        * Specify, with a context manager, that we want a cube written to
          NetCDF to define its own CF Conventions version::

            with iris.config.netcdf.context(conventions_override=True):
                iris.save('my_cube', 'my_dataset.nc')

        """
        # Define allowed `__dict__` keys first.
        self.__dict__["conventions_override"] = None

        # Now set specific values.
        setattr(self, "conventions_override", conventions_override)

    def __repr__(self):
        msg = "NetCDF options: {}."
        # Automatically populate with all currently accepted kwargs.
        options = ["{}={}".format(k, v) for k, v in self.__dict__.items()]
        joined = ", ".join(options)
        return msg.format(joined)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            # Can't add new names.
            msg = "Cannot set option {!r} for {} configuration."
            raise AttributeError(msg.format(name, self.__class__.__name__))
        if value is None:
            # Set an unset value to the name's default.
            value = self._defaults_dict[name]["default"]
        if self._defaults_dict[name]["options"] is not None:
            # Replace a bad value with a good one if there is a defined set of
            # specified good values. If there isn't, we can assume that
            # anything goes.
            if value not in self._defaults_dict[name]["options"]:
                good_value = self._defaults_dict[name]["default"]
                wmsg = (
                    "Attempting to set invalid value {!r} for "
                    "attribute {!r}. Defaulting to {!r}."
                )
                warnings.warn(
                    wmsg.format(value, name, good_value),
                    category=iris.warnings.IrisDefaultingWarning,
                )
                value = good_value
        self.__dict__[name] = value

    @property
    def _defaults_dict(self):
        # Set this as a property so that it isn't added to `self.__dict__`.
        return {
            "conventions_override": {
                "default": False,
                "options": [True, False],
            },
        }

    @contextlib.contextmanager
    def context(self, **kwargs):
        """Allow temporary modification of the options via a context manager.

        Accepted kwargs are the same as can be supplied to the Option.

        """
        # Snapshot the starting state for restoration at the end of the
        # contextmanager block.
        starting_state = self.__dict__.copy()
        # Update the state to reflect the requested changes.
        for name, value in kwargs.items():
            setattr(self, name, value)
        try:
            yield
        finally:
            # Return the state to the starting state.
            self.__dict__.clear()
            self.__dict__.update(starting_state)


netcdf = NetCDF()

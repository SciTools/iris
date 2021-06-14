# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Load, configure and register color map palettes and initialise
color map meta-data mappings.

"""

from functools import wraps
import os
import os.path
import re

import cf_units
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import numpy as np

import iris.config
import iris.cube

# Symmetric normalization function pivot points by SI unit.
PIVOT_BY_UNIT = {cf_units.Unit("K"): 273.15}

# Color map names by palette file metadata field value.
CMAP_BREWER = set()
_CMAP_BY_SCHEME = None
_CMAP_BY_KEYWORD = None
_CMAP_BY_STD_NAME = None

_MISSING_KWARG_CMAP = "missing kwarg cmap"
_MISSING_KWARG_NORM = "missing kwarg norm"


def is_brewer(cmap):
    """
    Determine whether the color map is a Cynthia Brewer color map.

    Args:

    * cmap:
        The color map instance.

    Returns:
        Boolean.

    """
    result = False
    if cmap is not None:
        result = cmap.name in CMAP_BREWER
    return result


def _default_cmap_norm(args, kwargs):
    """
    This function injects default cmap and norm behavour into the keyword
    arguments, based on the cube referenced within the positional arguments.
    """
    cube = None

    # Find the single cube reference within the positional arguments.
    for arg in args:
        if isinstance(arg, iris.cube.Cube):
            cube = arg
            break

    # Find the keyword arguments of interest.
    colors = kwargs.get("colors", None)
    # cmap = None to disable default behaviour.
    cmap = kwargs.get("cmap", _MISSING_KWARG_CMAP)
    # norm = None to disable default behaviour.
    norm = kwargs.get("norm", _MISSING_KWARG_NORM)

    # Note that "colors" and "cmap" keywords are mutually exclusive.
    if colors is None and cube is not None:
        std_name = cube.standard_name.lower() if cube.standard_name else ""

        # Perform default "cmap" keyword behaviour.
        if cmap == _MISSING_KWARG_CMAP:
            # Check for an exact match against standard name.
            cmaps = _CMAP_BY_STD_NAME.get(std_name, set())

            if len(cmaps) == 0:
                # Check for a fuzzy match against a keyword.
                for keyword in _CMAP_BY_KEYWORD.keys():
                    if keyword in std_name:
                        cmaps.update(_CMAP_BY_KEYWORD[keyword])

            # Add default color map to keyword arguments.
            if len(cmaps):
                cmap = sorted(cmaps, reverse=True)[0]
                kwargs["cmap"] = mpl_cm.get_cmap(cmap)

        # Perform default "norm" keyword behaviour.
        if norm == _MISSING_KWARG_NORM:
            if "anomaly" in std_name:
                # Determine the pivot point.
                pivot = PIVOT_BY_UNIT.get(cube.units, 0)
                norm = SymmetricNormalize(pivot)
                kwargs["norm"] = norm

    return args, kwargs


def cmap_norm(cube):
    """
    Determine the default :class:`matplotlib.colors.LinearSegmentedColormap`
    and :class:`iris.palette.SymmetricNormalize` instances associated with
    the cube.

    Args:

    * cube (:class:`iris.cube.Cube`):
        Source cube to generate default palette from.

    Returns:
        Tuple of :class:`matplotlib.colors.LinearSegmentedColormap` and
        :class:`iris.palette.SymmetricNormalize`

    """
    args, kwargs = _default_cmap_norm((cube,), {})
    return kwargs.get("cmap"), kwargs.get("norm")


def auto_palette(func):
    """
    Decorator wrapper function to control the default behaviour of the
    matplotlib cmap and norm keyword arguments.

    Args:

    * func (callable):
        Callable function to be wrapped by the decorator.

    Returns:
        Closure wrapper function.

    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        """
        Closure wrapper function to provide default keyword argument
        behaviour.

        """
        # Update the keyword arguments with defaults.
        args, kwargs = _default_cmap_norm(args, kwargs)
        # Call the wrapped function and return its result.
        return func(*args, **kwargs)

    # Return the closure wrapper function.
    return wrapper_func


class SymmetricNormalize(mpl_colors.Normalize):
    """
    Provides a symmetric normalization class around a given pivot point.
    """

    def __init__(self, pivot, *args, **kwargs):
        self.pivot = pivot
        self._vmin = None
        self._vmax = None
        mpl_colors.Normalize.__init__(self, *args, **kwargs)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.pivot)

    def _update(self, val, update_min=True, update_max=True):
        # Update both _vmin and _vmax from given value.
        val_diff = np.abs(val - self.pivot)
        vmin_diff = np.abs(self._vmin - self.pivot) if self._vmin else 0.0
        vmax_diff = np.abs(self._vmax - self.pivot) if self._vmax else 0.0
        diff = max(val_diff, vmin_diff, vmax_diff)

        if update_min:
            self._vmin = self.pivot - diff
        if update_max:
            self._vmax = self.pivot + diff

    @property
    def vmin(self):
        return getattr(self, "_vmin")

    @vmin.setter
    def vmin(self, val):
        if val is None:
            self._vmin = None
        elif self._vmax is None:
            # Don't set _vmax, it'll stop matplotlib from giving us one.
            self._update(val, update_max=False)
        else:
            # Set both _vmin and _vmax from value
            self._update(val)

    @property
    def vmax(self):
        return getattr(self, "_vmax")

    @vmax.setter
    def vmax(self, val):
        if val is None:
            self._vmax = None
        elif self._vmin is None:
            # Don't set _vmin, it'll stop matplotlib from giving us one.
            self._update(val, update_min=False)
        else:
            # Set both _vmin and _vmax from value
            self._update(val)


def _load_palette():
    """
    Load, configure and register color map palettes and initialise
    color map metadata mappings.

    """
    # Reference these module level namespace variables.
    global CMAP_BREWER, _CMAP_BY_SCHEME, _CMAP_BY_KEYWORD, _CMAP_BY_STD_NAME

    _CMAP_BY_SCHEME = {}
    _CMAP_BY_KEYWORD = {}
    _CMAP_BY_STD_NAME = {}

    filenames = []

    # Identify all .txt color map palette files.
    for root, dirs, files in os.walk(iris.config.PALETTE_PATH):
        # Prune any .svn directory from the tree walk.
        if ".svn" in dirs:
            del dirs[dirs.index(".svn")]
        # Identify any target .txt color map palette files.
        filenames.extend(
            [
                os.path.join(root, filename)
                for filename in files
                if os.path.splitext(filename)[1] == ".txt"
            ]
        )

    for filename in filenames:
        # Default color map name based on the file base-name (case-SENSITIVE).
        cmap_name = os.path.splitext(os.path.basename(filename))[0]
        cmap_scheme = None
        cmap_keywords = []
        cmap_std_names = []
        cmap_type = None

        # Perform default color map interpolation for quantization
        # levels per primary color.
        interpolate_flag = True

        # Read the file header.
        with open(filename) as file_handle:
            header = filter(
                lambda line: re.match(r"^\s*#.*:\s+.*$", line),
                file_handle.readlines(),
            )

        # Extract the file header metadata.
        for line in header:
            line = line.replace("#", "", 1).split(":")
            head = line[0].strip().lower()
            body = line[1].strip()

            if head == "name":
                # Case-SENSITIVE.
                cmap_name = "brewer_{}".format(body)

            if head == "scheme":
                # Case-insensitive.
                cmap_scheme = body.lower()

            if head == "keyword":
                # Case-insensitive.
                keywords = [part.strip().lower() for part in body.split(",")]
                cmap_keywords.extend(keywords)

            if head == "std_name":
                # Case-insensitive.
                std_names = [part.strip().lower() for part in body.split(",")]
                cmap_std_names.extend(std_names)

            if head == "interpolate":
                # Case-insensitive.
                interpolate_flag = body.lower() != "off"

            if head == "type":
                # Case-insensitive.
                cmap_type = body.lower()

        # Integrity check for meta-data 'type' field.
        assert cmap_type is not None, (
            'Missing meta-data "type" keyword for color map file, "%s"'
            % filename
        )
        assert (
            cmap_type == "rgb"
        ), 'Invalid type [%s] for color map file "%s"' % (cmap_type, filename)

        # Update the color map look-up dictionaries.
        CMAP_BREWER.add(cmap_name)

        if cmap_scheme is not None:
            scheme_group = _CMAP_BY_SCHEME.setdefault(cmap_scheme, set())
            scheme_group.add(cmap_name)

        for keyword in cmap_keywords:
            keyword_group = _CMAP_BY_KEYWORD.setdefault(keyword, set())
            keyword_group.add(cmap_name)

        for std_name in cmap_std_names:
            std_name_group = _CMAP_BY_STD_NAME.setdefault(std_name, set())
            std_name_group.add(cmap_name)

        # Load palette data and create the associated color map.
        cmap_data = np.loadtxt(filename)
        # Ensure to restrict the number of RGB quantization levels to
        # prevent color map interpolation.

        if interpolate_flag:
            # Perform default color map interpolation for quantization
            # levels per primary color.
            cmap = mpl_colors.LinearSegmentedColormap.from_list(
                cmap_name, cmap_data
            )
        else:
            # Restrict quantization levels per primary color (turn-off
            # interpolation).
            # Typically used for Brewer color maps.
            cmap = mpl_colors.LinearSegmentedColormap.from_list(
                cmap_name, cmap_data, N=len(cmap_data)
            )

        # Register the color map for use.
        mpl_cm.register_cmap(cmap=cmap)


# Ensure to load the color map palettes.
_load_palette()

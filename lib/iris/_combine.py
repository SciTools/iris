# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Generalised mechanism for combining cubes into larger ones.

Integrates merge and concatenate with the cube-equalisation options and the promotion of
hybrid reference dimensions on loading.

This is effectively a generalised "combine cubes" operation, but it is not (yet)
publicly available.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList


class CombineOptions(threading.local):
    """A container for cube combination options.

    Controls for generalised merge/concatenate options.  These are used as controls for
    both the :func:`iris.util.combine_cubes` utility method and the core Iris loading
    functions : see also  :data:`iris.loading.LoadPolicy`.

    It specifies a number of possible operations which may be applied to a list of
    cubes, in a definite sequence, all of which tend to combine cubes into a smaller
    number of larger or higher-dimensional cubes.

    Notes
    -----
    The individual configurable options are :

    * ``merge_concat_sequence`` = "m" / "c" / "cm" / "mc"
        Specifies whether to apply :meth:`~iris.cube.CubeList.merge`, or
        :meth:`~iris.cube.CubeList.concatenate` operations, or both, in either order.

    * ``merge_uses_unique`` = True / False
        When True, any merge operation will error if its result contains multiple
        identical cubes.  Otherwise (unique=False), that is a permitted result.

        .. Note::

            By default, in a normal :meth:`~iris.cube.CubeList.merge` operation on a
            :class:`~iris.cube.CubeList`, ``unique`` defaults to ``True``.
            For loading operations, however, the default is ``unique=False``, as this
            produces the intended behaviour when loading with multiple constraints.

    * ``repeat_until_unchanged`` = True / False
        When enabled, the configured "combine" operation will be repeated until the
        result is stable (no more cubes are combined).

    Several common sets of options are provided in :data:`~iris.LOAD_POLICY.SETTINGS` :

    *  ``"legacy"``
        Produces loading behaviour identical to Iris versions < 3.11, i.e. before the
        varying hybrid references were supported.

    * ``"default"``
        As "legacy" except that ``support_multiple_references=True``.  This differs
        from "legacy" only when multiple mergeable reference fields are encountered,
        in which case incoming cubes are extended into the extra dimension, and a
        concatenate step is added.

    * ``"recommended"``
        Enables multiple reference handling, *and* applies a merge step followed by
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

    """

    # Useful constants
    #: Valid option names
    OPTION_KEYS = [
        "merge_concat_sequence",
        "repeat_until_unchanged",
    ]  # this is a list, so we can update it in an inheriting class
    _OPTIONS_ALLOWED_VALUES = {
        "merge_concat_sequence": ("", "m", "c", "mc", "cm"),
        "repeat_until_unchanged": (False, True),
    }
    #: Settings content
    SETTINGS = {
        "legacy": dict(
            merge_concat_sequence="m",
            repeat_until_unchanged=False,
        ),
        "default": dict(
            merge_concat_sequence="m",
            repeat_until_unchanged=False,
        ),
        "recommended": dict(
            merge_concat_sequence="mc",
            repeat_until_unchanged=False,
        ),
        "comprehensive": dict(
            merge_concat_sequence="mc",
            repeat_until_unchanged=True,
        ),
    }
    #: Valid settings names
    SETTINGS_NAMES = list(SETTINGS.keys())

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
            A dictionary of options values, or one of the
            :data:`~iris.LoadPolicy.SETTINGS_NAMES` standard settings names,
            e.g. "legacy" or "comprehensive".
        * kwargs : dict
            Individual option settings, from :data:`~iris.LoadPolicy.OPTION_KEYS`.

        Note
        ----
        Keyword arguments are applied after the 'options' arg, and
        so will take precedence.

        """
        if options is None:
            options_dict = {}
        elif isinstance(options, str) and options in self.SETTINGS:
            options_dict = self.SETTINGS[options]
        elif isinstance(options, dict):
            options_dict = options
        else:
            msg = (
                f"arg `options` has unexpected type {type(options)!r}, "
                f"expected one of (None | str | dcit)."
            )
            raise TypeError(msg)

        # Override any options with keywords
        options_dict = options_dict.copy()  # don't modify original
        options_dict.update(**kwargs)
        bad_keys = [key for key in options_dict if key not in self.OPTION_KEYS]
        if bad_keys:
            msg = f"Unknown options {bad_keys} : valid options are {self.OPTION_KEYS}."
            raise ValueError(msg)

        # Implement all options by changing own content.
        for key, value in options_dict.items():
            setattr(self, key, value)

    def settings(self) -> dict:
        """Return an options dict containing the current settings."""
        return {key: getattr(self, key) for key in self.OPTION_KEYS}

    def __repr__(self):
        msg = f"{self.__class__.__name__}("
        msg += ", ".join(f"{key}={getattr(self, key)!r}" for key in self.OPTION_KEYS)
        msg += ")"
        return msg


def _combine_cubes(cubes: List[Cube], options: dict) -> CubeList:
    """Combine cubes as for load, according to "loading policy" options.

    Applies :meth:`~iris.cube.CubeList.merge`/:meth:`~iris.cube.CubeList.concatenate`
    steps to the given cubes, as determined by the 'settings'.

    Parameters
    ----------
    cubes : list of :class:`~iris.cube.Cube`
        A list of cubes to combine.
    options : dict
        Dictionary of settings options, as described for :class:`iris.CombineOptions`.

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

    if isinstance(cubes, CubeList):
        cubelist = cubes
    else:
        cubelist = CubeList(cubes)

    sequence = options["merge_concat_sequence"]
    while True:
        n_original_cubes = len(cubelist)

        if sequence[0] == "c":
            # concat if it comes first
            cubelist = cubelist.concatenate()
        if "m" in sequence:
            # merge if requested
            # NOTE: this needs "unique=False" to make "iris.load()" work correctly.
            # TODO: make configurable via options.
            cubelist = cubelist.merge(unique=False)
        if sequence[-1] == "c":
            # concat if it comes last
            cubelist = cubelist.concatenate()

        # Repeat if requested, *and* this step reduced the number of cubes
        if not options["repeat_until_unchanged"] or len(cubelist) >= n_original_cubes:
            break

    return cubelist


def _combine_load_cubes(cubes: List[Cube]) -> CubeList:
    # A special version to call _combine_cubes while also implementing the
    # _MULTIREF_DETECTION behaviour
    from iris import LOAD_POLICY

    options = LOAD_POLICY.settings()
    if (
        options["support_multiple_references"]
        and "c" not in options["merge_concat_sequence"]
    ):
        # Add a concatenate to implement the "multiref triggers concatenate" mechanism
        from iris.fileformats.rules import _MULTIREF_DETECTION

        if _MULTIREF_DETECTION.found_multiple_refs:
            options["merge_concat_sequence"] += "c"

    return _combine_cubes(cubes, options)

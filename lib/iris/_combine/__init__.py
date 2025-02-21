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

import threading
from typing import Mapping


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

    * ``equalise_cube_kwargs`` = dict or None
        If not None, this enables and provides keyword controls for a call to the
        :func:`iris.util.equalise_cubes` utility.  If active, this always occurs
        **before** any merge/concatenate phase.

    * ``merge_concat_sequence`` = "m" / "c" / "cm" / "mc"
        Specifies whether to apply :meth:`~iris.cube.CubeList.merge`, or
        :meth:`~iris.cube.CubeList.concatenate` operations, or both, in either order.

    * ``merge_uses_unique`` = True / False
        When True, any merge operation will error if its result contains multiple
        identical cubes.  Otherwise (unique=False), that is a permitted result.

        .. Note::

            By default, in a normal :meth:`~iris.cube.CubeList.merge` operation on a
            :class:`~iris.cube.CubeList`, unique is ``True`` unless specified otherwise.
            For loading operations, however, the default is ``unique=False``, as this
            is required to make sense when making for multiple

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
    OPTION_KEYS = (
        # "support_multiple_references",
        "merge_concat_sequence",
        "repeat_until_unchanged",
    )
    _OPTIONS_ALLOWED_VALUES = {
        # "support_multiple_references": (False, True),
        "merge_concat_sequence": ("", "m", "c", "mc", "cm"),
        "repeat_until_unchanged": (False, True),
    }
    SETTING_NAMES = ("legacy", "default", "recommended", "comprehensive")
    SETTINGS = {
        "legacy": dict(
            # support_multiple_references=False,
            merge_concat_sequence="m",
            repeat_until_unchanged=False,
        ),
        "default": dict(
            # support_multiple_references=True,
            merge_concat_sequence="m",
            repeat_until_unchanged=False,
        ),
        "recommended": dict(
            # support_multiple_references=True,
            merge_concat_sequence="mc",
            repeat_until_unchanged=False,
        ),
        "comprehensive": dict(
            # support_multiple_references=True,
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

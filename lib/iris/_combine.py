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

import contextlib
import threading
from typing import List, Mapping

import iris


class CombineOptions(threading.local):
    """A container for cube combination options.

    Controls for generalised merge/concatenate options : see :data:`iris.LOAD_POLICY`
    and :func:`iris.util.combine_cubes`.

    Also controls the detection and handling of cases where a hybrid coordinate
    uses multiple reference fields during loading : for example, a UM file which
    contains a series of fields describing time-varying orography.

    Options can be set directly, or via :meth:`~iris.LoadPolicy.set`, or changed for
    the scope of a code block with :meth:`~iris.LoadPolicy.context`.

    .. note ::

        The default behaviour will "fix" loading for cases like the time-varying
        orography case described above.  However, this is not strictly
        backwards-compatible.  If this causes problems, you can force identical loading
        behaviour to earlier Iris versions with ``LOAD_POLICY.set("legacy")`` or
        equivalent.

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

        >>> # Show how a CombineOptions acts in the context of a load operation
        >>> # (N.B. the LOAD_POLICY actually *is* a CombineOptions type object)
        >>> path = sample_data_path("time_varying_hybrid_height", "*.pp")
        >>> # "legacy" load behaviour allows merge but not concatenate
        >>> with LOAD_POLICY.context("legacy"):
        ...     cubes = iris.load(path, "x_wind")
        >>> print(cubes)
        0: x_wind / (m s-1)                    (time: 2; model_level_number: 5; latitude: 144; longitude: 192)
        1: x_wind / (m s-1)                    (time: 12; model_level_number: 5; latitude: 144; longitude: 192)
        2: x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)
        >>>
        >>> # "recommended" behaviour enables concatenation
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


def _combine_cubes_inner(
    cubes: List[iris.cube.Cube], options: dict
) -> iris.cube.CubeList:
    """Combine cubes, according to "combine options".

    As described for the main "iris.utils.combine_cubes".

    Parameters
    ----------
    cubes : list of Cube
        Cubes to combine.

    options : dict
        A list of options, as described in CombineOptions.

    Returns
    -------
        CubeList
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


def _combine_load_cubes(cubes):
    # A special version to call _combine_cubes_inner while also implementing the
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

    return _combine_cubes_inner(cubes, options)

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

import contextlib
import threading
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList


class CombineOptions(threading.local):
    """A control object for Iris loading and cube combination options.

    Both the iris loading functions and the "combine_cubes" utility apply a number of
    possible "cube combination" operations to a list of cubes, in a definite sequence,
    all of which tend to combine cubes into a smaller number of larger or
    higher-dimensional cubes.

    This object groups various control options for these behaviours, which apply to
    both the :func:`iris.util.combine_cubes` utility method and the core Iris loading
    functions "iris.load_xxx".

    The :class:`CombineOptions` class defines the allowed control options, while a
    global singleton object :data:`iris.COMBINE_POLICY` holds the current global
    default settings.

    The individual configurable options are :

    * ``equalise_cubes_kwargs`` = (dict or None)
        Specifies keywords for an :func:`iris.util.equalise_cubes` call, to be applied
        before any merge/concatenate step.  If ``None``, or empty, no equalisation step
        is performed.

    * ``merge_concat_sequence`` = "m" / "c" / "cm" / "mc"
        Specifies whether to apply :meth:`~iris.cube.CubeList.merge`, or
        :meth:`~iris.cube.CubeList.concatenate` operations, or both, in either order.

    * ``merge_unique`` = True / False
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

    * ``support_multiple_references`` = True / False
        When enabled, support cases where a hybrid coordinate has multiple reference
        fields : for example, a UM file which contains a series of fields describing a
        time-varying orography.

    Alternatively, certain fixed combinations of options can be selected by a
    "settings" name, one of :data:`CombineOptions.SETTINGS_NAMES` :

    *  ``"legacy"``
        Apply a plain merge step only, i.e. ``merge_concat_sequence="m"``.
        Other options are all "off".
        This produces loading behaviour identical to Iris versions < 3.11, i.e. before
        the varying hybrid references were supported.

    * ``"default"``
        As "legacy" except that ``support_multiple_references=True``.  This differs
        from "legacy" only when multiple mergeable reference fields are encountered,
        in which case incoming cubes are extended into the extra dimension, and a
        concatenate step is added.
        Since the handling of multiple references affects only loading operations,
        for the purposes of calls to :func:`~iris.util.combine_cubes`, this setting is
        *identical* to "legacy".

        .. Warning::

            The ``"default"`` setting **is** the initial default mode.

            This "fixes" loading for cases like the time-varying orography case
            described.  However, this setting is not strictly
            backwards-compatible.  If this causes problems, you can force identical
            loading behaviour to earlier Iris versions (< v3.11) with
            ``COMBINE_POLICY.set("legacy")`` or equivalent.

    * ``"recommended"``
        In addition to the "merge" step, allow a following "concatenate", i.e.
        ``merge_concat_sequence="mc"``.

    * ``"comprehensive"``
        As for "recommended", uses ``merge_concat_sequence="mc"``, but now also
        *repeats* the merge+concatenate steps until no further change is produced,
        i.e. ``repeat_until_unchanged=True``.
        Also applies a prior 'equalise_cubes' call, of the form
        ``equalise_cubes(cubes, apply_all=True)``.

        .. Note::

            The "comprehensive" policy makes a maximum effort to reduce the number of
            cubes to a minimum.  However, it still cannot combine cubes with a mixture
            of matching dimension and scalar coordinates.  This may be supported at
            some later date, but for now is not possible without specific user actions.

    .. testsetup::

        from iris import COMBINE_POLICY
        loadpolicy_old_settings = COMBINE_POLICY.settings()

    .. testcleanup::

        # restore original settings, so as not to upset other tests
        COMBINE_POLICY.set(loadpolicy_old_settings)

    Examples
    --------
    Note: :data:`COMBINE_POLICY` is the global control object, which determines
    the current default options for loading or :func:`iris.util.combine_cubes` calls.
    For the latter case, however, control via argument and keywords is also available.

    .. Note::

        The ``iris.COMBINE_POLICY`` can be adjusted by either:

        1. calling ``iris.COMBINE_POLICY.set(<something>)``, or
        2. using ``with COMBINE_POLICY.context(<something>): ...``, or
        3. assigning a property ``COMBINE_POLICY.<option> = <value>``, such as
           ``COMBINE_POLICY.merge_concat_sequence="cm"``

        What you should **not** ever do is to assign :data:`iris.COMBINE_POLICY` itself,
        e.g. ``iris.COMBINE_POLICY = CombineOptions("legacy")``, since in that case the
        original object still exists, and is still the one in control of load/combine
        operations.  Here, the correct approach would be
        ``iris.COMBINE_POLICY.set("legacy")``.

    >>> COMBINE_POLICY.set("legacy")
    >>> print(COMBINE_POLICY)
    CombineOptions(equalise_cubes_kwargs=None, merge_concat_sequence='m', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=False)
    >>>
    >>> COMBINE_POLICY.support_multiple_references = True
    >>> print(COMBINE_POLICY)
    CombineOptions(equalise_cubes_kwargs=None, merge_concat_sequence='m', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=True)

    >>> COMBINE_POLICY.set(merge_concat_sequence="cm")
    >>> print(COMBINE_POLICY)
    CombineOptions(equalise_cubes_kwargs=None, merge_concat_sequence='cm', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=True)

    >>> with COMBINE_POLICY.context("comprehensive"):
    ...    print(COMBINE_POLICY)
    CombineOptions(equalise_cubes_kwargs={'apply_all': True}, merge_concat_sequence='mc', merge_unique=False, repeat_until_unchanged=True, support_multiple_references=True)
    >>>
    >>> print(COMBINE_POLICY)
    CombineOptions(equalise_cubes_kwargs=None, merge_concat_sequence='cm', merge_unique=False, repeat_until_unchanged=False, support_multiple_references=True)

    .. Note::

        The name ``iris.LOAD_POLICY`` refers to the same thing as
        ``iris.COMBINE_POLICY``, and is still usable, but is no longer recommended.

    """

    # Useful constants
    #: Valid option names
    OPTION_KEYS = [
        "equalise_cubes_kwargs",  # N.B. gets special treatment in options checking
        "merge_concat_sequence",
        "merge_unique",
        "repeat_until_unchanged",
        "support_multiple_references",
    ]  # this is a list, so we can update it in an inheriting class
    _OPTIONS_ALLOWED_VALUES = {
        "merge_concat_sequence": ("", "m", "c", "mc", "cm"),
        "merge_unique": (True, False),
        "repeat_until_unchanged": (False, True),
        "support_multiple_references": (True, False),
    }
    #: Standard settings dictionaries
    SETTINGS: Dict[str, Dict[str, Any]] = {
        "legacy": dict(
            equalise_cubes_kwargs=None,
            merge_concat_sequence="m",
            merge_unique=False,
            repeat_until_unchanged=False,
            support_multiple_references=False,
        ),
        "default": dict(
            equalise_cubes_kwargs=None,
            merge_concat_sequence="m",
            merge_unique=False,
            repeat_until_unchanged=False,
            support_multiple_references=True,
        ),
        "recommended": dict(
            equalise_cubes_kwargs=None,
            merge_concat_sequence="mc",
            merge_unique=False,
            repeat_until_unchanged=False,
            support_multiple_references=True,
        ),
        "comprehensive": dict(
            equalise_cubes_kwargs={"apply_all": True},
            merge_concat_sequence="mc",
            merge_unique=False,
            repeat_until_unchanged=True,
            support_multiple_references=True,
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
            raise KeyError(f"CombineOptions object has no property '{key}'.")

        if key != "equalise_cubes_kwargs":
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
            options_dict: dict = {}
        elif isinstance(options, str):
            if options in self.SETTINGS:
                options_dict = self.SETTINGS[options]
            else:
                msg = (
                    f"arg 'options'={options!r}, which is not a valid settings name, "
                    f"expected one of {self.SETTINGS_NAMES}."
                )
                raise ValueError(msg)
        elif isinstance(options, dict):
            options_dict = options

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
            from iris import COMBINE_POLICY, sample_data_path

        >>> # Show how a CombineOptions acts in the context of a load operation
        >>> path = sample_data_path("time_varying_hybrid_height", "*.pp")
        >>>
        >>> # Show that "legacy" load behaviour allows merge but not concatenate
        >>> with COMBINE_POLICY.context("legacy"):
        ...     cubes = iris.load(path, "x_wind")
        >>> print(cubes)
        0: x_wind / (m s-1)                    (time: 2; model_level_number: 5; latitude: 144; longitude: 192)
        1: x_wind / (m s-1)                    (time: 12; model_level_number: 5; latitude: 144; longitude: 192)
        2: x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)
        >>>
        >>> # Show how "recommended" behaviour enables concatenation also
        >>> with COMBINE_POLICY.context("recommended"):
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

    def settings(self) -> dict:
        """Return a settings dict containing the current options settings."""
        return {key: getattr(self, key) for key in self.OPTION_KEYS}

    def __repr__(self):
        msg = f"{self.__class__.__name__}("
        msg += ", ".join(f"{key}={getattr(self, key)!r}" for key in self.OPTION_KEYS)
        msg += ")"
        return msg


def _combine_cubes(cubes: List[Cube], options: dict) -> CubeList:
    """Combine cubes as for load, according to "loading policy" options.

    This is the 'inner' implementation called by :func:`iris.util.combine_cubes`.
    Details of the operation and args are described there.

    It is also called by :func:`_combine_load_cubes`, which implements the
    ``support_multiple_references`` action within loading operations.

    Parameters
    ----------
    cubes : list of :class:`~iris.cube.Cube`
        A list of cubes to combine.
    options : dict
        Dictionary of settings options, as described for :class:`iris.CombineOptions`.

    Returns
    -------
    :class:`~iris.cube.CubeList`

    """
    from iris.cube import CubeList

    if isinstance(cubes, CubeList):
        cubelist = cubes
    else:
        cubelist = CubeList(cubes)

    eq_args = options.get("equalise_cubes_kwargs", None)
    if eq_args:
        # Skip missing (or empty) arg, as no effect : see `equalise_cubes`.
        from iris.util import equalise_cubes

        equalise_cubes(cubelist, **eq_args)

    sequence = options["merge_concat_sequence"]
    merge_unique = options.get("merge_unique", False)
    while True:
        n_original_cubes = len(cubelist)

        if sequence[0] == "c":
            # concat if it comes first
            cubelist = cubelist.concatenate()
        if "m" in sequence:
            # merge if requested.
            # NOTE: the 'unique' arg is configurable in the combine options.
            # All CombineOptions settings have "unique=False", as that is needed for
            #  "iris.load_xxx()" functions to work correctly.  However, the default
            #  for CubeList.merge() is "unique=True".
            cubelist = cubelist.merge(unique=merge_unique)
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
    from iris import COMBINE_POLICY

    options = COMBINE_POLICY.settings()
    if (
        options["support_multiple_references"]
        and "c" not in options["merge_concat_sequence"]
    ):
        # Add a concatenate to implement the "multiref triggers concatenate" mechanism
        from iris.fileformats.rules import _MULTIREF_DETECTION

        if _MULTIREF_DETECTION.found_multiple_refs:
            options["merge_concat_sequence"] += "c"

    return _combine_cubes(cubes, options)


#: An object to control default cube combination and loading options
COMBINE_POLICY = CombineOptions()

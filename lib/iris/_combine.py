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

from typing import List

import iris


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

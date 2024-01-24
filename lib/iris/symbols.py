# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Contains symbol definitions for use with :func:`iris.plot.symbols`."""

import itertools
import math

from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np

__all__ = ("CLOUD_COVER",)


# The thickness to use for lines, circles, etc.
_THICKNESS = 0.1


def _make_merged_patch(paths):
    # Convert a list of Path instances into a single, black PathPatch.

    # Prepare empty vertex/code arrays for the merged path.
    # The vertex array is initially flat for convenient initialisation,
    # but is then reshaped to (N, 2).
    total_len = sum(len(path) for path in paths)
    all_vertices = np.empty(total_len * 2)
    all_codes = np.empty(total_len, dtype=Path.code_type)

    # Copy vertex/code details from the source paths
    all_segments = itertools.chain(*(path.iter_segments() for path in paths))
    i_vertices = 0
    i_codes = 0
    for vertices, code in all_segments:
        n_vertices = len(vertices)
        all_vertices[i_vertices : i_vertices + n_vertices] = vertices
        i_vertices += n_vertices

        n_codes = n_vertices // 2
        if code == Path.STOP:
            code = Path.MOVETO
        all_codes[i_codes : i_codes + n_codes] = code
        i_codes += n_codes

    all_vertices.shape = (total_len, 2)

    return PathPatch(Path(all_vertices, all_codes), facecolor="black", edgecolor="none")


def _ring_path():
    # Returns a Path for a hollow ring.
    # The outer radius is 1, the inner radius is 1 - _THICKNESS.
    circle = Path.unit_circle()
    inner_radius = 1.0 - _THICKNESS
    vertices = np.concatenate(
        [circle.vertices[:-1], circle.vertices[-2::-1] * inner_radius]
    )
    codes = np.concatenate([circle.codes[:-1], circle.codes[:-1]])
    return Path(vertices, codes)


def _vertical_bar_path():
    # Returns a Path for a vertical rectangle, with width _THICKNESS, that will
    # nicely overlap the result of _ring_path().
    width = _THICKNESS / 2.0
    inner_radius = 1.0 - _THICKNESS
    vertices = np.array(
        [
            [-width, -inner_radius],
            [width, -inner_radius],
            [width, inner_radius],
            [-width, inner_radius],
            [-width, inner_radius],
        ]
    )
    codes = np.array(
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    )
    return Path(vertices, codes)


def _slot_path():
    # Returns a Path for a filled unit circle with a vertical rectangle
    # removed.
    circle = Path.unit_circle()
    vertical_bar = _vertical_bar_path()
    vertices = np.concatenate([circle.vertices[:-1], vertical_bar.vertices[-2::-1]])
    codes = np.concatenate([circle.codes[:-1], vertical_bar.codes[:-1]])
    return Path(vertices, codes)


def _left_bar_path():
    # Returns a Path for the left-hand side of a horizontal rectangle, with
    # height _THICKNESS, that will nicely overlap the result of _ring_path().
    inner_radius = 1.0 - _THICKNESS
    height = _THICKNESS / 2.0
    vertices = np.array(
        [
            [-inner_radius, -height],
            [0, -height],
            [0, height],
            [-inner_radius, height],
            [-inner_radius, height],
        ]
    )
    codes = np.array(
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    )
    return Path(vertices, codes)


def _slash_path():
    # Returns a Path for diagonal, bottom-left to top-right rectangle, with
    # width _THICKNESS, that will nicely overlap the result of _ring_path().
    half_width = _THICKNESS / 2.0
    central_radius = 1.0 - half_width

    cos45 = math.cos(math.radians(45))

    end_point_offset = cos45 * central_radius
    half_width_offset = cos45 * half_width

    vertices = np.array(
        [
            [
                -end_point_offset - half_width_offset,
                -end_point_offset + half_width_offset,
            ],
            [
                -end_point_offset + half_width_offset,
                -end_point_offset - half_width_offset,
            ],
            [
                end_point_offset + half_width_offset,
                end_point_offset - half_width_offset,
            ],
            [
                end_point_offset - half_width_offset,
                end_point_offset + half_width_offset,
            ],
            [
                -end_point_offset - half_width_offset,
                -end_point_offset + half_width_offset,
            ],
        ]
    )
    codes = np.array(
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    )
    return Path(vertices, codes)


def _backslash_path():
    # Returns a Path for diagonal, top-left to bottom-right rectangle, with
    # width _THICKNESS, that will nicely overlap the result of _ring_path().
    half_width = _THICKNESS / 2.0
    central_radius = 1.0 - half_width

    cos45 = math.cos(math.radians(45))

    end_point_offset = cos45 * central_radius
    half_width_offset = cos45 * half_width

    vertices = np.array(
        [
            [
                -end_point_offset - half_width_offset,
                end_point_offset - half_width_offset,
            ],
            [
                end_point_offset - half_width_offset,
                -end_point_offset - half_width_offset,
            ],
            [
                end_point_offset + half_width_offset,
                -end_point_offset + half_width_offset,
            ],
            [
                -end_point_offset + half_width_offset,
                end_point_offset + half_width_offset,
            ],
            [
                -end_point_offset - half_width_offset,
                end_point_offset - half_width_offset,
            ],
        ]
    )
    codes = np.array(
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    )
    return Path(vertices, codes)


def _wedge_fix(wedge_path):
    """Fix the problem with Path.wedge.

    Fixes the problem with Path.wedge where it doesn't initialise the first,
    and last two vertices.

    This fix should not have any side-effects once Path.wedge has been fixed,
    but will then be redundant and should be removed.

    This is fixed in MPL v1.3, raising a RuntimeError. A check is performed to
    allow for backward compatibility with MPL v1.2.x.

    """
    if wedge_path.vertices.flags.writeable:
        wedge_path.vertices[0] = 0
        wedge_path.vertices[-2:] = 0
    return wedge_path


CLOUD_COVER = {
    0: [_ring_path()],
    1: [_ring_path(), _vertical_bar_path()],
    2: [_ring_path(), _wedge_fix(Path.wedge(0, 90))],
    3: [_ring_path(), _wedge_fix(Path.wedge(0, 90)), _vertical_bar_path()],
    4: [_ring_path(), Path.unit_circle_righthalf()],
    5: [_ring_path(), Path.unit_circle_righthalf(), _left_bar_path()],
    6: [_ring_path(), _wedge_fix(Path.wedge(-180, 90))],
    7: [_slot_path()],
    8: [Path.unit_circle()],
    9: [_ring_path(), _slash_path(), _backslash_path()],
}
"""
A dictionary mapping WMO cloud cover codes to their corresponding symbol.

See https://www.wmo.int/pages/prog/www/DPFS/documents/485_Vol_I_en_colour.pdf
    Part II, Appendix II.4, Graphical Representation of Data, Analyses
    and Forecasts

"""


def _convert_paths_to_patches():
    # Convert the symbols defined as lists-of-paths into patches.
    for code, symbol in CLOUD_COVER.items():
        CLOUD_COVER[code] = _make_merged_patch(symbol)


_convert_paths_to_patches()

# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Helper routines specific to cell method parsing for netcdf-CF loading.

"""
import re
from typing import List
import warnings

from iris.coords import CellMethod

# Cell methods.
_CM_KNOWN_METHODS = [
    "point",
    "sum",
    "mean",
    "maximum",
    "minimum",
    "mid_range",
    "standard_deviation",
    "variance",
    "mode",
    "median",
]

_CM_COMMENT = "comment"
_CM_EXTRA = "extra"
_CM_INTERVAL = "interval"
_CM_METHOD = "method"
_CM_NAME = "name"
_CM_PARSE_NAME = re.compile(r"([\w_]+\s*?:\s+)+")
_CM_PARSE = re.compile(
    r"""
                           (?P<name>([\w_]+\s*?:\s+)+)
                           (?P<method>[\w_\s]+(?![\w_]*\s*?:))\s*
                           (?:
                               \(\s*
                               (?P<extra>.+)
                               \)\s*
                           )?
                       """,
    re.VERBOSE,
)


class UnknownCellMethodWarning(Warning):
    pass


def _split_cell_methods(nc_cell_methods: str) -> List[re.Match]:
    """
    Split a CF cell_methods attribute string into a list of zero or more cell
    methods, each of which is then parsed with a regex to return a list of match
    objects.

    Args:

    * nc_cell_methods: The value of the cell methods attribute to be split.

    Returns:

    * nc_cell_methods_matches: A list of the re.Match objects associated with
      each parsed cell method

    Splitting is done based on words followed by colons outside of any brackets.
    Validation of anything other than being laid out in the expected format is
    left to the calling function.
    """

    # Find name candidates
    name_start_inds = []
    for m in _CM_PARSE_NAME.finditer(nc_cell_methods):
        name_start_inds.append(m.start())

    # Remove those that fall inside brackets
    bracket_depth = 0
    for ind, cha in enumerate(nc_cell_methods):
        if cha == "(":
            bracket_depth += 1
        elif cha == ")":
            bracket_depth -= 1
            if bracket_depth < 0:
                msg = (
                    "Cell methods may be incorrectly parsed due to mismatched "
                    "brackets"
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
        if bracket_depth > 0 and ind in name_start_inds:
            name_start_inds.remove(ind)

    # List tuples of indices of starts and ends of the cell methods in the string
    name_start_inds.append(len(nc_cell_methods))
    method_indices = list(zip(name_start_inds[:-1], name_start_inds[1:]))

    # Index the string and match against each substring
    nc_cell_methods_matches = []
    for start_ind, end_ind in method_indices:
        nc_cell_method_str = nc_cell_methods[start_ind:end_ind]
        nc_cell_method_match = _CM_PARSE.match(nc_cell_method_str.strip())
        if not nc_cell_method_match:
            msg = (
                f"Failed to fully parse cell method string: {nc_cell_methods}"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        nc_cell_methods_matches.append(nc_cell_method_match)

    return nc_cell_methods_matches


def parse_cell_methods(nc_cell_methods):
    """
    Parse a CF cell_methods attribute string into a tuple of zero or
    more CellMethod instances.

    Args:

    * nc_cell_methods (str):
        The value of the cell methods attribute to be parsed.

    Returns:

    * cell_methods
        An iterable of :class:`iris.coords.CellMethod`.

    Multiple coordinates, intervals and comments are supported.
    If a method has a non-standard name a warning will be issued, but the
    results are not affected.

    """

    cell_methods = []
    if nc_cell_methods is not None:
        splits = _split_cell_methods(nc_cell_methods)
        if not splits:
            msg = (
                f"NetCDF variable cell_methods of {nc_cell_methods!r} "
                "contains no valid cell methods."
            )
            warnings.warn(msg, UserWarning)
        for m in splits:
            d = m.groupdict()
            method = d[_CM_METHOD]
            method = method.strip()
            # Check validity of method, allowing for multi-part methods
            # e.g. mean over years.
            method_words = method.split()
            if method_words[0].lower() not in _CM_KNOWN_METHODS:
                msg = "NetCDF variable contains unknown cell method {!r}"
                warnings.warn(
                    msg.format("{}".format(method_words[0])),
                    UnknownCellMethodWarning,
                )
            d[_CM_METHOD] = method
            name = d[_CM_NAME]
            name = name.replace(" ", "")
            name = name.rstrip(":")
            d[_CM_NAME] = tuple([n for n in name.split(":")])
            interval = []
            comment = []
            if d[_CM_EXTRA] is not None:
                #
                # tokenise the key words and field colon marker
                #
                d[_CM_EXTRA] = d[_CM_EXTRA].replace(
                    "comment:", "<<comment>><<:>>"
                )
                d[_CM_EXTRA] = d[_CM_EXTRA].replace(
                    "interval:", "<<interval>><<:>>"
                )
                d[_CM_EXTRA] = d[_CM_EXTRA].split("<<:>>")
                if len(d[_CM_EXTRA]) == 1:
                    comment.extend(d[_CM_EXTRA])
                else:
                    next_field_type = comment
                    for field in d[_CM_EXTRA]:
                        field_type = next_field_type
                        index = field.rfind("<<interval>>")
                        if index == 0:
                            next_field_type = interval
                            continue
                        elif index > 0:
                            next_field_type = interval
                        else:
                            index = field.rfind("<<comment>>")
                            if index == 0:
                                next_field_type = comment
                                continue
                            elif index > 0:
                                next_field_type = comment
                        if index != -1:
                            field = field[:index]
                        field_type.append(field.strip())
            #
            # cater for a shared interval over multiple axes
            #
            if len(interval):
                if len(d[_CM_NAME]) != len(interval) and len(interval) == 1:
                    interval = interval * len(d[_CM_NAME])
            #
            # cater for a shared comment over multiple axes
            #
            if len(comment):
                if len(d[_CM_NAME]) != len(comment) and len(comment) == 1:
                    comment = comment * len(d[_CM_NAME])
            d[_CM_INTERVAL] = tuple(interval)
            d[_CM_COMMENT] = tuple(comment)
            cell_method = CellMethod(
                d[_CM_METHOD],
                coords=d[_CM_NAME],
                intervals=d[_CM_INTERVAL],
                comments=d[_CM_COMMENT],
            )
            cell_methods.append(cell_method)
    return tuple(cell_methods)

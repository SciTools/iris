# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Dictionary operations for dealing with the CubeAttrsDict "split"-style attribute dictionaries.

The idea here is to convert a split-dictionary into a "plain" one for calculations,
whose keys are all pairs of the form ('global', <keyname>) or ('local', <keyname>).
And to convert back again after the operation, if the result is a dictionary.

For "strict" operations this clearly does all that is needed.  For lenient ones,
we _might_ want for local+global attributes of the same name to interact.
However, on careful consideration, it seems that this is not actually desirable for
any of the common-metadata operations.
So, we simply treat "global" and "local" attributes of the same name as entirely
independent. Which happily is also the easiest to code, and to explain.
"""

from collections.abc import Mapping, Sequence
from functools import wraps


def _convert_splitattrs_to_pairedkeys_dict(dic):
    """Convert a split-attributes dictionary to a "normal" dict.

    Transform a :class:`~iris.cube.CubeAttributesDict` "split" attributes dictionary
    into a 'normal' :class:`dict`, with paired keys of the form ('global', name) or
    ('local', name).

    If the input is *not* a split-attrs dict, it is converted to one before
    transforming it.  This will assign its keys to global/local depending on a standard
    set of choices (see :class:`~iris.cube.CubeAttributesDict`).
    """
    from iris.cube import CubeAttrsDict

    # Convert input to CubeAttrsDict
    if not hasattr(dic, "globals") or not hasattr(dic, "locals"):
        dic = CubeAttrsDict(dic)

    def _global_then_local_items(dic):
        # Routine to produce global, then local 'items' in order, and with all keys
        # "labelled" as local or global type, to ensure they are all unique.
        for key, value in dic.globals.items():
            yield ("global", key), value
        for key, value in dic.locals.items():
            yield ("local", key), value

    return dict(_global_then_local_items(dic))


def _convert_pairedkeys_dict_to_splitattrs(dic):
    """Convert an input with global/local paired keys back into a split-attrs dict.

    For now, this is always and only a :class:`iris.cube.CubeAttrsDict`.
    """
    from iris.cube import CubeAttrsDict

    result = CubeAttrsDict()
    for key, value in dic.items():
        keytype, keyname = key
        if keytype == "global":
            result.globals[keyname] = value
        else:
            assert keytype == "local"
            result.locals[keyname] = value
    return result


def adjust_for_split_attribute_dictionaries(operation):
    """Generate attribute-dictionaries to work with split attributes.

    Decorator to make a function of attribute-dictionaries work with split attributes.

    The wrapped function of attribute-dictionaries is currently always one of "equals",
    "combine" or "difference", with signatures like :
        equals(left: dict, right: dict) -> bool
        combine(left: dict, right: dict) -> dict
        difference(left: dict, right: dict) -> None | (dict, dict)

    The results of the wrapped operation are either :
    * for "equals" (or "__eq__") :  a boolean
    * for "combine" :  a (converted) attributes-dictionary
    * for "difference" :  a list of (None or "pair"), where a pair contains two
        dictionaries

    Before calling the wrapped operation, its inputs (left, right) are modified by
    converting any "split" dictionaries to a form where the keys are pairs
    of the form ("global", name) or ("local", name).

    After calling the wrapped operation, for "combine" or "difference", the result can
    contain a dictionary or dictionaries.  These are then transformed back from the
    'converted' form to split-attribute dictionaries, before returning.

    "Split" dictionaries  are all of class :class:`~iris.cube.CubeAttrsDict`, since
    the only usage of 'split' attribute dictionaries is in Cubes (i.e. they are not
    used for cube components).

    """

    @wraps(operation)
    def _inner_function(*args, **kwargs):
        # Convert all inputs into 'pairedkeys' type dicts
        args = [_convert_splitattrs_to_pairedkeys_dict(arg) for arg in args]

        result = operation(*args, **kwargs)

        # Convert known specific cases of 'pairedkeys' dicts in the result, and convert
        # those back into split-attribute dictionaries.
        if isinstance(result, Mapping):
            # Fix a result which is a single dictionary -- for "combine"
            result = _convert_pairedkeys_dict_to_splitattrs(result)
        elif isinstance(result, Sequence) and len(result) == 2:
            # Fix a result which is a pair of dictionaries -- for "difference"
            left, right = result
            left, right = (
                _convert_pairedkeys_dict_to_splitattrs(left),
                _convert_pairedkeys_dict_to_splitattrs(right),
            )
            result = result.__class__([left, right])
        # ELSE: leave other types of result unchanged. E.G. None, bool

        return result

    return _inner_function

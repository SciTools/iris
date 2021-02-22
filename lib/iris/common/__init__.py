# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A package for provisioning common Iris infrastructure.

"""

from typing import Mapping

from .lenient import *
from .metadata import *
from .mixin import *
from .resolve import *
from ..util import guess_coord_axis


def filter_cf(
    instances,
    item=None,
    standard_name=None,
    long_name=None,
    var_name=None,
    attributes=None,
    axis=None,
):
    """
    Filter a collection of objects by their metadata to fit the given metadata
    criteria. Criteria be one or both of: specific properties / other objects
    carrying metadata to be matched.

    Args:

    * instances
        An iterable of objects to be filtered.

    Kwargs:

    * item
        Either

        (a) a :attr:`standard_name`, :attr:`long_name`, or
        :attr:`var_name`. Defaults to value of `default`
        (which itself defaults to `unknown`) as defined in
        :class:`~iris.common.CFVariableMixin`.

        (b) a 'coordinate' instance with metadata equal to that of
        the desired coordinates. Accepts either a
        :class:`~iris.coords.DimCoord`, :class:`~iris.coords.AuxCoord`,
        :class:`~iris.aux_factory.AuxCoordFactory`,
        :class:`~iris.common.CoordMetadata` or
        :class:`~iris.common.DimCoordMetadata` or
        :class:`~iris.experimental.ugrid.ConnectivityMetadata`.
    * standard_name
        The CF standard name of the desired coordinate. If None, does not
        check for standard name.
    * long_name
        An unconstrained description of the coordinate. If None, does not
        check for long_name.
    * var_name
        The netCDF variable name of the desired coordinate. If None, does
        not check for var_name.
    * attributes
        A dictionary of attributes desired on the coordinates. If None,
        does not check for attributes.
    * axis
        The desired coordinate axis, see
        :func:`~iris.util.guess_coord_axis`. If None, does not check for
        axis. Accepts the values 'X', 'Y', 'Z' and 'T' (case-insensitive).

    Returns:
        A list of the objects supplied in the ``instances`` argument, limited
        to only those that matched the given criteria.

    """
    name = None
    obj = None

    if isinstance(item, str):
        name = item
    else:
        obj = item

    result = instances

    if name is not None:
        result = [instance for instance in result if instance.name() == name]

    if standard_name is not None:
        result = [
            instance
            for instance in result
            if instance.standard_name == standard_name
        ]

    if long_name is not None:
        result = [
            instance for instance in result if instance.long_name == long_name
        ]

    if var_name is not None:
        result = [
            instance for instance in result if instance.var_name == var_name
        ]

    if attributes is not None:
        if not isinstance(attributes, Mapping):
            msg = (
                "The attributes keyword was expecting a dictionary "
                "type, but got a %s instead." % type(attributes)
            )
            raise ValueError(msg)

        def attr_filter(instance):
            return all(
                k in instance.attributes
                and metadata._hexdigest(instance.attributes[k])
                == metadata._hexdigest(v)
                for k, v in attributes.items()
            )

        result = [instance for instance in result if attr_filter(instance)]

    if axis is not None:
        axis = axis.upper()
        result = [
            instance
            for instance in result
            if guess_coord_axis(instance) == axis
        ]

    if obj is not None:
        if hasattr(obj, "__class__") and issubclass(
            obj.__class__, BaseMetadata
        ):
            target_metadata = obj
        else:
            target_metadata = obj.metadata

        result = [
            instance
            for instance in result
            if instance.metadata == target_metadata
        ]

    return result

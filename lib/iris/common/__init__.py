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


def filter_cf(
    instances,
    item=None,
    standard_name=None,
    long_name=None,
    var_name=None,
    attributes=None,
):
    """
    Filter a list of :class:`iris.common.CFVariableMixin` subclasses to fit
    the given criteria.

    Kwargs:

    * item
        Either

        (a) a :attr:`standard_name`, :attr:`long_name`, or
        :attr:`var_name`. Defaults to value of `default`
        (which itself defaults to `unknown`) as defined in
        :class:`iris.common.CFVariableMixin`.

        (b) a 'coordinate' instance with metadata equal to that of
        the desired coordinates. Accepts either a
        :class:`iris.coords.DimCoord`, :class:`iris.coords.AuxCoord`,
        :class:`iris.aux_factory.AuxCoordFactory`,
        :class:`iris.common.CoordMetadata` or
        :class:`iris.common.DimCoordMetadata` or
        :class:`iris.experimental.ugrid.ConnectivityMetadata`.
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

    """
    name = None
    instance = None

    if isinstance(item, str):
        name = item
    else:
        instance = item

    result = instances

    if name is not None:
        result = [
            instance_ for instance_ in result if instance_.name() == name
        ]

    if standard_name is not None:
        result = [
            instance_
            for instance_ in result
            if instance_.standard_name == standard_name
        ]

    if long_name is not None:
        result = [
            instance_
            for instance_ in result
            if instance_.long_name == long_name
        ]

    if var_name is not None:
        result = [
            instance_ for instance_ in result if instance_.var_name == var_name
        ]

    if attributes is not None:
        if not isinstance(attributes, Mapping):
            msg = (
                "The attributes keyword was expecting a dictionary "
                "type, but got a %s instead." % type(attributes)
            )
            raise ValueError(msg)

        def attr_filter(instance_):
            return all(
                k in instance_.attributes
                and metadata._hexdigest(instance_.attributes[k])
                == metadata._hexdigest(v)
                for k, v in attributes.items()
            )

        result = [instance_ for instance_ in result if attr_filter(instance_)]

    if instance is not None:
        if hasattr(instance, "__class__") and issubclass(
            instance.__class__, BaseMetadata
        ):
            target_metadata = instance
        else:
            target_metadata = instance.metadata

        result = [
            instance_
            for instance_ in result
            if instance_.metadata == target_metadata
        ]

    return result

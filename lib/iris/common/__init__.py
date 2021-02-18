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
        if hasattr(instance, "__class__") and instance.__class__ in (
            CoordMetadata,
            DimCoordMetadata,
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

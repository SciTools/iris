# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from abc import ABCMeta
from collections import namedtuple
from collections.abc import Iterable


__all__ = [
    "BaseMetadata",
    "CellMeasureMetadata",
    "CoordMetadata",
    "CubeMetadata",
]


class _BaseMeta(ABCMeta):
    """
    Meta-class to support the convenience of creating a namedtuple from
    names/members of the metadata class hierarchy.

    """

    def __new__(mcs, name, bases, namespace):
        if "_names" in namespace and not getattr(
            namespace["_names"], "__isabstractmethod__", False
        ):
            namespace_names = namespace["_names"]
            names = []
            for base in bases:
                if hasattr(base, "_names"):
                    base_names = base._names
                    is_abstract = getattr(
                        base_names, "__isabstractmethod__", False
                    )
                    if not is_abstract:
                        if (
                            not isinstance(base_names, Iterable)
                        ) or isinstance(base_names, str):
                            base_names = (base_names,)
                        names.extend(base_names)

            if (not isinstance(namespace_names, Iterable)) or isinstance(
                namespace_names, str
            ):
                namespace_names = (namespace_names,)

            names.extend(namespace_names)

            if names:
                item = namedtuple(f"{name}Namedtuple", names)
                bases = list(bases)
                # Influence the appropriate MRO.
                bases.insert(0, item)
                bases = tuple(bases)

        return super().__new__(mcs, name, bases, namespace)


class BaseMetadata(metaclass=_BaseMeta):
    """
    Container for common metadata.

    """

    _names = (
        "standard_name",
        "long_name",
        "var_name",
        "units",
        "attributes",
    )

    __slots__ = ()

    def name(self, default="unknown"):

        """
        Returns a human-readable name.

        First it tries self.standard_name, then it tries the 'long_name'
        attribute, then the 'var_name' attribute, before falling back to
        the value of `default` (which itself defaults to 'unknown').

        """
        return self.standard_name or self.long_name or self.var_name or default

    def __lt__(self, other):
        #
        # Support Python2 behaviour for a "<" operation involving a
        # "NoneType" operand.
        #
        if not isinstance(other, self.__class__):
            return NotImplemented

        def _sort_key(item):
            keys = []
            for field in item._fields:
                value = getattr(item, field)
                keys.extend((value is not None, value))
            return tuple(keys)

        return _sort_key(self) < _sort_key(other)


class CellMeasureMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.coords.CellMeasure`.

    """

    _names = "measure"

    __slots__ = ()


class CoordMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.coords.Coord`.

    """

    _names = ("coord_system", "climatological")

    __slots__ = ()


class CubeMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.cube.Cube`.

    """

    _names = "cell_methods"

    __slots__ = ()

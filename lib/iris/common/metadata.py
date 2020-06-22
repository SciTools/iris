# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from abc import ABCMeta
from collections import namedtuple
from collections.abc import Iterable, Mapping
from functools import wraps
import re

from .lenient import LENIENT, lenient_service, qualname


__all__ = [
    "SERVICES_COMBINE",
    "SERVICES_DIFFERENCE",
    "SERVICES_EQUAL",
    "SERVICES",
    "AncillaryVariableMetadata",
    "BaseMetadata",
    "CellMeasureMetadata",
    "CoordMetadata",
    "CubeMetadata",
    "metadata_manager_factory",
]


# https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_set_components.html#object_name
_TOKEN_PARSE = re.compile(r"""^[a-zA-Z0-9][\w\.\+\-@]*$""")


class _NamedTupleMeta(ABCMeta):
    """
    Meta-class to support the convenience of creating a namedtuple from
    names/members of the metadata class hierarchy.

    """

    def __new__(mcs, name, bases, namespace):
        names = []

        for base in bases:
            if hasattr(base, "_members"):
                base_names = getattr(base, "_members")
                is_abstract = getattr(
                    base_names, "__isabstractmethod__", False
                )
                if not is_abstract:
                    if (not isinstance(base_names, Iterable)) or isinstance(
                        base_names, str
                    ):
                        base_names = (base_names,)
                    names.extend(base_names)

        if "_members" in namespace and not getattr(
            namespace["_members"], "__isabstractmethod__", False
        ):
            namespace_names = namespace["_members"]

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


class BaseMetadata(metaclass=_NamedTupleMeta):
    """
    Container for common metadata.

    """

    DEFAULT_NAME = "unknown"  # the fall-back name for metadata identity

    _members = (
        "standard_name",
        "long_name",
        "var_name",
        "units",
        "attributes",
    )

    __slots__ = ()

    @lenient_service
    def __eq__(self, other):
        """
        Determine whether the associated metadata members are equivalent.

        Args:

        * other (metadata):
            A metadata instance of the same type.

        Returns:
            Boolean.

        """
        result = NotImplemented
        if hasattr(other, "__class__") and other.__class__ is self.__class__:
            if LENIENT(self.__eq__) or LENIENT(self.equal):
                # Perform "lenient" equality.
                print("lenient __eq__")
                result = self._compare_lenient(other)
            else:
                # Perform "strict" equality.
                result = super().__eq__(other)

        return result

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

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result

        return result

    def _api_common(
        self, other, func_service, func_operation, action, lenient=None
    ):
        """
        Common entry-point for API facing lenient service methods.

        Args:

        * other (metadata):
            A metadata instance of the same type.

        * func_service (callable):
            The parent service method offering the API entry-point to the service.

        * func_operation (callable):
            The parent service method that provides the actual service.

        * action (str):
            The verb describing the service operation.

        Kwargs:

        * lenient (boolean):
            Enable/disable the lenient service operation. The default is to automatically
            detect whether this lenient service operation is enabled.

        Returns:
            The result of the service operation to the parent service caller.

        """
        if (
            not hasattr(other, "__class__")
            or other.__class__ is not self.__class__
        ):
            emsg = "Cannot {} {!r} with {!r}."
            raise TypeError(
                emsg.format(action, self.__class__.__name__, type(other))
            )

        if lenient is None:
            result = func_operation(other)
        else:
            if lenient:
                # Use qualname to disassociate from the instance bounded method.
                args, kwargs = (qualname(func_service),), dict()
            else:
                # Use qualname to guarantee that the instance bounded method
                # is a hashable key.
                args, kwargs = (), {qualname(func_service): False}

            with LENIENT.context(*args, **kwargs):
                result = func_operation(other)

        return result

    def _combine(self, other):
        """Perform associated metadata member combination."""
        if LENIENT(self.combine):
            # Perform "lenient" combine.
            print("lenient combine")
            values = self._combine_lenient(other)
        else:
            # Perform "strict" combine.
            print("strict combine")

            def func(field):
                value = getattr(self, field)
                return value if value == getattr(other, field) else None

            # Note that, for strict we use "_fields" not "_members".
            values = [func(field) for field in self._fields]

        return values

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members.

        Args:

        * other (BaseMetadata):
            The other metadata participating in the lenient combination.

        Returns:
            A list of combined metadata member values.

        """

        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            result = None
            if field == "units":
                # Perform "strict" combination for "units".
                result = left if left == right else None
            elif self._is_attributes(field, left, right):
                result = self._combine_lenient_attributes(left, right)
            else:
                if left == right:
                    result = left
                elif left is None:
                    result = right
                elif right is None:
                    result = left
            return result

        # Note that, we use "_members" not "_fields".
        return [func(field) for field in BaseMetadata._members]

    @staticmethod
    def _combine_lenient_attributes(left, right):
        """Leniently combine the dictionary members together."""
        sleft = set(left.items())
        sright = set(right.items())
        # Intersection of common items.
        common = sleft & sright
        # Items in sleft different from sright.
        dsleft = dict(sleft - sright)
        # Items in sright different from sleft.
        dsright = dict(sright - sleft)
        # Intersection of common item keys with different values.
        keys = set(dsleft.keys()) & set(dsright.keys())
        # Remove (in-place) common item keys with different values.
        [dsleft.pop(key) for key in keys]
        [dsright.pop(key) for key in keys]
        # Now bring the result together.
        result = dict(common)
        result.update(dsleft)
        result.update(dsright)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members.

        Args:

        * other (BaseMetadata):
            The other metadata participating in the lenient comparison.

        Returns:
            Boolean.

        """
        result = False

        # Use the "name" method to leniently compare "standard_name",
        # "long_name", and "var_name" in a well defined way.
        if self.name() == other.name():

            def func(field):
                left = getattr(self, field)
                right = getattr(other, field)
                if field == "units":
                    # Perform "strict" compare for "units".
                    result = left == right
                elif self._is_attributes(field, left, right):
                    result = self._compare_lenient_attributes(left, right)
                else:
                    # Perform "lenient" compare for members.
                    result = (left == right) or left is None or right is None
                return result

            # Note that, we use "_members" not "_fields".
            result = all([func(field) for field in BaseMetadata._members])

        return result

    @staticmethod
    def _compare_lenient_attributes(left, right):
        """Perform lenient compare between the dictionary members."""
        sleft = set(left.items())
        sright = set(right.items())
        # Items in sleft different from sright.
        dsleft = dict(sleft - sright)
        # Items in sright different from sleft.
        dsright = dict(sright - sleft)
        # Intersection of common item keys with different values.
        keys = set(dsleft.keys()) & set(dsright.keys())

        return not bool(keys)

    def _difference(self, other):
        """Perform associated metadata member difference."""
        if LENIENT(self.difference):
            # Perform "lenient" difference.
            print("lenient difference")
            values = self._difference_lenient(other)
        else:
            # Perform "strict" difference.
            print("strict difference")

            def func(field):
                left = getattr(self, field)
                right = getattr(other, field)
                if self._is_attributes(field, left, right):
                    result = self._difference_strict_attributes(left, right)
                else:
                    result = None if left == right else (left, right)
                return result

            # Note that, for strict we use "_fields" not "_members".
            values = [func(field) for field in self._fields]

        return values

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members.

        Args:

        * other (BaseMetadata):
            The other metadata participating in the lenient difference.

        Returns:
            A list of difference metadata member values.

        """

        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            if field == "units":
                # Perform "strict" difference for "units".
                result = None if left == right else (left, right)
            elif self._is_attributes(field, left, right):
                result = self._difference_lenient_attributes(left, right)
            else:
                # Perform "lenient" difference for members.
                result = (
                    (left, right)
                    if left is not None and right is not None and left != right
                    else None
                )
            return result

        # Note that, we use "_members" not "_fields".
        return [func(field) for field in BaseMetadata._members]

    @staticmethod
    def _difference_lenient_attributes(left, right):
        """Perform lenient difference between the dictionary members."""
        sleft = set(left.items())
        sright = set(right.items())
        # Items in sleft different from sright.
        dsleft = dict(sleft - sright)
        # Items in sright different from sleft.
        dsright = dict(sright - sleft)
        # Intersection of common item keys with different values.
        keys = set(dsleft.keys()) & set(dsright.keys())
        # Keep (in-place) common item keys with different values.
        [dsleft.pop(key) for key in list(dsleft.keys()) if key not in keys]
        [dsright.pop(key) for key in list(dsright.keys()) if key not in keys]

        if not bool(dsleft) and not bool(dsright):
            result = None
        else:
            result = (dsleft, dsright)

        return result

    @staticmethod
    def _difference_strict_attributes(left, right):
        """Perform strict difference between the dictionary members."""
        sleft = set(left.items())
        sright = set(right.items())
        # Items in sleft different from sright.
        dsleft = dict(sleft - sright)
        # Items in sright different from sleft.
        dsright = dict(sright - sleft)

        if not bool(dsleft) and not bool(dsright):
            result = None
        else:
            result = (dsleft, dsright)

        return result

    @staticmethod
    def _is_attributes(field, left, right):
        """Determine whether we have two 'attributes' dictionaries."""
        return (
            field == "attributes"
            and isinstance(left, Mapping)
            and isinstance(right, Mapping)
        )

    @lenient_service
    def combine(self, other, lenient=None):
        """
        Return a new metadata instance created by combining each of the
        associated metadata members.

        Args:

        * other (metadata):
            A metadata instance of the same type.

        Kwargs:

        * lenient (boolean):
            Enable/disable lenient combination. The default is to automatically
            detect whether this lenient operation is enabled.

        Returns:
            Metadata instance.

        """
        result = self._api_common(
            other, self.combine, self._combine, "combine", lenient=lenient
        )
        return self.__class__(*result)

    @lenient_service
    def difference(self, other, lenient=None):
        """
        Return a new metadata instance created by performing a difference
        comparison between each of the associated metadata members.

        A metadata member returned with a value of "None" indicates that there
        is no difference between the members being compared. Otherwise, a tuple
        of the different values is returned.

        Args:

        * other (metadata):
            A metadata instance of the same type.

        Kwargs:

        * lenient (boolean):
            Enable/disable lenient difference. The default is to automatically
            detect whether this lenient operation is enabled.

        Returns:
            Metadata instance.

        """
        result = self._api_common(
            other, self.difference, self._difference, "differ", lenient=lenient
        )
        return self.__class__(*result)

    @lenient_service
    def equal(self, other, lenient=None):
        """
        Determine whether the associated metadata members are equivalent.

        Args:

        * other (metadata):
            A metadata instance of the same type.

        Kwargs:

        * lenient (boolean):
            Enable/disable lenient equivalence. The default is to automatically
            detect whether this lenient operation is enabled.

        Returns:
            Boolean.

        """
        result = self._api_common(
            other, self.equal, self.__eq__, "compare", lenient=lenient
        )
        return result

    def name(self, default=None, token=False):
        """
        Returns a string name representing the identity of the metadata.

        First it tries standard name, then it tries the long name, then
        the NetCDF variable name, before falling-back to a default value,
        which itself defaults to the string 'unknown'.

        Kwargs:

        * default:
            The fall-back string representing the default name. Defaults to
            the string 'unknown'.
        * token:
            If True, ensures that the name returned satisfies the criteria for
            the characters required by a valid NetCDF name. If it is not
            possible to return a valid name, then a ValueError exception is
            raised. Defaults to False.

        Returns:
            String.

        """

        def _check(item):
            return self.token(item) if token else item

        default = self.DEFAULT_NAME if default is None else default

        result = (
            _check(self.standard_name)
            or _check(self.long_name)
            or _check(self.var_name)
            or _check(default)
        )

        if token and result is None:
            emsg = "Cannot retrieve a valid name token from {!r}"
            raise ValueError(emsg.format(self))

        return result

    @classmethod
    def token(cls, name):
        """
        Determine whether the provided name is a valid NetCDF name and thus
        safe to represent a single parsable token.

        Args:

        * name:
            The string name to verify

        Returns:
            The provided name if valid, otherwise None.

        """
        if name is not None:
            result = _TOKEN_PARSE.match(name)
            name = result if result is None else name

        return name


class AncillaryVariableMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.coords.AncillaryVariableMetadata`.

    """

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)


class CellMeasureMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.coords.CellMeasure`.

    """

    _members = "measure"

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members for cell measures.

        Args:

        * other (CellMeasureMetadata):
            The other cell measure metadata participating in the lenient
            combination.

        Returns:
            A list of combined metadata member values.

        """
        # Perform "strict" combination for "measure".
        value = self.measure if self.measure == other.measure else None
        # Perform lenient combination of the other parent members.
        result = super()._combine_lenient(other)
        result.append(value)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members for cell measures.

        Args:

        * other (CellMeasureMetadata):
            The other cell measure metadata participating in the lenient
            comparison.

        Returns:
            Boolean.

        """
        # Perform "strict" comparison for "measure".
        result = self.measure == other.measure
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members for cell measures.

        Args:

        * other (CellMeasureMetadata):
            The other cell measure metadata participating in the lenient
            difference.

        Returns:
            A list of difference metadata member values.

        """
        # Perform "strict" difference for "measure".
        value = (
            None
            if self.measure == other.measure
            else (self.measure, other.measure)
        )
        # Perform lenient difference of the other parent members.
        result = super()._difference_lenient(other)
        result.append(value)

        return result

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)


class CoordMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.coords.Coord`.

    """

    _members = ("coord_system", "climatological")

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members for coordinates.

        Args:

        * other (CoordMetadata):
            The other coordinate metadata participating in the lenient
            combination.

        Returns:
            A list of combined metadata member values.

        """
        # Perform "strict" combination for "coord_system" and "climatological".
        def func(field):
            value = getattr(self, field)
            return value if value == getattr(other, field) else None

        values = [func(field) for field in self._members]
        # Perform lenient combination of the other parent members.
        result = super()._combine_lenient(other)
        result.extend(values)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members for coordinates.

        Args:

        * other (CoordMetadata):
            The other coordinate metadata participating in the lenient
            comparison.

        Returns:
            Boolean.

        """
        # Perform "strict" comparison for "coord_system" and "climatological".
        result = (
            self.coord_system == other.coord_system
            and self.climatological == other.climatological
        )
        if result:
            # Perform lenient comparison of the other parent members.
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members for coordinates.

        Args:

        * other (CoordMetadata):
            The other coordinate metadata participating in the lenient
            difference.

        Returns:
            A list of difference metadata member values.

        """
        # Perform "strict" difference for "coord_system" and "climatological".
        def func(field):
            left = getattr(self, field)
            right = getattr(other, field)
            return None if left == right else (left, right)

        values = [func(field) for field in self._members]
        # Perform lenient difference of the other parent members.
        result = super()._difference_lenient(other)
        result.extend(values)

        return result

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)


class CubeMetadata(BaseMetadata):
    """
    Metadata container for a :class:`~iris.cube.Cube`.

    """

    _members = "cell_methods"

    __slots__ = ()

    @wraps(BaseMetadata.__eq__, assigned=("__doc__",), updated=())
    @lenient_service
    def __eq__(self, other):
        return super().__eq__(other)

    def _combine_lenient(self, other):
        """
        Perform lenient combination of metadata members for cubes.

        Args:

        * other (CubeMeatadata):
            The other cube metadata participating in the lenient combination.

        Returns:
            A list of combined metadata member values.

        """
        # Perform "strict" combination for "cell_methods".
        value = (
            self.cell_methods
            if self.cell_methods == other.cell_methods
            else None
        )
        # Perform lenient combination of the other parent members.
        result = super()._combine_lenient(other)
        result.append(value)

        return result

    def _compare_lenient(self, other):
        """
        Perform lenient equality of metadata members for cubes.

        Args:

        * other (CubeMetadata):
            The other cube metadata participating in the lenient comparison.

        Returns:
            Boolean.

        """
        # Perform "strict" comparison for "cell_methods".
        result = self.cell_methods == other.cell_methods
        if result:
            result = super()._compare_lenient(other)

        return result

    def _difference_lenient(self, other):
        """
        Perform lenient difference of metadata members for cubes.

        Args:

        * other (CubeMetadata):
            The other cube metadata participating in the lenient difference.

        Returns:
            A list of difference metadata member values.

        """
        # Perform "strict" difference for "cell_methods".
        value = (
            None
            if self.cell_methods == other.cell_methods
            else (self.cell_methods, other.cell_methods)
        )
        # Perform lenient difference of the other parent members.
        result = super()._difference_lenient(other)
        result.append(value)

        return result

    @property
    def _names(self):
        """
        A tuple containing the value of each name participating in the identity
        of a :class:`iris.cube.Cube`. This includes the standard name,
        long name, NetCDF variable name, and the STASH from the attributes
        dictionary.

        """
        standard_name = self.standard_name
        long_name = self.long_name
        var_name = self.var_name

        # Defensive enforcement of attributes being a dictionary.
        if not isinstance(self.attributes, Mapping):
            try:
                self.attributes = dict()
            except AttributeError:
                emsg = "Invalid '{}.attributes' member, must be a mapping."
                raise AttributeError(emsg.format(self.__class__.__name__))

        stash_name = self.attributes.get("STASH")
        if stash_name is not None:
            stash_name = str(stash_name)

        return standard_name, long_name, var_name, stash_name

    @wraps(BaseMetadata.combine, assigned=("__doc__",), updated=())
    @lenient_service
    def combine(self, other, lenient=None):
        return super().combine(other, lenient=lenient)

    @wraps(BaseMetadata.difference, assigned=("__doc__",), updated=())
    @lenient_service
    def difference(self, other, lenient=None):
        return super().difference(other, lenient=lenient)

    @wraps(BaseMetadata.equal, assigned=("__doc__",), updated=())
    @lenient_service
    def equal(self, other, lenient=None):
        return super().equal(other, lenient=lenient)

    @wraps(BaseMetadata.name)
    def name(self, default=None, token=False):
        def _check(item):
            return self.token(item) if token else item

        default = self.DEFAULT_NAME if default is None else default

        # Defensive enforcement of attributes being a dictionary.
        if not isinstance(self.attributes, Mapping):
            try:
                self.attributes = dict()
            except AttributeError:
                emsg = "Invalid '{}.attributes' member, must be a mapping."
                raise AttributeError(emsg.format(self.__class__.__name__))

        result = (
            _check(self.standard_name)
            or _check(self.long_name)
            or _check(self.var_name)
            or _check(str(self.attributes.get("STASH", "")))
            or _check(default)
        )

        if token and result is None:
            emsg = "Cannot retrieve a valid name token from {!r}"
            raise ValueError(emsg.format(self))

        return result


def metadata_manager_factory(cls, **kwargs):
    """
    A class instance factory function responsible for manufacturing
    metadata instances dynamically at runtime.

    The factory instances returned by the factory are capable of managing
    their metadata state, which can be proxied by the owning container.

    Args:

    * cls:
        A subclass of :class:`~iris.common.metadata.BaseMetadata`, defining
        the metadata to be managed.

    Kwargs:

    * kwargs:
        Initial values for the manufactured metadata instance. Unspecified
        fields will default to a value of 'None'.

    """

    def __init__(self, cls, **kwargs):
        # Restrict to only dealing with appropriate metadata classes.
        if not issubclass(cls, BaseMetadata):
            emsg = "Require a subclass of {!r}, got {!r}."
            raise TypeError(emsg.format(BaseMetadata.__name__, cls))

        #: The metadata class to be manufactured by this factory.
        self.cls = cls

        # Initialise the metadata class fields in the instance.
        for field in self.fields:
            setattr(self, field, None)

        # Populate with provided kwargs, which have already been verified
        # by the factory.
        for field, value in kwargs.items():
            setattr(self, field, value)

    def __eq__(self, other):
        if not hasattr(other, "cls"):
            return NotImplemented
        match = self.cls is other.cls
        if match:
            match = self.values == other.values

        return match

    def __getstate__(self):
        """Return the instance state to be pickled."""
        return {field: getattr(self, field) for field in self.fields}

    def __ne__(self, other):
        match = self.__eq__(other)
        if match is not NotImplemented:
            match = not match

        return match

    def __reduce__(self):
        """
        Dynamically created classes at runtime cannot be pickled, due to not
        being defined at the top level of a module. As a result, we require to
        use the __reduce__ interface to allow 'pickle' to recreate this class
        instance, and dump and load instance state successfully.

        """
        return metadata_manager_factory, (self.cls,), self.__getstate__()

    def __repr__(self):
        args = ", ".join(
            [
                "{}={!r}".format(field, getattr(self, field))
                for field in self.fields
            ]
        )
        return "{}({})".format(self.__class__.__name__, args)

    def __setstate__(self, state):
        """Set the instance state when unpickling."""
        for field, value in state.items():
            setattr(self, field, value)

    @property
    def fields(self):
        """Return the name of the metadata members."""
        # Proxy for built-in namedtuple._fields property.
        return self.cls._fields

    @property
    def values(self):
        fields = {field: getattr(self, field) for field in self.fields}
        return self.cls(**fields)

    # Restrict factory to appropriate metadata classes only.
    if not issubclass(cls, BaseMetadata):
        emsg = "Require a subclass of {!r}, got {!r}."
        raise TypeError(emsg.format(BaseMetadata.__name__, cls))

    # Check whether kwargs have valid fields for the specified metadata.
    if kwargs:
        extra = [field for field in kwargs.keys() if field not in cls._fields]
        if extra:
            bad = ", ".join(map(lambda field: "{!r}".format(field), extra))
            emsg = "Invalid {!r} field parameters, got {}."
            raise ValueError(emsg.format(cls.__name__, bad))

    # Define the name, (inheritance) bases and namespace of the dynamic class.
    name = "MetadataManager"
    bases = ()
    namespace = {
        "DEFAULT_NAME": cls.DEFAULT_NAME,
        "__init__": __init__,
        "__eq__": __eq__,
        "__getstate__": __getstate__,
        "__ne__": __ne__,
        "__reduce__": __reduce__,
        "__repr__": __repr__,
        "__setstate__": __setstate__,
        "fields": fields,
        "name": cls.name,
        "token": cls.token,
        "values": values,
    }

    # Account for additional "CubeMetadata" specialised class behaviour.
    if cls is CubeMetadata:
        namespace["_names"] = cls._names

    # Dynamically create the class.
    Metadata = type(name, bases, namespace)
    # Now manufacture an instance of that class.
    metadata = Metadata(cls, **kwargs)

    return metadata


#: Convenience collection of lenient metadata combine services.
SERVICES_COMBINE = (
    AncillaryVariableMetadata.combine,
    BaseMetadata.combine,
    CellMeasureMetadata.combine,
    CoordMetadata.combine,
    CubeMetadata.combine,
)


#: Convenience collection of lenient metadata difference services.
SERVICES_DIFFERENCE = (
    AncillaryVariableMetadata.difference,
    BaseMetadata.difference,
    CellMeasureMetadata.difference,
    CoordMetadata.difference,
    CubeMetadata.difference,
)


#: Convenience collection of lenient metadata equality services.
SERVICES_EQUAL = (
    AncillaryVariableMetadata.__eq__,
    AncillaryVariableMetadata.equal,
    BaseMetadata.__eq__,
    BaseMetadata.equal,
    CellMeasureMetadata.__eq__,
    CellMeasureMetadata.equal,
    CoordMetadata.__eq__,
    CoordMetadata.equal,
    CubeMetadata.__eq__,
    CubeMetadata.equal,
)


#: Convenience collection of lenient metadata services.
SERVICES = SERVICES_COMBINE + SERVICES_DIFFERENCE + SERVICES_EQUAL

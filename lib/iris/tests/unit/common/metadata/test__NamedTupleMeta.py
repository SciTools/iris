# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.metadata._NamedTupleMeta`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from abc import abstractmethod

from iris.common.metadata import _NamedTupleMeta


class Test(tests.IrisTest):
    @staticmethod
    def names(classes):
        return [cls.__name__ for cls in classes]

    @staticmethod
    def emsg_generate(members):
        if isinstance(members, str):
            members = (members,)
        emsg = ".* missing {} required positional argument{}: {}"
        args = ", ".join([f"{member!r}" for member in members[:-1]])
        count = len(members)
        if count == 1:
            args += f"{members[-1]!r}"
        elif count == 2:
            args += f" and {members[-1]!r}"
        else:
            args += f", and {members[-1]!r}"
        plural = "s" if count > 1 else ""
        return emsg.format(len(members), plural, args)

    def test__no_bases_with_abstract_members_property(self):
        class Metadata(metaclass=_NamedTupleMeta):
            @property
            @abstractmethod
            def _members(self):
                pass

        expected = ["object"]
        self.assertEqual(self.names(Metadata.__bases__), expected)
        expected = ["Metadata", "object"]
        self.assertEqual(self.names(Metadata.__mro__), expected)
        emsg = "Can't instantiate abstract class .* with abstract method.* _members"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Metadata()

    def test__no_bases_single_member(self):
        member = "arg_one"

        class Metadata(metaclass=_NamedTupleMeta):
            _members = member

        expected = ["MetadataNamedtuple"]
        self.assertEqual(self.names(Metadata.__bases__), expected)
        expected = ["Metadata", "MetadataNamedtuple", "tuple", "object"]
        self.assertEqual(self.names(Metadata.__mro__), expected)
        emsg = self.emsg_generate(member)
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Metadata()
        metadata = Metadata(1)
        self.assertEqual(metadata._fields, (member,))
        self.assertEqual(metadata.arg_one, 1)

    def test__no_bases_multiple_members(self):
        members = ("arg_one", "arg_two")

        class Metadata(metaclass=_NamedTupleMeta):
            _members = members

        expected = ["MetadataNamedtuple"]
        self.assertEqual(self.names(Metadata.__bases__), expected)
        expected = ["Metadata", "MetadataNamedtuple", "tuple", "object"]
        self.assertEqual(self.names(Metadata.__mro__), expected)
        emsg = self.emsg_generate(members)
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Metadata()
        values = range(len(members))
        metadata = Metadata(*values)
        self.assertEqual(metadata._fields, members)
        expected = dict(zip(members, values))
        self.assertEqual(metadata._asdict(), expected)

    def test__multiple_bases_multiple_members(self):
        members_parent = ("arg_one", "arg_two")
        members_child = ("arg_three", "arg_four")

        class MetadataParent(metaclass=_NamedTupleMeta):
            _members = members_parent

        class MetadataChild(MetadataParent):
            _members = members_child

        # Check the parent class...
        expected = ["MetadataParentNamedtuple"]
        self.assertEqual(self.names(MetadataParent.__bases__), expected)
        expected = [
            "MetadataParent",
            "MetadataParentNamedtuple",
            "tuple",
            "object",
        ]
        self.assertEqual(self.names(MetadataParent.__mro__), expected)
        emsg = self.emsg_generate(members_parent)
        with self.assertRaisesRegex(TypeError, emsg):
            _ = MetadataParent()
        values_parent = range(len(members_parent))
        metadata_parent = MetadataParent(*values_parent)
        self.assertEqual(metadata_parent._fields, members_parent)
        expected = dict(zip(members_parent, values_parent))
        self.assertEqual(metadata_parent._asdict(), expected)

        # Check the dependent child class...
        expected = ["MetadataChildNamedtuple", "MetadataParent"]
        self.assertEqual(self.names(MetadataChild.__bases__), expected)
        expected = [
            "MetadataChild",
            "MetadataChildNamedtuple",
            "MetadataParent",
            "MetadataParentNamedtuple",
            "tuple",
            "object",
        ]
        self.assertEqual(self.names(MetadataChild.__mro__), expected)
        emsg = self.emsg_generate((*members_parent, *members_child))
        with self.assertRaisesRegex(TypeError, emsg):
            _ = MetadataChild()
        fields_child = (*members_parent, *members_child)
        values_child = range(len(fields_child))
        metadata_child = MetadataChild(*values_child)
        self.assertEqual(metadata_child._fields, fields_child)
        expected = dict(zip(fields_child, values_child))
        self.assertEqual(metadata_child._asdict(), expected)


if __name__ == "__main__":
    tests.main()

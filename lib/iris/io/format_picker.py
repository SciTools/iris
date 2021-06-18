# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A module to provide convenient file format identification through a combination of filename extension
and file based *magic* numbers.


To manage a collection of FormatSpecifications for loading::

    import iris.io.format_picker as fp
    import matplotlib.pyplot as plt
    fagent = fp.FormatAgent()
    png_spec = fp.FormatSpecification('PNG image', fp.MagicNumber(8),
                                      0x89504E470D0A1A0A,
                                      handler=lambda filename: plt.imread(filename),
                                      priority=5
                                      )
    fagent.add_spec(png_spec)

To identify a specific format from a file::

    with open(png_filename, 'rb') as png_fh:
        handling_spec = fagent.get_spec(png_filename, png_fh)

In the example, handling_spec will now be the png_spec previously added to the agent.

Now that a specification has been found, if a handler has been given with the specification, then the file can be handled::

    handler = handling_spec.handler
    if handler is None:
       raise ValueError('File cannot be handled.')
    else:
       result = handler(filename)

The calling sequence of handler is dependent on the function given in the original specification and can be customised to your project's needs.


"""

from collections.abc import Callable
import functools
import os
import struct


class FormatAgent:
    """
    The FormatAgent class is the containing object which is responsible for identifying the format of a given file
    by interrogating its children FormatSpecification instances.

    Typically a FormatAgent will be created empty and then extended with the :meth:`FormatAgent.add_spec` method::

        agent = FormatAgent()
        agent.add_spec(NetCDF_specification)

    Less commonly, this can also be written::

        agent = FormatAgent(NetCDF_specification)

    """

    def __init__(self, format_specs=None):
        """ """
        self._format_specs = list(format_specs or [])
        self._format_specs.sort()

    def add_spec(self, format_spec):
        """Add a FormatSpecification instance to this agent for format consideration."""
        self._format_specs.append(format_spec)
        self._format_specs.sort()

    def __repr__(self):
        return "FormatAgent(%r)" % self._format_specs

    def __str__(self):
        prefix = " * " if len(self._format_specs) > 1 else ""
        return prefix + "\n * ".join(
            ["%s" % format_spec for format_spec in self._format_specs]
        )

    def get_spec(self, basename, buffer_obj):
        """
        Pick the first FormatSpecification which can handle the given
        filename and file/buffer object.

        .. note::

            ``buffer_obj`` may be ``None`` when a seekable file handle is not
            feasible (such as over the http protocol). In these cases only the
            format specifications which do not require a file handle are
            tested.

        """
        element_cache = {}
        for format_spec in self._format_specs:
            # For the case where a buffer_obj is None (such as for the
            # http protocol) skip any specs which require a fh - they
            # don't match.
            if buffer_obj is None and format_spec.file_element.requires_fh:
                continue

            fmt_elem = format_spec.file_element
            fmt_elem_value = format_spec.file_element_value

            # cache the results for each file element
            if repr(fmt_elem) not in element_cache:
                # N.B. File oriented as this is assuming seekable stream.
                if buffer_obj is not None and buffer_obj.tell() != 0:
                    # reset the buffer if tell != 0
                    buffer_obj.seek(0)

                element_cache[repr(fmt_elem)] = fmt_elem.get_element(
                    basename, buffer_obj
                )

            # If we have a callable object, then call it and tests its result, otherwise test using basic equality
            if isinstance(fmt_elem_value, Callable):
                matches = fmt_elem_value(element_cache[repr(fmt_elem)])
            elif element_cache[repr(fmt_elem)] == fmt_elem_value:
                matches = True
            else:
                matches = False

            if matches:
                return format_spec

        printable_values = {}
        for key, value in element_cache.items():
            value = str(value)
            if len(value) > 50:
                value = value[:50] + "..."
            printable_values[key] = value
        msg = (
            "No format specification could be found for the given buffer."
            " File element cache:\n {}".format(printable_values)
        )
        raise ValueError(msg)


@functools.total_ordering
class FormatSpecification:
    """
    Provides the base class for file type definition.

    Every FormatSpecification instance has a name which can be accessed with the :attr:`FormatSpecification.name` property and
    a FileElement, such as filename extension or 32-bit magic number, with an associated value for format identification.

    """

    def __init__(
        self,
        format_name,
        file_element,
        file_element_value,
        handler=None,
        priority=0,
        constraint_aware_handler=False,
    ):
        """
        Constructs a new FormatSpecification given the format_name and particular FileElements

        Args:

        * format_name - string name of fileformat being described
        * file_element - FileElement instance of the element which identifies this FormatSpecification
        * file_element_value - The value that the file_element should take if a file matches this FormatSpecification

        Kwargs:

        * handler - function which will be called when the specification has been identified and is required to handler a format.
                            If None, then the file can still be identified but no handling can be done.
        * priority - Integer giving a priority for considering this specification where higher priority means sooner consideration.

        """
        if not isinstance(file_element, FileElement):
            raise ValueError(
                "file_element must be an instance of FileElement, got %r"
                % file_element
            )

        self._file_element = file_element
        self._file_element_value = file_element_value
        self._format_name = format_name
        self._handler = handler
        self.priority = priority
        self.constraint_aware_handler = constraint_aware_handler

    def __hash__(self):
        # Hashed by specification for consistent ordering in FormatAgent (including self._handler in this hash
        # for example would order randomly according to object id)
        return hash(self._file_element)

    @property
    def file_element(self):
        return self._file_element

    @property
    def file_element_value(self):
        return self._file_element_value

    @property
    def name(self):
        """The name of this FileFormat. (Read only)"""
        return self._format_name

    @property
    def handler(self):
        """The handler function of this FileFormat. (Read only)"""
        return self._handler

    def _sort_key(self):
        return (-self.priority, self.name, self.file_element)

    def __lt__(self, other):
        if not isinstance(other, FormatSpecification):
            return NotImplemented

        return self._sort_key() < other._sort_key()

    def __eq__(self, other):
        if not isinstance(other, FormatSpecification):
            return NotImplemented

        return self._sort_key() == other._sort_key()

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        # N.B. loader is not always going to provide a nice repr if it is a lambda function, hence a prettier version is available in __str__
        return "FormatSpecification(%r, %r, %r, handler=%r, priority=%s)" % (
            self._format_name,
            self._file_element,
            self._file_element_value,
            self.handler,
            self.priority,
        )

    def __str__(self):
        return "%s%s (priority %s)" % (
            self.name,
            " (no handler available)" if self.handler is None else "",
            self.priority,
        )


class FileElement:
    """
    Represents a specific aspect of a FileFormat which can be identified using the given element getter function.

    """

    def __init__(self, requires_fh=True):
        """
        Constructs a new file element, which may require a file buffer.

        Kwargs:

        * requires_fh - Whether this FileElement needs a file buffer.

        """
        self.requires_fh = requires_fh

    def get_element(self, basename, file_handle):
        """Called when identifying the element of a file that this FileElement is representing."""
        raise NotImplementedError("get_element must be defined in a subclass")

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class MagicNumber(FileElement):
    """A :class:`FileElement` that returns a byte sequence in the file."""

    len_formats = {4: ">L", 8: ">Q"}

    def __init__(self, num_bytes, offset=None):
        FileElement.__init__(self)
        self._num_bytes = num_bytes
        self._offset = offset

    def get_element(self, basename, file_handle):
        if self._offset is not None:
            file_handle.seek(self._offset)
        bytes = file_handle.read(self._num_bytes)
        fmt = self.len_formats.get(self._num_bytes)
        if len(bytes) != self._num_bytes:
            raise EOFError(file_handle.name)
        if fmt is None:
            result = bytes
        else:
            result = struct.unpack(fmt, bytes)[0]
        return result

    def __repr__(self):
        return "MagicNumber({}, {})".format(self._num_bytes, self._offset)


class FileExtension(FileElement):
    """A :class:`FileElement` that returns the extension from the filename."""

    def get_element(self, basename, file_handle):
        return os.path.splitext(basename)[1]


class LeadingLine(FileElement):
    """A :class:`FileElement` that returns the first line from the file."""

    def get_element(self, basename, file_handle):
        return file_handle.readline()


class UriProtocol(FileElement):
    """
    A :class:`FileElement` that returns the "scheme" and "part" from a URI,
    using :func:`~iris.io.decode_uri`.

    """

    def __init__(self):
        FileElement.__init__(self, requires_fh=False)

    def get_element(self, basename, file_handle):
        from iris.io import decode_uri

        return decode_uri(basename)[0]

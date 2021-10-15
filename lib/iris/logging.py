# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A package for provisioning logging infra-structure.

"""

import logging
import threading
from typing import Optional, Union

__all__ = ["DATEFMT", "FMT", "IrisFormatter", "get_logger"]


#: The default ``datefmt`` string of the logger formatter.
DATEFMT: str = "%d-%m-%Y %H:%M:%S"

#: The default ``fmt`` string of the logger formatter.
FMT: str = "%(asctime)s %(name)s %(levelname)s - %(message)s"

# Critical region lock to protect against race conditions
# when adding a handler to a logger.
_LOCK = threading.Lock()


class IrisFormatter(logging.Formatter):
    """
    .. versionadded:: 3.2.0

    A :class:`logging.Formatter` that always appends the class name,
    if available, and the function name to a logging message.

    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        fmt : str, optional
            The format string of the :class:`logging.Formatter`.
            If ``None``, defaults to :data:`FMT`.
        datefmt : str, optional
            The date format string of the :class:`logging.Formatter`.
            If ``None``, defaults to :data:`DATEFMT`.

        """
        if fmt is None:
            fmt = FMT
        if datefmt is None:
            datefmt = DATEFMT
        super().__init__(fmt=fmt, datefmt=datefmt, style="%")

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the provided record into a logging message.

        Parameters
        ----------
        record : LogRecord
            The :class:`logging.LogRecord` that requires to be formatted.

        Returns
        -------
        str
            The formatted message of the log record.

        """
        result = super().format(record)
        if "cls" in record.__dict__:
            extra = "[caller:%(cls)s.%(funcName)s]"
        else:
            extra = "[caller:%(funcName)s]"
        result = f"{result} {extra % record.__dict__}"
        return result


def get_logger(
    name: str, level: Optional[Union[int, str]] = None
) -> logging.Logger:
    """
    .. versionadded:: 3.2.0

    Create and configure a :class:`logging.Logger`.

    Child loggers will simply propagate their messages to the singleton root
    logger in the logging hierarchy, or the first parent logger with a handler
    configured. Typically, this will be the top-level ``iris`` logger.

    The root logger, if specified by ``name``, will be configured with a
    :class:`logging.StreamHandler` and a custom :class:`Formatter`, as will the
    top-level ``iris`` logger. No other loggers will be configured with a
    handler.

    Parameters
    ----------
    name : str
        The name of the logger. Typically this is the module filename
        (``__name__``) that owns the logger. Note that, the singleton root
        logger is selected with a ``name`` of ``None`` or ``root``.
    level : int or str, optional
        The threshold level of the logger. If ``None``, defaults to ``WARNING``
        for the ``root`` logger, ``NOTSET`` for the top-level logger, ``INFO``
        otherwise.

    Returns
    -------
    logging.Logger
        A configured :class:`logging.Logger`.

    """
    # Determine if this is the root logger.
    root = name is None or name == "root"

    # This is a convenience, and makes the use case more obvious.
    if name == "root":
        name = None

    # Determine if this is the top-level logger.
    top = name == __package__

    if level is None:
        level = "WARNING" if root else "NOTSET" if top else "INFO"

    # Create the named logger. If it already exists, logging
    # takes care logger management, and will return the existing
    # logger.
    logger = logging.getLogger(name)

    # Set the logger level, which is different to the effective level.
    logger.setLevel(level)

    if not root:
        # Children propagate to the top-level logger.
        logger.propagate = not top

    # Create and add the handler, if required.
    if root or top:
        with _LOCK:
            # Give the handler a specific name.
            handler_name = f"{'root' if root else __package__}_handler"
            logger_handler_names = [
                handler.get_name() for handler in logger.handlers
            ]

            # Ensure that we only ever add our handler to the logger once.
            if handler_name not in logger_handler_names:
                # Create a logging handler.
                handler = logging.StreamHandler()
                # Set the handler name.
                handler.set_name(handler_name)
                # Set the handler custom formatter.
                handler.setFormatter(IrisFormatter())
                # Add the handler to the logger.
                logger.addHandler(handler)

    return logger

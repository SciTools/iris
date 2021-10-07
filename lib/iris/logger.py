# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Infrastructure to support logging.

"""

import logging
from typing import Optional, Union

__all__ = ["IrisFormatter", "get_logger"]


#: The ``datefmt`` string of the root logger formatter.
ROOT_DATEFMT: str = "%d-%m-%Y %H:%M:%S"

#: The ``fmt`` string of the root logger formatter.
ROOT_FMT: str = "%(asctime)s %(name)s %(levelname)s - %(message)s"


class IrisFormatter(logging.Formatter):
    """
    .. versionadded:: 3.2.0

    A :class:`logging.Formatter` that always appends the class name,
    if available, and the function name to a logging message.

    """

    def __init__(
        self, fmt: Optional[str] = None, datefmt: Optional[str] = None
    ):
        if fmt is None:
            fmt = ROOT_FMT
        if datefmt is None:
            datefmt = ROOT_DATEFMT
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
            extra = "[%(cls)s.%(funcName)s]"
        else:
            extra = "[%(funcName)s]"
        result = f"{result} {extra % record.__dict__}"
        return result


def get_logger(
    name: str, level: Optional[Union[int, str]] = None, root: bool = False
) -> logging.Logger:
    """
    .. versionadded:: 3.2.0

    Create a :class:`logging.Logger`, with a :class:`logging.StreamHandler`
    and a custom :class:`IrisFormatter` for root loggers.

    Parameters
    ----------
    name : str
        The name of the logger. Typically this is the module filename
        (``__name__``) that owns the logger.
    level : int or str, optional
        The threshold level of the logger. If the logger is a ``root`` logger,
        then defaults to ``NOTSET``, otherwise ``INFO``.
    root : bool, default=False
        Specify whether this is a root logger, or a child logger that will
        propagate its message to the first parent root logger.

    Returns
    -------
    logging.Logger
        A configured :class:`logging.Logger`.

    """
    if level is None:
        level = "NOTSET" if root else "INFO"

    # Create the named logger.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Children propagate to the first parent root logger.
    logger.propagate = not root

    # Create and add the handler to the root logger, if required.
    if root:
        # Create a logging handler.
        handler = logging.StreamHandler()
        handler.setFormatter(IrisFormatter())
        # Add the handler to the logger.
        logger.addHandler(handler)

    return logger

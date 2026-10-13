# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Native Zarr support, built on zarr-python.

.. z_reference:: iris.fileformats.zarr
   :tags: topic_load_save

   API reference

Reads Zarr version 2 and version 3 stores, local or remote. Not NCZarr: that
is netCDF-c's own Zarr driver, reached through a URL fragment such as
``file:///path/store.zarr#mode=nczarr,file``, and it stays where it is in
:mod:`iris.fileformats.netcdf`.

``zarr`` is an optional dependency. Nothing in :mod:`iris.fileformats` imports
this package until a Zarr target is actually loaded, so Iris installs and
imports without it.

This package shadows the third-party ``zarr`` distribution by name. Python 3
resolves ``import zarr`` absolutely, so inside these modules it still finds the
third-party package; xarray's ``xarray.backends.zarr`` has the same
arrangement.

See sections 4.1 and 4.4 of
``docs/superpowers/specs/2026-09-21-zarr-io-design.md``.

"""

import importlib.metadata

from packaging.version import Version

__all__ = ["load_cubes"]

#: The version floor ``docs/src/user_manual/explanation/zarr_io.rst`` and
#: ``requirements/py3*.yml`` both promise.
_MINIMUM_VERSION = "3.0.8"

#: Raised whether ``zarr`` is missing entirely or present below
#: :data:`_MINIMUM_VERSION`. Defined once here, not in
#: :func:`iris.fileformats._load_zarr`, so that the two routes a caller can
#: reach a Zarr load through - ``iris.load`` and calling
#: :func:`load_cubes` directly - produce the same text rather than each
#: carrying its own copy.
_MISSING_MESSAGE = (
    "Unable to load a Zarr store - the 'zarr' package is not "
    "installed. Install zarr >=3.0.8, for example with "
    "'conda install -c conda-forge zarr'."
)


def _check_zarr_version() -> None:
    """Raise a clear ``ImportError`` if ``zarr`` is installed below the floor.

    Reads installed-distribution metadata rather than importing ``zarr``
    itself: a zarr-python 2.x ``Group`` has no ``.metadata``, so importing it
    only to compare `__version__` would load far enough to reach
    :class:`~iris.fileformats.zarr._dataset.ZarrDataset` and fail there
    instead, with a bare ``AttributeError``.

    Does nothing if ``zarr`` is not installed at all - :func:`__getattr__`
    below reports that case itself, from the import it goes on to attempt.

    Raises
    ------
    ImportError
        If ``zarr`` is installed below :data:`_MINIMUM_VERSION`.

    """
    try:
        installed = importlib.metadata.version("zarr")
    except importlib.metadata.PackageNotFoundError:
        return

    if Version(installed) < Version(_MINIMUM_VERSION):
        raise ImportError(_MISSING_MESSAGE)


def __getattr__(name):
    """Forward ``load_cubes`` to :mod:`iris.fileformats.zarr.loader`.

    PEP 562, not a plain ``from .loader import load_cubes``: ``loader``
    imports ``._dataset``, which imports ``zarr`` at module level, so an
    eager import here would make the dependency load with the package rather
    than with a Zarr load - breaking the promise this module's own docstring
    makes, that nothing imports ``zarr`` until a Zarr target is actually
    loaded. Resist "simplifying" this back to a plain import.

    Parameters
    ----------
    name : str
        The attribute being looked up. Python calls this only for names that
        are not already module attributes.

    Returns
    -------
    Any
        ``load_cubes``, the only name this forwards.

    Raises
    ------
    AttributeError
        For any other name.
    ImportError
        If ``zarr`` is not installed, or is installed below
        :data:`_MINIMUM_VERSION`.

    """
    if name != "load_cubes":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _check_zarr_version()

    try:
        from .loader import load_cubes
    except ImportError as exception:
        raise ImportError(_MISSING_MESSAGE) from exception

    return load_cubes

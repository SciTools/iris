# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Native Zarr support, built on zarr-python.

.. z_reference:: iris.fileformats.zarr
   :tags: topic_load_save

   API reference

Reads Zarr version 2 and version 3 stores, local or remote, and writes version
3 ones. Not NCZarr: that is netCDF-c's own Zarr driver, reached through a URL
fragment such as ``file:///path/store.zarr#mode=nczarr,file``, and it stays
where it is in :mod:`iris.fileformats.netcdf`.

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

import importlib
import importlib.metadata

from packaging.version import Version

__all__ = ["Saver", "load_cubes", "save"]

#: What `__getattr__` forwards, and the module it lives in. The import is
#: deferred because `._dataset` imports `zarr` at module level, and this
#: package promises that nothing does so until a Zarr target is used.
_FORWARDED = {
    "load_cubes": ".loader",
    "save": ".saver",
    "Saver": ".saver",
}

#: The version floor ``docs/src/user_manual/explanation/zarr_io.rst`` and
#: ``requirements/py3*.yml`` both promise.
_MINIMUM_VERSION = "3.0.8"

#: Raised whether ``zarr`` is missing entirely or present below
#: :data:`_MINIMUM_VERSION`. Defined once here, not in
#: :func:`iris.fileformats._load_zarr`, so that every route a caller can
#: reach a Zarr store through - ``iris.load``, ``iris.save``, and calling
#: :func:`load_cubes` directly - produces the same text rather than each
#: carrying its own copy. "Use", not "load": ``iris.save`` raises this too.
_MISSING_MESSAGE = (
    "Unable to use a Zarr store - the 'zarr' package is not "
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
    """Forward the public names to the modules that define them.

    PEP 562, not a plain ``from .loader import load_cubes``: ``loader`` and
    ``saver`` both import ``._dataset``, which imports ``zarr`` at module
    level, so an eager import here would make the dependency load with the
    package rather than with a Zarr load or save - breaking the promise this
    module's own docstring makes, that nothing imports ``zarr`` until a Zarr
    target is actually used. Resist "simplifying" this back to a plain
    import.

    Parameters
    ----------
    name : str
        The attribute being looked up. Python calls this only for names that
        are not already module attributes.

    Returns
    -------
    Any
        The object of that name, from the module :data:`_FORWARDED` names.

    Raises
    ------
    AttributeError
        For any other name.
    ImportError
        If ``zarr`` is not installed, or is installed below
        :data:`_MINIMUM_VERSION`.

    """
    module_name = _FORWARDED.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _check_zarr_version()

    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as exception:
        raise ImportError(_MISSING_MESSAGE) from exception

    return getattr(module, name)

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

__all__ = ["load_cubes"]


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

    """
    if name != "load_cubes":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .loader import load_cubes

    return load_cubes

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

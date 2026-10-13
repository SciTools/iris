# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A package for converting cubes to and from specific file formats.

.. z_reference:: iris.fileformats
   :tags: topic_load_save

   API reference
"""

from pathlib import Path

from iris.io import REMOTE_STORE_SCHEMES, _is_nczarr_fragment, decode_uri
from iris.io.format_picker import (
    DataSourceObjectProtocol,
    FileExtension,
    FormatAgent,
    FormatSpecification,
    LeadingLine,
    MagicNumber,
    UriProtocol,
    ZarrStore,
)

from . import name, netcdf, nimrod, pp, um

__all__ = ["FORMAT_AGENT"]


FORMAT_AGENT = FormatAgent()
FORMAT_AGENT.__doc__ = (
    "The FORMAT_AGENT is responsible for identifying the "
    "format of a given URI. New formats can be added "
    "with the **add_spec** method."
)


def _uri_is_http_not_nczarr(uri_parts):
    scheme, _part, fragment = uri_parts
    return scheme in ["http", "https"] and not _is_nczarr_fragment(fragment)


def _uri_is_nczarr(uri_parts):
    _scheme, _part, fragment = uri_parts
    return _is_nczarr_fragment(fragment)


#: What a version 3 and a version 2 Zarr store put at the root of a store.
_ZARR_MARKERS = ("zarr.json", ".zgroup")


def _is_zarr_store(target):
    """Whether `target` names a native Zarr store.

    Locally the test is a marker file, which costs one stat and is what tells
    a real store from a directory someone named ``.zarr``. Remotely it is the
    scheme and the suffix, because reading a marker file would cost a network
    round trip before Iris knows it wants the target at all; a remote
    ``.zarr`` that turns out not to be a store fails when zarr-python opens
    it, with zarr-python's message.

    These are two tests for two kinds of target, not two chances for one. A
    third rule matching any path ending in ``.zarr`` would subsume the remote
    one and leave the scheme test unreachable.
    """
    if not isinstance(target, str):
        # iris.io.load_data_objects passes open datasets, not paths.
        return False

    scheme, part, fragment = decode_uri(target)
    if _is_nczarr_fragment(fragment):
        # The user asked for netCDF-c's Zarr driver by name. The store on
        # disk is a real one, so the local test below would otherwise claim it.
        return False

    if scheme in REMOTE_STORE_SCHEMES:
        return part.rstrip("/").endswith(".zarr")

    directory = Path(part)
    return any((directory / marker).is_file() for marker in _ZARR_MARKERS)


def _load_zarr(*args, **kwargs):
    """Load from Zarr stores, reporting clearly when zarr-python is absent."""
    try:
        from iris.fileformats.zarr import load_cubes
    except ImportError as exception:
        message = (
            "Unable to load a Zarr store - the 'zarr' package is not "
            "installed. Install zarr >=3.0.8, for example with "
            "'conda install -c conda-forge zarr'."
        )
        raise ImportError(message) from exception

    return load_cubes(*args, **kwargs)


#
# PP files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Post Processing file (PP)",
        MagicNumber(4),
        0x00000100,
        pp.load_cubes,
        priority=5,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Post Processing file (PP) little-endian",
        MagicNumber(4),
        0x00010000,
        pp.load_cubes_little_endian,
        priority=3,
        constraint_aware_handler=True,
    )
)


#
# GRIB files.
#
def _load_grib(*args, **kwargs):
    try:
        from iris_grib import load_cubes
    except ImportError:
        raise RuntimeError(
            'Unable to load GRIB file - "iris_grib" package is not installed.'
        )

    return load_cubes(*args, **kwargs)


# NB. Because this is such a "fuzzy" check, we give this a very low
# priority to avoid collateral damage from false positives.
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "GRIB",
        MagicNumber(100),
        lambda header_bytes: b"GRIB" in header_bytes,
        _load_grib,
        priority=1,
    )
)


#
# netCDF files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NetCDF",
        MagicNumber(4),
        0x43444601,
        netcdf.load_cubes,
        priority=5,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NetCDF 64 bit offset format",
        MagicNumber(4),
        0x43444602,
        netcdf.load_cubes,
        priority=5,
        constraint_aware_handler=True,
    )
)


# This covers both v4 and v4 classic model.
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NetCDF_v4",
        MagicNumber(8),
        0x894844460D0A1A0A,
        netcdf.load_cubes,
        priority=5,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NetCDF OPeNDAP",
        UriProtocol(),
        _uri_is_http_not_nczarr,
        netcdf.load_cubes,
        priority=6,
        constraint_aware_handler=True,
    )
)

# NetCDF file presented as an open, readable netCDF4 dataset (or mimic).
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NetCDF dataset",
        DataSourceObjectProtocol(),
        lambda object: all(
            hasattr(object, x) for x in ("variables", "dimensions", "groups", "ncattrs")
        ),
        # Note: this uses the same call as the above "NetCDF_v4" (and "NetCDF OPeNDAP")
        # The handler itself needs to detect what is passed + handle it appropriately.
        netcdf.load_cubes,
        priority=4,
        constraint_aware_handler=True,
    )
)


#
# Zarr files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "Zarr",
        ZarrStore(),
        _is_zarr_store,
        _load_zarr,
        priority=5,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NcZarr",
        UriProtocol(),
        _uri_is_nczarr,
        netcdf.load_cubes,
        priority=3,
    )
)


#
# UM Fieldsfiles.
#
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Fieldsfile (FF) pre v3.1",
        MagicNumber(8),
        0x000000000000000F,
        um.load_cubes,
        priority=3,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Fieldsfile (FF) post v5.2",
        MagicNumber(8),
        0x0000000000000014,
        um.load_cubes,
        priority=4,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Fieldsfile (FF) ancillary",
        MagicNumber(8),
        0xFFFFFFFFFFFF8000,
        um.load_cubes,
        priority=3,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Fieldsfile (FF) converted with ieee to 32 bit",
        MagicNumber(4),
        0x00000014,
        um.load_cubes_32bit_ieee,
        priority=3,
        constraint_aware_handler=True,
    )
)


FORMAT_AGENT.add_spec(
    FormatSpecification(
        "UM Fieldsfile (FF) ancillary converted with ieee to 32 bit",
        MagicNumber(4),
        0xFFFF8000,
        um.load_cubes_32bit_ieee,
        priority=3,
        constraint_aware_handler=True,
    )
)


#
# NIMROD files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NIMROD", MagicNumber(4), 0x00000200, nimrod.load_cubes, priority=3
    )
)

#
# NAME files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification(
        "NAME III",
        LeadingLine(),
        lambda line: line.lstrip().startswith(b"NAME III"),
        name.load_cubes,
        priority=5,
    )
)


#
# ABF/ABL
# TODO: now deprecated, remove later
#
def load_cubes_abf_abl(*args, **kwargs):
    from . import abf

    return abf.load_cubes(*args, **kwargs)


FORMAT_AGENT.add_spec(
    FormatSpecification("ABF", FileExtension(), ".abf", load_cubes_abf_abl, priority=3)
)


FORMAT_AGENT.add_spec(
    FormatSpecification("ABL", FileExtension(), ".abl", load_cubes_abf_abl, priority=3)
)

# (C) British Crown Copyright 2010 - 2012, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
A package for converting cubes to and from specific file formats.

"""

import iris.io.format_picker as fp
from iris.io.format_picker import FormatSpecification as FormatSpec


__all__ = ['FORMAT_AGENT']


FORMAT_AGENT = fp.FormatAgent()
FORMAT_AGENT.__doc__ = "The FORMAT_AGENT is responsible for identifying the " \
                       "format of a given URI. New formats can be added " \
                       "with the **add_spec** method."


#
# PP files.
#
def _pp(filenames, callback=None):
    import iris.fileformats.pp
    return iris.fileformats.pp.load_cubes(filenames, callback=callback)


FORMAT_AGENT.add_spec(FormatSpec('UM Post Processing file (PP)',
                                 fp.MAGIC_NUMBER_32_BIT,
                                 0x00000100,
                                 _pp,
                                 priority=5))


def _pp_little_endian(filename, *args, **kwargs):
    msg = 'PP file {!r} contains little-endian data, ' \
          'please convert to big-endian with command line utility "bigend".'
    raise ValueError(msg.format(filename))


FORMAT_AGENT.add_spec(FormatSpec('UM Post Processing file (PP) little-endian',
                                 fp.MAGIC_NUMBER_32_BIT,
                                 0x00010000,
                                 _pp_little_endian,
                                 priority=5))


#
# GRIB files.
#
def _grib(filenames, callback=None):
    import iris.fileformats.grib
    return iris.fileformats.grib.load_cubes(filenames, callback=callback)


FORMAT_AGENT.add_spec(FormatSpec('GRIB',
                                 fp.MAGIC_NUMBER_32_BIT,
                                 0x47524942,
                                 _grib,
                                 priority=5))


#
# netCDF files.
#
def _netcdf(filenames, callback=None):
    import iris.fileformats.netcdf
    return iris.fileformats.netcdf.load_cubes(filenames, callback=callback)


FORMAT_AGENT.add_spec(FormatSpec('NetCDF',
                                 fp.MAGIC_NUMBER_32_BIT,
                                 0x43444601,
                                 _netcdf,
                                 priority=5))


FORMAT_AGENT.add_spec(FormatSpec('NetCDF 64 bit offset format',
                                 fp.MAGIC_NUMBER_32_BIT,
                                 0x43444602,
                                 _netcdf,
                                 priority=5))
    

# This covers both v4 and v4 classic model.
FORMAT_AGENT.add_spec(FormatSpec('NetCDF_v4',
                                 fp.MAGIC_NUMBER_64_BIT,
                                 0x894844460D0A1A0A,
                                 _netcdf,
                                 priority=5))


#
# UM Fieldsfiles.
#
def _ff(filenames, callback=None):
    import iris.fileformats.ff
    return iris.fileformats.ff.load_cubes(filenames, callback=callback)


FORMAT_AGENT.add_spec(FormatSpec('UM Fieldsfile (FF) pre v3.1',
                                 fp.MAGIC_NUMBER_64_BIT,
                                 0x000000000000000F,
                                 _ff,
                                 priority=4))


FORMAT_AGENT.add_spec(FormatSpec('UM Fieldsfile (FF) post v5.2',
                                 fp.MAGIC_NUMBER_64_BIT,
                                 0x0000000000000014,
                                 _ff,
                                 priority=4))


FORMAT_AGENT.add_spec(FormatSpec('UM Fieldsfile (FF) ancillary',
                                 fp.MAGIC_NUMBER_64_BIT,
                                 0xFFFFFFFFFFFF8000,
                                 _ff,
                                 priority=4))

#
# NIMROD files.
#
def _nimrod(filenames, callback=None):
    import iris.fileformats.nimrod
    return iris.fileformats.nimrod.load_cubes(filenames, callback=callback)


FORMAT_AGENT.add_spec(FormatSpec('NIMROD',
                                 fp.MAGIC_NUMBER_32_BIT,
                                 0x00000200,
                                 _nimrod,
                                 priority=5))

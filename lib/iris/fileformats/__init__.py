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
import pp
import ff
import grib
import netcdf


__all__ = ['FORMAT_AGENT']


def _pp_little_endian(filename, *args, **kwargs):
    raise ValueError('PP file %r contains little-endian data, please convert to big-endian with command line utility "bigend".' % filename)


FORMAT_AGENT = fp.FormatAgent()
FORMAT_AGENT.__doc__ = "The FORMAT_AGENT is responsible for identifying the format of a given URI. New " \
                       "formats can be added with the **add_spec** method."

_PP_spec = fp.FormatSpecification('UM Post Processing file (PP)', 
                                  fp.MAGIC_NUMBER_32_BIT, 0x00000100,
                                  pp.load_cubes,
                                  priority=5,
                                  )
FORMAT_AGENT.add_spec(_PP_spec)

_PP_little_endian_spec = fp.FormatSpecification('UM Post Processing file (PP) little-endian', 
                                  fp.MAGIC_NUMBER_32_BIT, 0x00010000,
                                  _pp_little_endian,
                                  priority=5,
                                  )

FORMAT_AGENT.add_spec(_PP_little_endian_spec)

_GRIB_spec = fp.FormatSpecification('GRIB', 
                                    fp.MAGIC_NUMBER_32_BIT, 0x47524942,
                                    grib.load_cubes,
                                    priority=5,
                                    )
FORMAT_AGENT.add_spec(_GRIB_spec)


_NetCDF_spec = fp.FormatSpecification('NetCDF', 
                                      fp.MAGIC_NUMBER_32_BIT, 0x43444601,
                                      netcdf.load_cubes,
                                      priority=5,
                                      )
FORMAT_AGENT.add_spec(_NetCDF_spec)


_NetCDF_64_bit_offset_spec = fp.FormatSpecification('NetCDF 64 bit offset format', 
                                      fp.MAGIC_NUMBER_32_BIT, 0x43444602,
                                      netcdf.load_cubes,
                                      priority=5,
                                      )
FORMAT_AGENT.add_spec(_NetCDF_64_bit_offset_spec)
    
# NetCDF4 - recognise a different magicnum, but treat just the same
# NOTE: this covers both v4 and v4 "classic model", the signature is the same
_NetCDF_v4_spec = fp.FormatSpecification('NetCDF_v4', 
                                      fp.MAGIC_NUMBER_64_BIT, 0x894844460d0a1a0a,
                                      netcdf.load_cubes,
                                      priority=5,
                                      )
FORMAT_AGENT.add_spec(_NetCDF_v4_spec)
    
# Fields files    
_FF_3p1_spec = fp.FormatSpecification('UM Fields file (FF) pre v3.1', 
                                  fp.MAGIC_NUMBER_64_BIT, 0x0000000000000014,
                                  ff.load_cubes,
                                  priority=4
                                  )
FORMAT_AGENT.add_spec(_FF_3p1_spec)
    
    
_FF_5p2_spec = fp.FormatSpecification('UM Fields file (FF) post v5.2', 
                                  fp.MAGIC_NUMBER_64_BIT, 0x000000000000000F,
                                  ff.load_cubes,
                                  priority=4,                              
                                  )
FORMAT_AGENT.add_spec(_FF_5p2_spec)

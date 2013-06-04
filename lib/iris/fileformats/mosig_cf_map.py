# (C) British Crown Copyright 2010 - 2013, Met Office
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

# DO NOT EDIT: AUTO-GENERATED RULES

import collections


CF = collections.namedtuple('CF', 'name unit')


MOSIG_STASH_TO_CF = {
    'm01s00i001': CF(name='surface_air_pressure', unit='Pa'),
    'm01s00i002': CF(name='eastward_wind', unit=''),
    'm01s00i003': CF(name='northward_wind', unit=''),
    'm01s01i208': CF(name='toa_outgoing_shortwave_flux', unit='W m-2'),
    'm01s01i232': CF(name='tendency_of_air_temperature_due_to_shortwave_heating', unit=''),
    'm01s01i233': CF(name='tendency_of_air_temperature_due_to_shortwave_heating_assuming_clear_sky', unit=''),
    'm01s01i237': CF(name='net_downward_shortwave_flux_in_air', unit='W m-2'),
    'm01s01i238': CF(name='tropopause_upwelling_shortwave_flux', unit=''),
    'm01s01i242': CF(name='large_scale_cloud_liquid_water_content_of_atmosphere_layer', unit=''),
    'm01s02i207': CF(name='surface_downwelling_longwave_flux_in_air', unit='W m-2'),
    'm01s02i208': CF(name='surface_downwelling_longwave_flux_in_air_assuming_clear_sky', unit='W m-2'),
    'm01s02i233': CF(name='tendency_of_air_temperature_due_to_longwave_heating_assuming_clear_sky', unit=''),
    'm01s02i237': CF(name='tropopause_net_downward_longwave_flux', unit=''),
    'm01s02i238': CF(name='tropopause_downwelling_longwave_flux', unit=''),
    'm01s03i202': CF(name='downward_heat_flux_in_soil', unit=''),
    'm01s03i219': CF(name='surface_downward_eastward_stress', unit='Pa'),
    'm01s03i220': CF(name='surface_downward_northward_stress', unit='Pa'),
    'm01s03i225': CF(name='eastward_wind', unit='m s-1'),
    'm01s03i226': CF(name='northward_wind', unit='m s-1'),
    'm01s03i234': CF(name='surface_upward_latent_heat_flux', unit='W m-2'),
    'm01s03i237': CF(name='specific_humidity', unit='1'),
    'm01s03i249': CF(name='wind_speed', unit='m s-1'),
    'm01s03i258': CF(name='surface_snow_melt_heat_flux', unit='W m-2'),
    'm01s03i261': CF(name='gross_primary_productivity_of_carbon', unit='kg m-2 s-1'),
    'm01s03i262': CF(name='net_primary_productivity_of_carbon', unit='kg m-2 s-1'),
    'm01s03i295': CF(name='surface_snow_area_fraction_where_land', unit='%'),
    'm01s03i298': CF(name='water_sublimation_flux', unit='kg m-2 s-1'),
    'm01s05i205': CF(name='convective_rainfall_rate', unit='kg m-2 s-1'),
    'm01s05i206': CF(name='convective_snowfall_flux', unit='kg m-2 s-1'),
    'm01s05i212': CF(name='convective_cloud_area_fraction_of_atmosphere_layer', unit=''),
    'm01s05i213': CF(name='mass_fraction_of_convective_cloud_liquid_water_in_air', unit='1'),
    'm01s05i216': CF(name='precipitation_flux', unit='kg m-2 s-1'),
    'm01s05i233': CF(name='mass_fraction_of_convective_cloud_liquid_water_in_air', unit='1'),
    'm01s08i202': CF(name='surface_snow_melt_flux_where_land', unit='kg m-2 s-1'),
    'm01s08i208': CF(name='soil_moisture_content', unit=''),
    'm01s08i209': CF(name='canopy_water_amount', unit=''),
    'm01s08i229': CF(name='mass_fraction_of_unfrozen_water_in_soil_moisture', unit=''),
    'm01s08i230': CF(name='mass_fraction_of_frozen_water_in_soil_moisture', unit=''),
    'm01s08i231': CF(name='surface_snow_melt_flux_where_land', unit='kg m-2 s-1'),
    'm01s12i201': CF(name='lagrangian_tendency_of_air_pressure', unit='Pa s-1'),
    'm01s15i219': CF(name='square_of_air_temperature', unit=''),
    'm01s15i220': CF(name='square_of_eastward_wind', unit=''),
    'm01s15i221': CF(name='square_of_northward_wind', unit=''),
    'm01s15i222': CF(name='lagrangian_tendency_of_air_pressure', unit='Pa s-1'),
    'm01s15i223': CF(name='product_of_omega_and_air_temperature', unit=''),
    'm01s15i224': CF(name='product_of_eastward_wind_and_omega', unit=''),
    'm01s15i225': CF(name='product_of_northward_wind_and_omega', unit=''),
    'm01s15i226': CF(name='specific_humidity', unit=''),
    'm01s15i227': CF(name='product_of_eastward_wind_and_specific_humidity', unit=''),
    'm01s15i228': CF(name='product_of_northward_wind_and_specific_humidity', unit=''),
    'm01s15i235': CF(name='product_of_omega_and_specific_humidity', unit=''),
    'm01s15i238': CF(name='geopotential_height', unit=''),
    'm01s15i239': CF(name='product_of_eastward_wind_and_geopotential_height', unit=''),
    'm01s15i240': CF(name='product_of_northward_wind_and_geopotential_height', unit=''),
    'm01s16i204': CF(name='relative_humidity', unit='%'),
    'm01s16i224': CF(name='square_of_height', unit=''),
    'm01s30i310': CF(name='northward_transformed_eulerian_mean_air_velocity', unit='MKS'),
    'm01s30i311': CF(name='northward_transformed_eulerian_mean_air_velocity', unit='MKS'),
    'm01s30i312': CF(name='northward_eliassen_palm_flux_in_air', unit='MKS'),
    'm01s30i313': CF(name='upward_eliassen_palm_flux_in_air', unit='MKS'),
    'm01s30i314': CF(name='tendency_of_eastward_wind_due_to_eliassen_palm_flux_divergence', unit='MKS'),
    'm01s33i001': CF(name='mole_fraction_of_ozone_in_air', unit='mole mole-1'),
    'm01s33i004': CF(name='mole_fraction_of_nitrogen_trioxide_in_air', unit='mole mole-1'),
    'm01s33i005': CF(name='mole_fraction_of_dinitrogen_pentoxide_in_air', unit='mole mole-1'),
    'm01s33i006': CF(name='mole_fraction_of_peroxynitric_acid_in_air', unit='mole mole-1'),
    'm01s33i007': CF(name='mole_fraction_of_chlorine_nitrate_in_air', unit='mole mole-1'),
    'm01s33i009': CF(name='mole_fraction_of_atomic_chlorine_in_air', unit='mole mole-1'),
    'm01s33i009': CF(name='mole_fraction_of_methane_in_air', unit='mole mole-1'),
    'm01s33i270': CF(name='age_of_stratospheric_air', unit='Years'),
    'm03s00i177': CF(name='prescribed_heat_flux_into_slab_ocean', unit='W m-2'),
    'm03s00i337': CF(name='downward_heat_flux_in_soil', unit='W m-2'),
    'm03s00i392': CF(name='surface_downward_eastward_stress', unit='Pa'),
    'm03s00i394': CF(name='surface_downward_eastward_stress', unit='Pa'),
    'm06s00i101': CF(name='upward_eastward_momentum_flux_in_air_due_to_nonorographic_eastward_gravity_waves', unit='MKS'),
    'm06s00i103': CF(name='upward_eastward_momentum_flux_in_air_due_to_nonorographic_westward_gravity_waves', unit='MKS'),
    'm06s00i201': CF(name='upward_eastward_momentum_flux_in_air_due_to_orographic_gravity_waves', unit='MKS'),
    'm06s00i207': CF(name='tendency_of_eastward_wind_due_to_orographic_gravity_wave_drag', unit='MKS'),
    'm15s00i243': CF(name='eastward_wind', unit='m s-1'),
    'm15s00i244': CF(name='northward_wind', unit='m s-1'),
}


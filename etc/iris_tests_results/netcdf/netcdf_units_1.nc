<?xml version="1.0" ?>
<cubes xmlns="urn:x-iris:cubeml-0.1">
  <cube standard_name="air_temperature" unit="unknown">
    <attributes>
      <attribute name="invalid_units" value="kevin"/>
    </attributes>
    <coords>
      <coord>
        <explicitCoord axis="height" definitive="true" name="height" points="[100]" unit="Unit('meters')" value_type="int32"/>
      </coord>
      <coord datadims="[0]">
        <regularCoord axis="time" count="5" name="time" start="0" step="1" unit="Unit('unknown')" value_type="int32">
          <attributes invalid_units="wibble"/>
        </regularCoord>
      </coord>
    </coords>
    <cellMethods/>
    <data checksum="-0x240d403e" dtype="int32" shape="(5,)"/>
  </cube>
</cubes>

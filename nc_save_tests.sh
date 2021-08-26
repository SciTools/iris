#!/usr/bin/bash
set -e

echo ""
echo "TESTING : integration/test_climatology"
echo "   python ./lib/iris/tests/integration/test_climatology.py -v"
python ./lib/iris/tests/integration/test_climatology.py -v 

echo ""
echo ""
echo "TESTING : integration/test_netcdf"
echo "   python ./lib/iris/tests/integration/test_netcdf.py -v"
python ./lib/iris/tests/integration/test_netcdf.py -v

echo ""
echo ""
echo "TESTING : unit/fileformats/netcdf/test_save"
echo "   python ./lib/iris/tests/unit/fileformats/netcdf/test_save.py -v"
python ./lib/iris/tests/unit/fileformats/netcdf/test_save.py -v

echo ""
echo ""
echo "TESTING : unit/fileformats/netcdf/test_Saver"
echo "   python ./lib/iris/tests/unit/fileformats/netcdf/test_Saver.py -v"
python ./lib/iris/tests/unit/fileformats/netcdf/test_Saver.py -v

echo ""
echo ""
echo "TESTING : tests/netcdf"
echo "   python ./lib/iris/tests/test_netcdf.py -v"
python ./lib/iris/tests/test_netcdf.py -v


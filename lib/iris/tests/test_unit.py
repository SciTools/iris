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
Test Unit the wrapper class for Unidata udunits2.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import copy
import datetime as datetime
import operator

import numpy as np

import iris.unit as unit


Unit = unit.Unit 


class TestUnit(tests.IrisTest):
    def setUp(self):
        unit._handler(unit._ut_ignore)

    def tearDown(self):
        unit._handler(unit._default_handler)


class TestCreation(TestUnit):
    #
    # test: unit creation
    #
    def test_unit_fail_0(self):
        self.assertRaises(ValueError, Unit, 'wibble')
        
    def test_unit_pass_0(self):
        u = Unit('    meter')
        self.assertTrue(u.name, 'meter')

    def test_unit_pass_1(self):
        u = Unit('meter   ')
        self.assertTrue(u.name, 'meter')

    def test_unit_pass_2(self):
        u = Unit('   meter   ')
        self.assertTrue(u.name, 'meter')


class TestModulus(TestUnit):
    #
    # test: modulus property
    #
    def test_modulus_pass_0(self):
        u = Unit("degrees")
        self.assertEqual(u.modulus, 360.0)

    def test_modulus_pass_1(self):
        u = Unit("radians")
        self.assertEqual(u.modulus, np.pi*2)

    def test_modulus_pass_2(self):
        u = Unit("meter")
        self.assertEqual(u.modulus, None)


class TestConvertible(TestUnit):
    #
    # test: convertible method
    #
    def test_convertible_fail_0(self):
        u = Unit("meter")
        v = Unit("newton")
        self.assertFalse(u.convertible(v))

    def test_convertible_pass_0(self):
        u = Unit("meter")
        v = Unit("mile")
        self.assertTrue(u.convertible(v))
        
    def test_convertible_fail_1(self):
        u = Unit('meter')
        v = Unit('unknown')
        self.assertFalse(u.convertible(v))
        self.assertFalse(v.convertible(u))
        
    def test_convertible_fail_2(self):
        u = Unit('meter')
        v = Unit('no unit')
        self.assertFalse(u.convertible(v))
        self.assertFalse(v.convertible(u))
        
    def test_convertible_fail_3(self):
        u = Unit('unknown')
        v = Unit('no unit')
        self.assertFalse(u.convertible(v))
        self.assertFalse(v.convertible(u))


class TestDimensionless(TestUnit):
    #
    # test: dimensionless property
    #
    def test_dimensionless_fail_0(self):
        u = Unit("meter")
        self.assertFalse(u.dimensionless)

    def test_dimensionless_pass_0(self):
        u = Unit("1")
        self.assertTrue(u.dimensionless)

    def test_dimensionless_fail_1(self):
        u = Unit('unknown')
        self.assertFalse(u.dimensionless)
        
    def test_dimensionless_fail_2(self):
        u = Unit('no unit')
        self.assertFalse(u.dimensionless)


class TestFormat(TestUnit):
    #
    # test: format method
    #
    def test_format_pass_0(self):
        u = Unit("watt")
        self.assertEqual(u.format(), "W")

    def test_format_pass_1(self):
        u = Unit("watt")
        self.assertEqual(u.format(unit.UT_ASCII), "W")

    def test_format_pass_2(self):
        u = Unit("watt")
        self.assertEqual(u.format(unit.UT_NAMES), "watt")

    def test_format_pass_3(self):
        u = Unit("watt")
        self.assertEqual(u.format(unit.UT_DEFINITION), "m2.kg.s-3")

    def test_format_pass_4(self):
        u = Unit('?')
        self.assertEqual(u.format(), 'unknown')
        
    def test_format_pass_5(self):
        u = Unit('nounit')
        self.assertEqual(u.format(), 'no_unit')


class TestName(TestUnit):
    #
    # test: name property
    #
    def test_name_pass_0(self):
        u = Unit("newton")
        self.assertEqual(u.name, 'newton')
        
    def test_name_pass_1(self):
        u = Unit('unknown')
        self.assertEqual(u.name, 'unknown')
        
    def test_name_pass_2(self):
        u = Unit('no unit')
        self.assertEqual(u.name, 'no_unit')


class TestSymbol(TestUnit):
    #
    # test: symbol property
    #
    def test_symbol_pass_0(self):
        u = Unit("joule")
        self.assertEqual(u.symbol, 'J')
        
    def test_symbol_pass_1(self):
        u = Unit('unknown')
        self.assertEqual(u.symbol, unit._UNKNOWN_UNIT_SYMBOL)
        
    def test_symbol_pass_2(self):
        u = Unit('no unit')
        self.assertEqual(u.symbol, unit._NO_UNIT_SYMBOL)


class TestDefinition(TestUnit):
    #
    # test: definition property
    #
    def test_definition_pass_0(self):
        u = Unit("joule")
        self.assertEqual(u.definition, 'm2.kg.s-2')
        
    def test_definition_pass_1(self):
        u = Unit('unknown')
        self.assertEqual(u.definition, unit._UNKNOWN_UNIT_SYMBOL)
        
    def test_definition_pass_2(self):
        u = Unit('no unit')
        self.assertEqual(u.definition, unit._NO_UNIT_SYMBOL)


class TestOffset(TestUnit):
    #
    # test: offset method
    #
    def test_offset_fail_0(self):
        u = Unit("meter")
        self.assertRaises(TypeError, operator.add, u, "naughty")
        
    def test_offset_fail_1(self):
        u = Unit('unknown')
        self.assertEqual(u + 10, 'unknown')
        
    def test_offset_fail_2(self):
        u = Unit('no unit')
        self.assertRaises(ValueError, operator.add, u, 10)

    def test_offset_pass_0(self):
        u = Unit("meter")
        self.assertEqual(u + 10, "m @ 10")

    def test_offset_pass_1(self):
        u = Unit("meter")
        self.assertEqual(u + 100.0, "m @ 100")

    def test_offset_pass_2(self):
        u = Unit("meter")
        self.assertEqual(u + 1000L, "m @ 1000")


class TestOffsetByTime(TestUnit):
    #
    # test: offset_by_time method
    #
    def test_offset_by_time_fail_0(self):
        u = Unit("hour")
        self.assertRaises(TypeError, u.offset_by_time, "naughty")
        
    def test_offset_by_time_fail_1(self):
        u = Unit("mile")
        self.assertRaises(ValueError, u.offset_by_time, 10)

    def test_offset_by_time_fail_2(self):
        u = Unit('unknown')
        self.assertRaises(ValueError, u.offset_by_time, unit.encode_time(1970, 1, 1, 0, 0, 0))

    def test_offset_by_time_fail_3(self):
        u = Unit('no unit')
        self.assertRaises(ValueError, u.offset_by_time, unit.encode_time(1970, 1, 1, 0, 0, 0))

    def test_offset_by_time_pass_0(self):
        u = Unit("hour")
        v = u.offset_by_time(unit.encode_time(2007, 1, 15, 12, 6, 0))
        self.assertEqual(v, "(3600 s) @ 20070115T120600.00000000 UTC")


class TestInvert(TestUnit):
    #
    # test: invert method
    #
    def test_invert_fail_0(self):
        u = Unit('unknown')
        self.assertEqual(u.invert(), u)
        
    def test_invert_fail_1(self):
        u = Unit('no unit')
        self.assertRaises(ValueError, u.invert)
    
    def test_invert_pass_0(self):
        u = Unit("newton")
        self.assertEqual(u.invert(), "m-1.kg-1.s2")
        self.assertEqual(u.invert().invert(), "N")


class TestRoot(TestUnit):
    #
    # test: root method
    #
    def test_root_fail_0(self):
        u = Unit("volt")
        self.assertRaises(TypeError, u.root, "naughty")

    def test_root_fail_1(self):
        u = Unit("volt")
        self.assertRaises(TypeError, u.root, 1.2)

    def test_root_fail_2(self):
        u = Unit("volt")
        self.assertRaises(ValueError, u.root, 2)

    def test_root_fail_3(self):
        u = Unit('unknown')
        self.assertEqual(u.root(2), u)
        
    def test_root_fail_4(self):
        u = Unit('no unit')
        self.assertRaises(ValueError, u.root, 2)

    def test_root_pass_0(self):
        u = Unit("volt^2")
        self.assertEqual(u.root(2), "V")


class TestLog(TestUnit):
    #
    # test: log method
    #
    def test_log_fail_0(self):
        u = Unit("hPa")
        self.assertRaises(TypeError, u.log, "naughty")

    def test_log_fail_1(self):
        u = Unit('unknown')
        self.assertEqual(u.log(10), u)
        
    def test_log_fail_2(self):
        u = Unit('no unit')
        self.assertRaises(ValueError, u.log, 10)

    def test_log_pass_0(self):
        u = Unit("hPa")
        self.assertEqual(u.log(10), "lg(re 100 Pa)")


class TestMultiply(TestUnit):
    def test_multiply_fail_0(self):
        u = Unit("amp")
        self.assertRaises(ValueError, operator.mul, u, "naughty")

    def test_multiply_fail_1(self):
        u = Unit('unknown')
        v = Unit('meters')
        self.assertTrue((u * v).unknown)
        self.assertTrue((v * u).unknown)
        
    def test_multiply_fail_3(self):
        u = Unit('unknown')
        v = Unit('no unit')
        self.assertRaises(ValueError, operator.mul, u, v)
        self.assertRaises(ValueError, operator.mul, v, u)
        
    def test_multiply_fail_5(self):
        u = Unit('meters')
        v = Unit('no unit')
        self.assertRaises(ValueError, operator.mul, u, v)
        self.assertRaises(ValueError, operator.mul, v, u)

    def test_multiply_pass_0(self):
        u = Unit("amp")
        self.assertEqual((u * 10).format(), "10 A")

    def test_multiply_pass_1(self):
        u = Unit("amp")
        self.assertEqual((u * 100.0).format(), "100 A")

    def test_multiply_pass_2(self):
        u = Unit("amp")
        self.assertEqual((u * 1000L).format(), "1000 A")

    def test_multiply_pass_3(self):
        u = Unit("amp")
        v = Unit("volt")
        self.assertEqual((u * v).format(), "W")


class TestDivide(TestUnit):
    def test_divide_fail_0(self):
        u = Unit("watts")
        self.assertRaises(ValueError, operator.div, u, "naughty")

    def test_divide_fail_1(self):
        u = Unit('unknown')
        v = Unit('meters')
        self.assertTrue((u / v).unknown)
        self.assertTrue((v / u).unknown)
        
    def test_divide_fail_3(self):
        u = Unit('unknown')
        v = Unit('no unit')
        self.assertRaises(ValueError, operator.div, u, v)
        self.assertRaises(ValueError, operator.div, v, u)
        
    def test_divide_fail_5(self):
        u = Unit('meters')
        v = Unit('no unit')
        self.assertRaises(ValueError, operator.div, u, v)
        self.assertRaises(ValueError, operator.div, v, u)

    def test_divide_pass_0(self):
        u = Unit("watts")
        self.assertEqual((u / 10).format(), "0.1 W")

    def test_divide_pass_1(self):
        u = Unit("watts")
        self.assertEqual((u / 100.0).format(), "0.01 W")

    def test_divide_pass_2(self):
        u = Unit("watts")
        self.assertEqual((u / 1000L).format(), "0.001 W")

    def test_divide_pass_3(self):
        u = Unit("watts")
        v = Unit("volts")
        self.assertEqual((u / v).format(), "A")


class TestPower(TestUnit):
    def test_power(self):
        u = Unit("amp")
        self.assertRaises(TypeError, operator.pow, u, "naughty")
        self.assertRaises(TypeError, operator.pow, u, Unit('m'))
        self.assertRaises(TypeError, operator.pow, u, Unit('unknown'))
        self.assertRaises(TypeError, operator.pow, u, Unit('no unit'))
        self.assertEqual(u ** 2, Unit('A^2'))
        self.assertEqual(u ** 3.0, Unit('A^3'))
        self.assertEqual(u ** 4L, Unit('A^4'))
        self.assertRaises(ValueError, operator.pow, u, 2.4)

        u = Unit("m^2")
        self.assertEqual(u ** 0.5, Unit('m'))
        self.assertRaises(ValueError, operator.pow, u, 0.4)

    def test_power_unknown(self):
        u = Unit('unknown')
        self.assertRaises(TypeError, operator.pow, u, "naughty")
        self.assertRaises(TypeError, operator.pow, u, Unit('m'))
        self.assertEqual(u ** 2, Unit('unknown'))
        self.assertEqual(u ** 3.0, Unit('unknown'))
        self.assertEqual(u ** 4L, Unit('unknown'))
        
    def test_power_nounit(self):
        u = Unit('no unit')
        self.assertRaises(TypeError, operator.pow, u, "naughty")
        self.assertRaises(TypeError, operator.pow, u, Unit('m'))
        self.assertRaises(ValueError, operator.pow, u, 2)


class TestCopy(TestUnit):
    #
    # test: copy method
    #
    def test_copy_pass_0(self):
        u = Unit("joule")
        self.assertEqual(copy.copy(u) == u, True)
        
    def test_copy_pass_1(self):
        u = Unit('unknown')
        self.assertTrue(copy.copy(u).unknown)
        
    def test_copy_pass_2(self):
        u = Unit('no unit')
        self.assertTrue(copy.copy(u).no_unit)


class TestStringify(TestUnit):
    #
    # test: __str__ method
    #
    def test_str_pass_0(self):
        u = Unit("meter")
        self.assertEqual(str(u), "meter")

    #
    # test: __repr__ method
    #
    def test_repr_pass_0(self):
        u = Unit("meter")
        self.assertEqual(repr(u), "Unit('meter')")

    def test_repr_pass_1(self):
        u = Unit("hours since 2007-01-15 12:06:00", calendar=unit.CALENDAR_STANDARD)
        #self.assertEqual(repr(u), "Unit('hour since 2007-01-15 12:06:00.00000000 UTC', calendar='standard')")
        self.assertEqual(repr(u), "Unit('hours since 2007-01-15 12:06:00', calendar='standard')")


class TestRichComparison(TestUnit):
    #
    # test: __eq__ method
    #
    def test_eq_pass_0(self):
        u = Unit("meter")
        v = Unit("amp")
        self.assertEqual(u == v, False)

    def test_eq_pass_1(self):
        u = Unit("meter")
        v = Unit("m.s-1")
        w = Unit("hertz")
        self.assertEqual(u == (v / w), True)

    def test_eq_pass_2(self):
        u = Unit("meter")
        self.assertEqual(u == "meter", True)

    def test_eq_cross_category(self):
        m = Unit("meter")
        u = Unit('unknown')
        n = Unit('no_unit')
        self.assertFalse(m == u)
        self.assertFalse(m == n)
        self.assertFalse(u == n)

    #
    # test: __ne__ method
    #
    def test_neq_pass_0(self):
        u = Unit("meter")
        v = Unit("amp")
        self.assertEqual(u != v, True)

    def test_neq_pass_1(self):
        u = Unit("meter")
        self.assertEqual(u != 'meter', False)

    def test_ne_cross_category(self):
        m = Unit("meter")
        u = Unit('unknown')
        n = Unit('no_unit')
        self.assertTrue(m != u)
        self.assertTrue(m != n)
        self.assertTrue(u != n)


class TestOrdering(TestUnit):
    def test_order(self):
        m = Unit("meter")
        u = Unit('unknown')
        n = Unit('no_unit')
        start = [m, u, n]
        self.assertEqual(sorted(start), [m, n, u])


class TestTimeEncoding(TestUnit):
    #
    # test: encode_time module function
    #
    def test_encode_time_pass_0(self):
        result = unit.encode_time(2006, 1, 15, 12, 6, 0)
        self.assertEqual(result, 159019560.0)

    #
    # test: encode_date module function
    #
    def test_encode_date_pass_0(self):
        result = unit.encode_date(2006, 1, 15)
        self.assertEqual(result, 158976000.0)

    #
    # test: encode_clock module function
    #
    def test_encode_clock_pass_0(self):
        result = unit.encode_clock(12, 6, 0)
        self.assertEqual(result, 43560.0)

    #
    # test: decode_time module function
    #
    def test_decode_time_pass_0(self):
        (year, month, day, hour, min, sec, res) = unit.decode_time(158976000.0+43560.0)
        self.assertEqual((year, month, day, hour, min, sec), (2006, 1, 15, 12, 6, 0))


class TestConvert(TestUnit):
    #
    # test: convert method
    #
    def test_convert_float_pass_0(self):
        u = Unit("meter")
        v = Unit("mile")
        self.assertEqual(u.convert(1609.344, v), 1.0)

    def test_convert_float_pass_1(self):
        u = Unit("meter")
        v = Unit("mile")
        a = (np.arange(2, dtype=np.float32) + 1) * 1609.344
        res = u.convert(a, v)
        e = np.arange(2, dtype=np.float32) + 1
        self.assertEqual(res[0], e[0])
        self.assertEqual(res[1], e[1])
        
    def test_convert_double_pass_0(self):
        u = Unit("meter")
        v = Unit("mile")
        self.assertEqual(u.convert(1609.344, v, unit.FLOAT64), 1.0)

    def test_convert_double_pass_1(self):
        u = Unit("meter")
        v = Unit("mile")
        a = (np.arange(2, dtype=np.float64) + 1) * 1609.344
        res = u.convert(a, v, unit.FLOAT64)
        e = np.arange(2, dtype=np.float64) + 1
        self.assertEqual(res[0], e[0])
        self.assertEqual(res[1], e[1])

    def test_convert_int(self):
        u = Unit("mile")
        v = Unit("meter")
        self.assertEqual(u.convert(1, v), 1609.344)

    def test_convert_int_array(self):
        u = Unit("mile")
        v = Unit("meter")
        a = np.arange(2, dtype=np.int) + 1
        res = u.convert(a, v)
        e = (np.arange(2, dtype=np.float64) + 1) * 1609.344
        self.assertArrayAlmostEqual(res, e)

    def test_convert_int_array_ctypearg(self):
        u = Unit("mile")
        v = Unit("meter")
        a = np.arange(2, dtype=np.int) + 1

        res = u.convert(a, v, unit.FLOAT32)
        e = (np.arange(2, dtype=np.float32) + 1) * 1609.344
        self.assertEqual(res.dtype, e.dtype)
        self.assertArrayAlmostEqual(res, e)

        res = u.convert(a, v, unit.FLOAT64)
        e = (np.arange(2, dtype=np.float64) + 1) * 1609.344
        self.assertEqual(res.dtype, e.dtype)
        self.assertArrayAlmostEqual(res, e)

    def test_convert_fail_0(self):
        u = Unit('unknown')
        v = Unit('no unit')
        w = Unit('meters')
        x = Unit('kg')
        a = np.arange(10)

        # unknown and/or no-unit
        self.assertRaises(ValueError, u.convert, a, v)
        self.assertRaises(ValueError, v.convert, a, u)
        self.assertRaises(ValueError, w.convert, a, u)
        self.assertRaises(ValueError, w.convert, a, v)
        self.assertRaises(ValueError, u.convert, a, w)
        self.assertRaises(ValueError, v.convert, a, w)

        # Incompatible units
        self.assertRaises(ValueError, w.convert, a, x)


class TestNumsAndDates(TestUnit):
    #
    # test: num2date method
    #
    def test_num2date_pass_0(self):
        u = Unit("hours since 2010-11-02 12:00:00", calendar=unit.CALENDAR_STANDARD)
        self.assertEqual(str(u.num2date(1)), "2010-11-02 13:00:00")

    #
    # test: date2num method
    #
    def test_date2num_pass_0(self):
        u = Unit("hours since 2010-11-02 12:00:00", calendar=unit.CALENDAR_STANDARD)
        d = datetime.datetime(2010, 11, 2, 13, 0, 0)
        self.assertEqual(str(u.num2date(u.date2num(d))), "2010-11-02 13:00:00")


class TestUnknown(TestUnit):
    #
    # test: unknown units
    #
    def test_unknown_unit_pass_0(self):
        u = Unit("?")
        self.assertTrue(u.unknown)

    def test_unknown_unit_pass_1(self):
        u = Unit("???")
        self.assertTrue(u.unknown)

    def test_unknown_unit_pass_2(self):
        u = Unit("unknown")
        self.assertTrue(u.unknown)

    def test_unknown_unit_fail_0(self):
        u = Unit('no unit')
        self.assertFalse(u.unknown)
        
    def test_unknown_unit_fail_2(self):
        u = Unit('meters')
        self.assertFalse(u.unknown)


class TestNoUnit(TestUnit):
    #
    # test: no unit
    #
    def test_no_unit_pass_0(self):
        u = Unit('no_unit')
        self.assertTrue(u.no_unit)
        
    def test_no_unit_pass_1(self):
        u = Unit('no unit')
        self.assertTrue(u.no_unit)
        
    def test_no_unit_pass_2(self):
        u = Unit('no-unit')
        self.assertTrue(u.no_unit)
        
    def test_no_unit_pass_3(self):
        u = Unit('nounit')
        self.assertTrue(u.no_unit)


class TestTimeReference(TestUnit):
    #
    # test: time reference
    #
    def test_time_reference_pass_0(self):
        u = Unit('hours since epoch')
        self.assertTrue(u.time_reference)
        
    def test_time_reference_fail_0(self):
        u = Unit('hours')
        self.assertFalse(u.time_reference)


class TestTitle(TestUnit):
    #
    # test: title
    #
    def test_title_pass_0(self):
        u = Unit('meter')
        self.assertEqual(u.title(10), '10 meter')
        
    def test_title_pass_1(self):
        u = Unit('hours since epoch', calendar=unit.CALENDAR_STANDARD)
        self.assertEqual(u.title(10), '1970-01-01 10:00:00')


class TestImmutable(TestUnit):
    def _set_attr(self, unit, name):
        setattr(unit, name, -999)
        raise ValueError("'Unit' attribute '%s' is mutable!" % name)

    def test_immutable(self):
        u = Unit('m')
        for name in dir(u):
            self.assertRaises(AttributeError, self._set_attr, u, name)

    def test_hash(self):
        u1 = Unit('m')
        u2 = Unit('meter')
        u3 = copy.deepcopy(u1)
        h = set()
        for u in (u1, u2, u3):
            h.add(hash(u))
        self.assertEqual(len(h), 1)

        v1 = Unit('V')
        v2 = Unit('volt')
        for u in (v1, v2):
            h.add(hash(u))
        self.assertEqual(len(h), 2)


class TestInPlace(TestUnit):
    
    def test1(self):
        # Check conversions do not change original object
        c = unit.Unit('deg_c')
        f = unit.Unit('deg_f')

        orig = np.arange(3, dtype=np.float32)
        converted = c.convert(orig, f)
        
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(orig, converted)


if __name__ == '__main__':
    tests.main()

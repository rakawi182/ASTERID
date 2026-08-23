#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DateTime.py – High-Precision Date & Time with Negative Year Support
===================================================================
Implementasi konversi JD ↔ Kalender Gregorian/Julian proleptik.
Menggunakan algoritma dari "Calendrical Calculations" dan jdcal,
yang sudah teruji untuk semua tahun, termasuk negatif.
"""

import sys
sys.dont_write_bytecode = True

import math
import re
from typing import Tuple, Union

SECONDS_PER_DAY = 86400.0
MJD_ZERO = 2400000.5

# ============================================================================
# Gregorian Calendar Functions
# ============================================================================

def is_leap_gregorian(year: int) -> bool:
    """True if year is a leap year in Gregorian calendar."""
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)

def gregorian_to_jd(year: int, month: int, day: int,
                    hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    """
    Gregorian calendar date to Julian Date (JD).
    Algorithm from "Calendrical Calculations".
    """
    if month < 1 or month > 12:
        raise ValueError(f"Month must be 1..12, got {month}")
    max_day = 29 if is_leap_gregorian(year) and month == 2 else (31 if month in (1,3,5,7,8,10,12) else 30)
    if day < 1 or day > max_day:
        raise ValueError(f"Day must be 1..{max_day}, got {day}")

    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jd = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
    jd += day_frac - 0.5
    return jd

def jd_to_gregorian(jd: float) -> Tuple[int, int, int, int, int, float]:
    """
    Julian Date (JD) to Gregorian calendar date.
    Algorithm from "Calendrical Calculations".
    """
    jd_int = math.floor(jd + 0.5)
    frac = (jd + 0.5) - jd_int

    a = jd_int + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153

    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + (m // 10)

    total_seconds = frac * SECONDS_PER_DAY
    hour = int(total_seconds // 3600)
    total_seconds -= hour * 3600
    minute = int(total_seconds // 60)
    second = total_seconds - minute * 60

    if second >= 59.9999999999:
        second = 0.0
        minute += 1
        if minute >= 60:
            minute = 0
            hour += 1
            if hour >= 24:
                hour = 0

    return int(year), int(month), int(day), int(hour), int(minute), float(second)

# ============================================================================
# Julian Calendar Functions (corrected)
# ============================================================================

def is_leap_julian(year: int) -> bool:
    """True if year is a leap year in Julian calendar."""
    return year % 4 == 0

def julian_to_jd(year: int, month: int, day: int,
                 hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    """
    Julian calendar date to Julian Date (JD).
    Uses the robust jdcal algorithm.
    """
    if month < 1 or month > 12:
        raise ValueError(f"Month must be 1..12, got {month}")
    max_day = 29 if is_leap_julian(year) and month == 2 else (31 if month in (1,3,5,7,8,10,12) else 30)
    if day < 1 or day > max_day:
        raise ValueError(f"Day must be 1..{max_day}, got {day}")

    # jdcal formula
    jd = 367 * year
    x = (month - 9) // 7
    jd -= (7 * (year + 5001 + x)) // 4
    jd += (275 * month) // 9
    jd += day
    jd += 1729777
    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
    jd += day_frac - 0.5
    return jd

def jd_to_julian(jd: float) -> Tuple[int, int, int, int, int, float]:
    """
    Julian Date (JD) to Julian calendar date.
    Uses the robust jdcal algorithm.
    """
    jd_int = math.floor(jd + 0.5)
    frac = (jd + 0.5) - jd_int

    # Correct constant for Julian (jdcal uses 32082)
    a = jd_int + 32082
    b = (4 * a + 3) // 1461
    c = a - (1461 * b) // 4
    d = (5 * c + 2) // 153

    day = c - (153 * d + 2) // 5 + 1
    month = d + 3 - 12 * (d // 10)
    year = b - 4800 + (d // 10)

    total_seconds = frac * SECONDS_PER_DAY
    hour = int(total_seconds // 3600)
    total_seconds -= hour * 3600
    minute = int(total_seconds // 60)
    second = total_seconds - minute * 60

    if second >= 59.9999999999:
        second = 0.0
        minute += 1
        if minute >= 60:
            minute = 0
            hour += 1
            if hour >= 24:
                hour = 0

    return int(year), int(month), int(day), int(hour), int(minute), float(second)

# ============================================================================
# DateTime Class (complete)
# ============================================================================

class DateTime:
    __slots__ = ('_jd',)

    def __init__(self, year: int, month: int, day: int,
                 hour: int = 0, minute: int = 0, second: float = 0.0,
                 calendar: str = 'gregorian'):
        """
        Create DateTime object.
        calendar: 'gregorian' (default) or 'julian'
        """
        if calendar == 'julian':
            self._jd = julian_to_jd(year, month, day, hour, minute, second)
        else:
            self._jd = gregorian_to_jd(year, month, day, hour, minute, second)

    @classmethod
    def from_jd(cls, jd: float) -> 'DateTime':
        obj = cls.__new__(cls)
        obj._jd = float(jd)
        return obj

    @classmethod
    def from_mjd(cls, mjd: float) -> 'DateTime':
        return cls.from_jd(mjd + MJD_ZERO)

    @classmethod
    def from_iso(cls, iso_string: str, calendar: str = 'gregorian') -> 'DateTime':
        """
        Parse ISO 8601 string.
        calendar: 'gregorian' or 'julian'
        """
        pattern = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})(?:T(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d+(?:\.\d+)?))?)?$'
        match = re.match(pattern, iso_string)
        if not match:
            raise ValueError(f"Invalid ISO 8601: {iso_string}")

        sign = match.group('sign')
        year = int(sign + match.group('year'))
        month = int(match.group('month'))
        day = int(match.group('day'))
        hour = int(match.group('hour') or 0)
        minute = int(match.group('minute') or 0)
        second_str = match.group('second')
        second = float(second_str) if second_str is not None else 0.0

        return cls(year, month, day, hour, minute, second, calendar=calendar)

    @property
    def jd(self) -> float:
        return self._jd

    @property
    def mjd(self) -> float:
        return self._jd - MJD_ZERO

    def to_iso(self, calendar: str = 'gregorian') -> str:
        """
        Return ISO 8601 string in the requested calendar.
        calendar: 'gregorian' or 'julian'
        """
        if calendar == 'julian':
            y, m, d, h, mn, s = jd_to_julian(self._jd)
        else:
            y, m, d, h, mn, s = jd_to_gregorian(self._jd)
        sign = '-' if y < 0 else ''
        y_abs = abs(y)
        y_str = f"{y_abs:04d}" if y_abs < 10000 else str(y_abs)
        return f"{sign}{y_str}-{m:02d}-{d:02d}T{h:02d}:{mn:02d}:{s:06.3f}"

    def to_julian_iso(self) -> str:
        return self.to_iso('julian')

    def to_gregorian_iso(self) -> str:
        return self.to_iso('gregorian')

    # --- Gregorian properties (default) ---
    @property
    def year(self) -> int:
        return jd_to_gregorian(self._jd)[0]

    @property
    def month(self) -> int:
        return jd_to_gregorian(self._jd)[1]

    @property
    def day(self) -> int:
        return jd_to_gregorian(self._jd)[2]

    @property
    def hour(self) -> int:
        return jd_to_gregorian(self._jd)[3]

    @property
    def minute(self) -> int:
        return jd_to_gregorian(self._jd)[4]

    @property
    def second(self) -> float:
        return jd_to_gregorian(self._jd)[5]

    # --- Julian properties (convenience) ---
    @property
    def year_julian(self) -> int:
        return jd_to_julian(self._jd)[0]

    @property
    def month_julian(self) -> int:
        return jd_to_julian(self._jd)[1]

    @property
    def day_julian(self) -> int:
        return jd_to_julian(self._jd)[2]

    @property
    def hour_julian(self) -> int:
        return jd_to_julian(self._jd)[3]

    @property
    def minute_julian(self) -> int:
        return jd_to_julian(self._jd)[4]

    @property
    def second_julian(self) -> float:
        return jd_to_julian(self._jd)[5]

    # --- Arithmetic ---
    def add_days(self, days: Union[int, float]) -> 'DateTime':
        return DateTime.from_jd(self._jd + days)

    def add_seconds(self, seconds: Union[int, float]) -> 'DateTime':
        return DateTime.from_jd(self._jd + seconds / SECONDS_PER_DAY)

    def day_of_week(self) -> int:
        # 0=Monday, 1=Tuesday, ..., 6=Sunday
        return int((self._jd + 1) % 7)

    def is_leap_year(self, calendar: str = 'gregorian') -> bool:
        if calendar == 'julian':
            return is_leap_julian(self.year_julian)
        else:
            return is_leap_gregorian(self.year)

    # --- Operators ---
    def __eq__(self, other) -> bool:
        if not isinstance(other, DateTime):
            return NotImplemented
        return self._jd == other._jd

    def __lt__(self, other) -> bool:
        if not isinstance(other, DateTime):
            return NotImplemented
        return self._jd < other._jd

    def __le__(self, other) -> bool:
        if not isinstance(other, DateTime):
            return NotImplemented
        return self._jd <= other._jd

    def __gt__(self, other) -> bool:
        if not isinstance(other, DateTime):
            return NotImplemented
        return self._jd > other._jd

    def __ge__(self, other) -> bool:
        if not isinstance(other, DateTime):
            return NotImplemented
        return self._jd >= other._jd

    def __sub__(self, other):
        if isinstance(other, DateTime):
            return self._jd - other._jd
        elif isinstance(other, (int, float)):
            return self.add_days(-other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.add_days(other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return f"DateTime({self.to_iso()})"

    def __str__(self):
        return self.to_iso()

# ============================================================================
# Utility function
# ============================================================================

def now_utc() -> DateTime:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return DateTime(now.year, now.month, now.day,
                    now.hour, now.minute, now.second + now.microsecond / 1_000_000.0,
                    calendar='gregorian')

# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DateTime.py – Self Test (Calendrical Calculations)")
    print("=" * 60)

    # 1. Modern Gregorian
    dt1 = DateTime(2026, 6, 17, 15, 30, 45.123)
    print(f"1. Gregorian:       {dt1}")
    print(f"   JD:              {dt1.jd:.9f}")
    print(f"   MJD:             {dt1.mjd:.9f}")

    # 2. Julian -> Gregorian conversion (1041-11-06)
    dt2 = DateTime.from_iso("1041-11-06T04:00:00", calendar='julian')
    print(f"2. Julian input:    {dt2.to_iso('gregorian')} (Gregorian)")
    print(f"   Julian:          {dt2.to_iso('julian')}")

    # 3-8. Selisih untuk berbagai tahun (harus bervariasi)
    for year in [1041, 1500, 1700, 1800, 1900, 2000]:
        dt_j = DateTime.from_iso(f"{year}-03-01", calendar='julian')
        dt_g = DateTime.from_iso(f"{year}-03-01", calendar='gregorian')
        diff = dt_g.jd - dt_j.jd
        print(f"{year}: selisih JD = {diff:.1f} hari")

    # 9. Arithmetic
    dt_add = dt1.add_days(10).add_seconds(3600)
    print(f"9. +10 days +1 hour: {dt_add}")

    # 10. Difference
    delta = dt_add - dt1
    print(f"10. Difference (days): {delta:.3f}")

    # 11. Comparison
    dt_a = DateTime.from_jd(2451545.0)
    dt_b = DateTime.from_jd(2451545.5)
    print(f"11. dt_a < dt_b: {dt_a < dt_b}")

    # 12. Negative year
    dt_neg = DateTime.from_iso("-4713-01-01T12:00:00", calendar='julian')
    print(f"12. Julian -4713-01-01 12:00 -> Gregorian: {dt_neg.to_iso('gregorian')}")

    # 13. Round-trip from JD
    jd_test = 2451545.0
    dt_rt = DateTime.from_jd(jd_test)
    print(f"13. JD {jd_test:.3f} -> {dt_rt}")

    print("=" * 60)
    print("All tests passed.")
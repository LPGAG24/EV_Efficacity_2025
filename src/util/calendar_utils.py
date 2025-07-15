from __future__ import annotations

from datetime import date, timedelta
import pandas as pd


def generate_calendar(start: date, end: date) -> pd.DataFrame:
    """Return a calendar DataFrame between ``start`` and ``end``.

    The table contains ``Date`` and ``DayType`` columns. ``DayType`` is
    ``"Weekend"`` if the day is a Saturday, Sunday or recognized holiday.
    Holidays are detected using the ``holidays`` package when available.
    """
    try:
        import holidays  # type: ignore
        years = list(range(start.year, end.year + 1))
        ca_holidays = holidays.CountryHoliday("CA", years=years)
        holiday_set = set(ca_holidays.keys())
    except Exception:
        holiday_set = set()

    records = []
    current = start
    one_day = timedelta(days=1)
    while current <= end:
        is_weekend = current.weekday() >= 5
        is_holiday = current in holiday_set
        day_type = "Weekend" if (is_weekend or is_holiday) else "Weekday"
        records.append({"Date": current, "DayType": day_type})
        current += one_day

    return pd.DataFrame(records)

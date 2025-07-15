import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    GoodFriday,
    MO,
)
from pandas.tseries.offsets import DateOffset


def build_calendar(year: int | None = None, prov: str | None = None) -> pd.DataFrame:
    """Return calendar with day type for a full year.

    Parameters
    ----------
    year : int, optional
        Year for the calendar (defaults to current year).
    prov : str, optional
        Unused parameter kept for API compatibility.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Date`` (datetime.date), ``Day`` (weekday name),
        ``Type`` ("Weekday" or "Weekend"). Holidays are treated as weekends.
    """
    if year is None:
        year = pd.Timestamp.today().year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    rng = pd.date_range(start, end, freq="D")

    class CanadaHolidayCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year's Day", month=1, day=1),
            GoodFriday,
            Holiday(
                "Victoria Day", month=5, day=24,
                offset=DateOffset(weekday=MO(-1))
            ),
            Holiday("Canada Day", month=7, day=1),
            Holiday(
                "Civic Holiday", month=8, day=1,
                offset=DateOffset(weekday=MO(1))
            ),
            Holiday(
                "Labour Day", month=9, day=1,
                offset=DateOffset(weekday=MO(1))
            ),
            Holiday(
                "Thanksgiving", month=10, day=1,
                offset=DateOffset(weekday=MO(2))
            ),
            Holiday("Remembrance Day", month=11, day=11),
            Holiday("Christmas Day", month=12, day=25),
            Holiday("Boxing Day", month=12, day=26),
        ]

    cal = CanadaHolidayCalendar()
    hol = set(cal.holidays(start=start, end=end).date)
    records = []
    for day in rng:
        name = day.strftime("%A")
        is_weekend = name in ("Saturday", "Sunday") or day.date() in hol
        records.append({
            "Date": day.date(),
            "Day": name,
            "Type": "Weekend" if is_weekend else "Weekday",
        })
    return pd.DataFrame(records)

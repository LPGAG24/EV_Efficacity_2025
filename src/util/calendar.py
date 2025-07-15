import pandas as pd
from datetime import date


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

    # Basic list of federal holidays (approximation)
    hol = {
        date(year, 1, 1),   # New Year's Day
        date(year, 7, 1),   # Canada Day
        date(year, 12, 25), # Christmas Day
    }
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

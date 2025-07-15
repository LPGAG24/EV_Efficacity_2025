import pandas as pd
from util.calendar import build_calendar


def test_calendar_length():
    df = build_calendar(2022)
    assert len(df) == 365 or len(df) == 366
    assert set(df.columns) == {"Date", "Day", "Type"}


def test_holiday_is_weekend():
    df = build_calendar(2022)
    jan1 = df[df["Date"] == pd.to_datetime("2022-01-01").date()].iloc[0]
    assert jan1["Type"] == "Weekend"



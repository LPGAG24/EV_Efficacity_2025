import pandas as pd
import numpy as np
from datetime import date

import pytest

try:
    from carRecharge import CarRecharge
except Exception:  # scipy missing
    CarRecharge = None

from carDistribution import CarDistribution
from util.calculator import validate_home_work_percent
from util.calendar_utils import generate_calendar


def sample_distribution_df():
    return pd.DataFrame({
        "Province": ["Ontario", "Ontario", "Ontario"],
        "Vehicle Type": [
            "Total, road motor vehicle registrations",
            "Subcompact",
            "Pickup truck",
        ],
        "Fuel Type": ["All fuel types", "All fuel types", "All fuel types"],
        "Vehicles nb": [100, 40, 60],
    })


@pytest.mark.skipif(CarRecharge is None, reason="scipy is required")
def test_get_weekend_profile():
    cr = CarRecharge(apply_defaults=True)
    df = cr.get_weekend_profile()
    assert set(df.columns) == {"DayType", "Hour", "ChargingPerc"}
    assert sorted(df["DayType"].unique()) == ["Weekday", "Weekend"]
    sums = df.groupby("DayType")["ChargingPerc"].sum()
    assert np.isclose(sums.loc["Weekday"], 1.0)
    assert np.isclose(sums.loc["Weekend"], 1.0)


def test_validate_home_work_percent():
    h, w, c = validate_home_work_percent(60, 50)
    assert np.isclose(h + w + c, 1.0)
    assert h < 0.6 and w < 0.5


def test_generate_calendar():
    cal = generate_calendar(date(2023, 1, 1), date(2023, 1, 3))
    assert len(cal) == 3
    assert cal.iloc[0]["DayType"] == "Weekend"


def test_vehicle_count():
    dist = CarDistribution(sample_distribution_df(), Province="Ontario")
    counts = dist.get_vehicle_count("Ontario")
    sub = counts[counts["Vehicle Type"] == "Subcompact"]["Count"].iloc[0]
    assert sub == 40

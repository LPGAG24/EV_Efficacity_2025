import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep_canada import km_to_kwh
from carDistribution import CarDistribution


def test_km_to_kwh():
    assert km_to_kwh(100) == 18
    assert km_to_kwh(50) == 9


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


def test_getitem_tuple():
    dist = CarDistribution(sample_distribution_df(), Province="Ontario")
    res = dist["Ontario", "Subcompact", "All fuel types"]
    assert len(res) == 1
    assert res.iloc[0]["Vehicles nb"] == 40

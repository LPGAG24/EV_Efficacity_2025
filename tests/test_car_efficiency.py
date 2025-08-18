import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carEfficiency import CarEfficiency


def build_sample_data():
    return pd.DataFrame(
        {
            "Vehicle class": ["Compact", "Subcompact"],
            "Combined (Le/100 km)": [
                "5.0 (20 kWh/100 km)",
                "4.0 (16 kWh/100 km)",
            ],
            "Range (km)": [400, 300],
            "Combined (kWh/100 km)": [20, 16],
        }
    )


def test_month_efficiency_multiplier():
    data = build_sample_data()
    ce = CarEfficiency(data, month_coeffs={2: 1.5})
    base = ce.get_efficiency_by_type()
    feb = ce.get_efficiency_by_type(month=2)
    col = base.columns[1]
    assert (feb[col] == base[col] * 1.5).all()


def test_month_mean_efficiency():
    data = build_sample_data()
    ce = CarEfficiency(data, month_coeffs={2: 1.5})
    summer_mean = ce.get_mean_efficiency()
    feb_mean = ce.get_mean_efficiency(month=2)
    assert feb_mean == summer_mean * 1.5

import os
import sys
import pandas as pd

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from util.calculator import calculate_grid_power


def test_weekly_energy_for_200k_quebec_cars():
    car_count = 200_000
    efficiency = 20.0  # kWh/100km typical efficiency
    distance = 40.0    # km driven per day
    try:
        from carRecharge import CarRecharge
        profile = CarRecharge().get_weekly_profile()
    except Exception:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        records = []
        for day in days:
            for hour in range(24):
                records.append({"Day": day, "Hour": hour, "ChargingPerc": 1/24})
        profile = pd.DataFrame(records)
    total_energy = calculate_grid_power(
        car_count,
        efficiency,
        distance,
        charging_profile=profile,
    )
    expected_daily = car_count * efficiency * distance / 100.0
    expected_weekly = expected_daily * 7
    assert total_energy == expected_weekly

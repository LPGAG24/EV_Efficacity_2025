# validate_ev_model.py
import os
import sys
import pandas as pd
import numpy as np

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from carUsage import CarUsage
from carEfficiency import CarEfficiency
from carRecharge import CarRecharge
from carDistribution import CarDistribution
from data_prep_canada import fetch_statcan_fleet, download_ckan_resource
from util.calculator import calculate_grid_power

def check_range(name, value, low, high):
    """Return True if value within [low, high], else False"""
    return low <= value <= high

def main():
    print("=== EV_Efficacity_2025 Validation ===")
    pass_count = 0
    fail_count = 0

    # --- 1) Daily distance check ---
    cu = CarUsage()
    cu.fetchData()
    avg_distances = cu.average_daily_distance()

    print("\n[Daily Distance Validation]")
    for day, dist in avg_distances.items():
        ok = check_range(day, dist, 15, 70)
        print(f"{day}: {dist:.1f} km/day -> {'PASS' if ok else 'FAIL'}")
        pass_count += ok
        fail_count += not ok

    # --- 2) Efficiency & battery check ---
    df_ev = download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9")
    ce = CarEfficiency(df_ev)
    ce.set_efficiency_by_type()
    ce.set_battery_by_type()

    print("\n[EV Efficiency Validation]")
    for _, row in ce.efficiency_by_vehicle_type.iterrows():
        consumption = row.iloc[1]
        ok = check_range(row["Vehicle class"], consumption, 12, 30)
        print(f"{row['Vehicle class']}: {consumption:.1f} kWh/100 km -> {'PASS' if ok else 'FAIL'}")
        pass_count += ok
        fail_count += not ok

    print("\n[Battery Size Validation]")
    for _, row in ce.battery_by_vehicle_type.iterrows():
        ok = check_range(row["Vehicle class"], row["Battery_kWh"], 20, 120)
        print(f"{row['Vehicle class']}: {row['Battery_kWh']:.1f} kWh -> {'PASS' if ok else 'FAIL'}")
        pass_count += ok
        fail_count += not ok

    # --- 3) Charging behaviour check ---
    cr = CarRecharge()
    print("\n[Charging Stats Validation]")
    for source in ["Residential", "Public"]:
        e = np.mean([cr.sample_energy_kwh(source) for _ in range(1000)])
        d = np.mean([cr.sample_charging_duration_h(source) for _ in range(1000)])
        f = np.mean([cr.sample_frequency_per_day(source) for _ in range(1000)])

        ok_e = check_range(source, e, 5, 30)
        ok_d = check_range(source, d, 0.5, 4)
        ok_f = check_range(source, f, 0, 2)

        print(f"{source} energy: {e:.1f} kWh -> {'PASS' if ok_e else 'FAIL'}")
        print(f"{source} duration: {d:.2f} h -> {'PASS' if ok_d else 'FAIL'}")
        print(f"{source} freq/day: {f:.2f} -> {'PASS' if ok_f else 'FAIL'}")

        pass_count += ok_e + ok_d + ok_f
        fail_count += (not ok_e) + (not ok_d) + (not ok_f)

    # --- 4) Fleet distribution sanity check ---
    fleet_df = fetch_statcan_fleet()
    cd = CarDistribution(fleet_df, Province="Canada")
    print("\n[Fleet Distribution]")
    print(cd.get_fuel_type())

    # --- 5) Test: weekly energy for 200k Quebec cars ---
    print("\n[Weekly Energy Test for 200k Quebec Cars]")
    car_count = 200_000
    efficiency = 20.0  # kWh/100 km typical efficiency
    distance = 40.0    # km/day
    try:
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
    ok_weekly = total_energy == expected_weekly
    print(f"Weekly energy: {total_energy:.1f} kWh -> {'PASS' if ok_weekly else 'FAIL'}")
    pass_count += ok_weekly
    fail_count += not ok_weekly

    # --- Final realism score ---
    total_tests = pass_count + fail_count
    score = (pass_count / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nRealism score: {score:.1f}% ({pass_count} PASS / {total_tests} total checks)")

if __name__ == "__main__":
    main()

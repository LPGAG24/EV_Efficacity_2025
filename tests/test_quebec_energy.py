# validate_ev_model.py
import numpy as np
from carUsage import CarUsage
from carEfficiency import CarEfficiency
from carRecharge import CarRecharge
from carDistribution import CarDistribution
from data_prep_canada import fetch_statcan_fleet, download_ckan_resource

def check_range(name, value, low, high):
    """Return True if value within [low, high], else False"""
    return low <= value <= high

def main():
    print("=== EV_Efficacity_2025 Validation ===")

    # --- 1) Daily distance check ---
    cu = CarUsage()
    cu.fetchData()
    avg_distances = cu.average_daily_distance()

    print("\n[Daily Distance Validation]")
    for day, dist in avg_distances.items():
        ok = check_range(day, dist, 15, 70)  # 15–70 km/day realistic range
        print(f"{day}: {dist:.1f} km/day -> {'PASS' if ok else 'FAIL'}")

    # --- 2) Efficiency & battery check ---
    df_ev = download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9")
    ce = CarEfficiency(df_ev)
    ce.set_efficiency_by_type()
    ce.set_battery_by_type()

    print("\n[EV Efficiency Validation]")
    for _, row in ce.efficiency_by_vehicle_type.iterrows():
        consumption = row.iloc[1]  # second column is kWh/100 km
        ok = check_range(row["Vehicle class"], consumption, 12, 30)
        print(f"{row['Vehicle class']}: {consumption:.1f} kWh/100 km -> {'PASS' if ok else 'FAIL'}")

    print("\n[Battery Size Validation]")
    for _, row in ce.battery_by_vehicle_type.iterrows():
        ok = check_range(row["Vehicle class"], row["Battery_kWh"], 20, 120)
        print(f"{row['Vehicle class']}: {row['Battery_kWh']:.1f} kWh -> {'PASS' if ok else 'FAIL'}")

    # --- 3) Charging behaviour check ---
    cr = CarRecharge()
    print("\n[Charging Stats Validation]")
    for source in ["Residential", "Public"]:
        e = np.mean([cr.sample_energy_kwh(source) for _ in range(1000)])
        d = np.mean([cr.sample_charging_duration_h(source) for _ in range(1000)])
        f = np.mean([cr.sample_frequency_per_day(source) for _ in range(1000)])

        ok_e = check_range(source, e, 5, 30)   # kWh/session
        ok_d = check_range(source, d, 0.5, 4)  # hours
        ok_f = check_range(source, f, 0, 2)    # sessions/day

        print(f"{source} energy: {e:.1f} kWh -> {'PASS' if ok_e else 'FAIL'}")
        print(f"{source} duration: {d:.2f} h -> {'PASS' if ok_d else 'FAIL'}")
        print(f"{source} freq/day: {f:.2f} -> {'PASS' if ok_f else 'FAIL'}")

    # --- 4) Fleet distribution sanity check ---
    fleet_df = fetch_statcan_fleet()
    cd = CarDistribution(fleet_df, Province="Canada")
    print("\n[Fleet Distribution]")
    print(cd.get_fuel_type())

if __name__ == "__main__":
    main()

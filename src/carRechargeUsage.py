import pandas as pd
import numpy as np
import math
import re

"""EV Charging and Usage Simulation Module

Combines vehicle usage (daily distances, recharge needs) with charging profiles
(public, residential, DCFC) based on probabilistic time-of-day distributions.
"""

from carUsage import CarUsage
from carRecharge import CarRecharge
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -- Simulate usage data
    usage_data = pd.DataFrame({
        "Day": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        "Distance_km": [30, 45, 50, 40, 35, 20, 15]
    })
    cu = CarUsage()
    cu.set_data(usage_data)
    avg_dist = cu.average_daily_distance()
    recharge_kwh = cu.recharge_needed(efficiency_wh_per_km=150.0)
    print("Average distances (km):", avg_dist)
    print("Recharge needed (kWh):", recharge_kwh)

    # -- Define MDPI-derived peaks and scaling factors
    peaks_public = [(7.5, 25), (8.5, 0.25), (12, 0.25), (13.5, 0.20)]
    peaks_residential = [(17, 0.25), (18, 0.25)]
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekends = ["Saturday","Sunday"]
    # scaling factors from MDPI: public weekend ~40% of weekday; residential weekday ~80% of weekend
    public_weekend_scale = 0.4
    residential_weekday_scale = 0.8

    # -- Generate profiles
    cr = CarRecharge(usage_data)
    # Public weekday
    cr.set_car_charging_prop(day=weekdays, peaks=peaks_public, base=0.01, sigma=2.0)
    # Public weekend (scaled)
    scaled_public_peaks = [(h, amp * public_weekend_scale) for h, amp in peaks_public]
    cr.set_car_charging_prop(day=weekends, peaks=scaled_public_peaks, base=0.01, sigma=2.0)
    df_pub = cr.get_charging_profile_df()

    # Residential weekend
    cr.set_car_charging_prop(day=weekends, peaks=peaks_residential, base=0.01, sigma=2.0)
    # Residential weekday (scaled)
    scaled_res_peaks = [(h, amp * residential_weekday_scale) for h, amp in peaks_residential]
    cr.set_car_charging_prop(day=weekdays, peaks=scaled_res_peaks, base=0.01, sigma=2.0)
    df_res = cr.get_charging_profile_df()

    # -- Aggregate average per hour
    avg_pub_weekday = df_pub[df_pub['Day'].isin(weekdays)].groupby('Hour')['ChargingPerc'].mean()
    avg_pub_weekend = df_pub[df_pub['Day'].isin(weekends)].groupby('Hour')['ChargingPerc'].mean()
    avg_res_weekend = df_res[df_res['Day'].isin(weekends)].groupby('Hour')['ChargingPerc'].mean()
    avg_res_weekday = df_res[df_res['Day'].isin(weekdays)].groupby('Hour')['ChargingPerc'].mean()

    # -- Plotting
    plt.figure(figsize=(10,6))
    plt.plot(avg_pub_weekday.index, avg_pub_weekday.values, label='Public (Weekdays)')
    plt.plot(avg_pub_weekend.index, avg_pub_weekend.values, label='Public (Weekends)')
    plt.plot(avg_res_weekday.index, avg_res_weekday.values, '--', label='Residential (Weekdays)')
    plt.plot(avg_res_weekend.index, avg_res_weekend.values, '--', label='Residential (Weekends)')
    plt.xlabel("Hour of Day")
    plt.ylabel("Charging Probability")
    plt.title("EV Charging Profiles: Public vs Residential Across Week/Weekend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

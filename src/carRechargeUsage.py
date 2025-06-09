import pandas as pd
import numpy as np
import math
import re

"""
EV Charging and Usage Simulation Module

Combines vehicle usage (daily distances, recharge needs) with charging profiles
(public, residential, DCFC) based on probabilistic time-of-day distributions.
"""

class CarUsage:
    """Analyze vehicle usage to compute average daily distances and recharge energy needs."""

    def __init__(self):
        self.data: pd.DataFrame = None
        self.weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends = ["Saturday", "Sunday"]
        self.days: dict[str, pd.DataFrame] = {}

    def set_data(self, data: pd.DataFrame) -> None:
        """Load raw usage data (columns: Day, Distance_km)."""
        self.data = data
        for day in self.weekdays + self.weekends:
            self.days[day] = self.data[self.data["Day"] == day]

    def average_daily_distance(self) -> dict[str, float]:
        """Return {day: average distance traveled (km)}."""
        return {day: (df["Distance_km"].sum() / len(df) if len(df) else 0.0)
                for day, df in self.days.items()}

    def recharge_needed(self, efficiency_wh_per_km: float) -> dict[str, float]:
        """Compute daily recharge need (kWh) given efficiency (Wh/km)."""
        avg = self.average_daily_distance()
        return {day: (dist * efficiency_wh_per_km / 1000.0)
                for day, dist in avg.items()}

    def fetch_data_nrcan(self, year: int = 2022) -> pd.DataFrame:
        """Fetch NRCAN table data and produce DataFrame of annual km by province."""
        # simplified: user can integrate their existing fetchData logic here
        raise NotImplementedError("Use customized fetchData logic to load NRCan data.")


class CarRecharge:
    """Generate charging probability profiles for each hour of each day."""

    CHARGER_SPEED = {"Public": {"Level 2": {"mean": 8.83, "STD": 8.04, "Time": 3}},  # kW
                     "Residential": {"Level 2": {"mean": 12, "STD": 10.28, "Time": 12}},
                     "DCFC": {"mean": 150.0}}  # kW

    def __init__(self, data: pd.DataFrame):
        # data: DataFrame with columns [Day, any] to infer days
        self.weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends = ["Saturday", "Sunday"]
        self.charging_profile = {
            day: [0.0] * 24 for day in self.weekdays + self.weekends
        }

    def set_car_charging_perc(self, day: str = None, charging=None):
        """Set explicit hourly charging percentages."""
        # existing logic from CarRecharge.set_car_charging_perc...
        if isinstance(charging, dict):
            mapping = charging
        elif isinstance(charging, list):
            if len(charging) == 24 and all(isinstance(x, (int, float)) for x in charging):
                days = [day] if day else self.weekdays + self.weekends
                for d in days:
                    self.charging_profile[d] = list(charging)
                return
            elif all(isinstance(x, (tuple, list)) and len(x) == 2 for x in charging):
                mapping = {int(h): float(p) for h, p in charging}
            else:
                raise ValueError("If `charging` is a list it must be either 24 floats or a list of (hour, percent) pairs")
        else:
            raise ValueError("charging must be a dict or list")

        profile = [0.0] * 24
        for h, pct in mapping.items():
            if not 0 <= h < 24: raise ValueError(f"Hour must be 0–23; got {h}")
            if not 0.0 <= pct <= 1.0: raise ValueError(f"Percent must be 0.0–1.0; got {pct}")
            profile[h] = pct

        days = [day] if day else self.weekdays + self.weekends
        for d in days:
            self.charging_profile[d] = profile.copy()

    def set_car_charging_prop(self, day: str = None, peaks: list[tuple[int, float]] = None,
                               base: float = 0.02, sigma: float = 2.0):
        """Build Gaussian-based 24h charging profile with peaks and base level."""
        if not peaks:
            raise ValueError("`peaks` must be provided as list of (hour, weight) tuples")
        charging = [base] * 24
        for center, amp in peaks:
            for h in range(24):
                dx = h - center
                charging[h] += amp * math.exp(-0.5 * (dx * dx) / (sigma * sigma))
        # cap and normalize
        charging = [min(1.0, c) for c in charging]
        total = sum(charging)
        charging = [c / total for c in charging]

        days = day if day else self.weekdays + self.weekends
        for d in days:
            self.charging_profile[d] = charging.copy()

    def get_charging_profile_df(self) -> pd.DataFrame:
        """Return DataFrame with columns [Day, Hour, ChargingPerc]."""
        records = []
        for day, profile in self.charging_profile.items():
            for hr, pct in enumerate(profile):
                records.append({"Day": day, "Hour": hr, "ChargingPerc": pct})
        return pd.DataFrame(records)
    


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
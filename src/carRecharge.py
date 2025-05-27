import pandas as pd
import numpy as np
import math


"""
carRecharge:
    - A class for analyzing vehicle usage data for weekday and weekends.

"""
class CarRecharge:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.days: dict[str, pd.DataFrame] = {}
        self.weekdays: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends: list[str] = ["Saturday", "Sunday"]
        for day in self.weekdays + self.weekends:
            self.days[day] = self.data[self.data["Day"] == day]
        # Charging profile: dict like {"Monday": [0.1, 0.05, ..., 0.2] }
        self.charging_profile: dict[str, list[float]] = {day: [0.0]*24 for day in self.weekdays + self.weekends}


    def set_car_charging_perc(self, day: str = None, charging=None):
        """
        Set percentage of cars charging for each hour.
        
        charging may be:
          - a dict {hour: percent, ...}
          - a list of (hour, percent) tuples
          - a full 24-element list of floats
        Any hours you don’t specify get 0.0.
        """
        if isinstance(charging, dict):
            mapping = charging
        elif isinstance(charging, list):
            if len(charging) == 24 and all(isinstance(x, (int, float)) for x in charging):
                days = [day] if day else self.weekdays + self.weekends
                for d in days:
                    self.charging_profile[d] = list(charging)
                return
            # otherwise expect list of (hour, percent)
            elif all(isinstance(x, (tuple, list)) and len(x) == 2 for x in charging):
                mapping = {int(h): float(p) for h, p in charging}
            else:
                raise ValueError("If `charging` is a list it must be either 24 floats or a list of (hour, percent) pairs")
        else:
            raise ValueError("charging must be a dict or list")

        profile = [0.0] * 24
        for h, pct in mapping.items():
            h = int(h)
            if not (0 <= h < 24): raise ValueError(f"Hour must be in 0–23; got {h}")
            if not (0.0 <= pct <= 1.0): raise ValueError(f"Percent must be 0.0–1.0; got {pct}")
            profile[h] = pct

        days = [day] if day else self.weekdays + self.weekends
        for d in days:
            self.charging_profile[d] = profile.copy()


    def set_car_charging_prop(self,
                               day: str = None,
                               peaks: list[tuple[int, float]] = None,
                               base: float = 0.02,
                               sigma: float = 2.0):
        """
        Build a 24-hour charging profile by placing Gaussian peaks at given hours.

        peaks:  list of (hour, amplitude) pairs, e.g. [(8,0.25),(12,0.25),(18,0.5)]
        base:   the flat minimum probability at all hours outside the peaks
        sigma:  standard deviation (in hours) of each Gaussian bell

        Any hours not under a Gaussian get 'base'; the final profile is capped at 1.0.
        """
        if not peaks or not all(isinstance(h, int) and 0 <= h < 24 for h, _ in peaks):
            raise ValueError("`peaks` must be list of (hour 0–23, weight) tuples")

        charging = [base] * 24

        for center, amp in peaks:
            for h in range(24):
                dx = h - center
                charging[h] += amp * math.exp(-0.5 * (dx*dx) / (sigma*sigma))

        charging = [min(1.0, c) for c in charging]

        #make sure sum of charging does not surpass 1.0
        charging = [c / sum(charging) for c in charging]

        days = [day] if day else self.weekdays + self.weekends
        for d in days:
            self.charging_profile[d] = charging.copy()


    def get_charging_profile_df(self):
        """Return a DataFrame with day/hour as index and charging percentage as value."""
        records = []
        for day, profile in self.charging_profile.items():
            for hour, perc in enumerate(profile):
                records.append({"Day": day, "Hour": hour, "ChargingPerc": perc})
        return pd.DataFrame(records)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    data = pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Distance_km": [10, 15, 20, 25, 30, 5, 10]
    })
    
    car_recharge = CarRecharge(data)
    car_recharge.set_car_charging_perc(day="Monday", charging=[0.1]*24)
    print(car_recharge.get_charging_profile_df())
    
    car_recharge.set_car_charging_prop(peaks=[(8, 0.25), (12, 0.25), (18, 0.5)], base=0.02)
    print(car_recharge.get_charging_profile_df())

    # Plotting the charging profile for Monday
    profile_df = car_recharge.get_charging_profile_df()
    monday_profile = profile_df[profile_df["Day"] == "Monday"]
    plt.bar(monday_profile["Hour"], monday_profile["ChargingPerc"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Charging Percentage")
    plt.title("Charging Profile for Monday")
    plt.show()

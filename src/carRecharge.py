import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

class CarRecharge:
    """Generate simple charging probability profiles."""

    def __init__(self) -> None:
        self.weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends = ["Saturday", "Sunday"]
        self.charging_profile = {d: [0.0] * 24 for d in self.weekdays + self.weekends}

    def set_car_charging_prop(self, *, day, peaks, base: float = 0.02, sigma: float = 2.0):
        profile = [base] * 24
        for mu, amp in peaks:
            for h in range(24):
                profile[h] += amp * math.exp(-((h - mu) ** 2) / (2 * sigma * sigma))
        s = sum(profile)
        profile = [p / s for p in profile]
        days = day if isinstance(day, list) else [day]
        for d in days:
            self.charging_profile[d] = profile.copy()

    def get_weekly_profile(self) -> pd.DataFrame:
        records = []
        for d, prof in self.charging_profile.items():
            for h, p in enumerate(prof):
                records.append({"Day": d, "Hour": h, "ChargingPerc": p})
        return pd.DataFrame(records)

    def get_30min_profile(self, day: str) -> np.ndarray:
        arr = np.array(self.charging_profile[day])
        x = np.arange(24)
        f = interp1d(x, arr, kind="linear")
        t = np.linspace(0, 23.5, 48)
        y = f(t)
        y[y < 0] = 0
        return y / y.sum()

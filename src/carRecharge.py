import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from typing import Union

class CarRecharge:
    """EV charging‑behaviour model with statistical sampling (MDPI‑based).

    Default peak‑hour definitions and weekday/weekend scaling are baked‑in so
    that a user can spin up the object *without* having to re‑enter those
    values every time.
    """

    # ──────────────────────────────────────────────────────────────────
    # 0)  CONSTANTS — power, MDPI stats, default peaks & ratios
    # ──────────────────────────────────────────────────────────────────
    CHARGER_SPEED = {
        "Public":      {"Level 2": {"mean":  8.83, "STD":  8.04}},
        "Residential": {"Level 2": {"mean": 12.06, "STD": 10.28}},
        "DCFC":        {"mean": 150.0}
    }

    _ENERGY_STATS      = {"Residential": (12.06, 10.28), "Public": ( 8.83,  8.04)}
    _DURATION_STATS    = {"Residential": (156.61, 115.64), "Public": (144.56, 119.03)}  # minutes
    _FREQ_POISSON_LAMB = {"Residential": 0.73,            "Public": 0.63}

    # ‑‑ Default Gaussian peak specifications (µ [h], amplitude) ‑‑
    _PEAKS = {
        "Public_Weekday"          : [(7.5, 0.25), (8.5, 0.25), (12.0, 0.25), (13.5, 0.20)],
        "Residential_Weekend"     : [(17.0, 0.25), (18.0, 0.25)],
        "Public_Weekend_Scale"    : 0.4,   # 40 % of weekday public demand
        "Residential_Weekday_Scale": 0.8    # 80 % of weekend residential demand
    }

    # ──────────────────────────────────────────────────────────────────
    # 1)  INITIALISATION
    # ──────────────────────────────────────────────────────────────────
    def __init__(self, source_df: pd.DataFrame | None = None, *, apply_defaults: bool = True):
        """Parameters
        ----------
        source_df       : optional DataFrame to attach (e.g., pre‑existing events)
        apply_defaults  : if *True*, builds default weekday/weekend profiles
                          based on `_PEAKS`; set to False if you prefer to
                          construct the profiles manually with
                          `set_car_charging_prop()`.
        """
        self.data      = source_df
        self.weekdays  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends  = ["Saturday", "Sunday"]
        self.charging_profile: dict[str, list[float]] = {d: [0.0]*24 for d in self.weekdays + self.weekends}

        if apply_defaults:
            self._build_default_profiles()

    # ──────────────────────────────────────────────────────────────────
    # 2)  DEFAULT‑PROFILE BUILDER
    # ──────────────────────────────────────────────────────────────────
    def _build_default_profiles(self) -> None:
        """Populate `self.charging_profile` with the built‑in peaks/ratios."""
        # Public – weekday
        self.set_car_charging_prop(day=self.weekdays,
                                   peaks=self._PEAKS["Public_Weekday"])
        # Public – weekend (scaled)
        scale_pub = self._PEAKS["Public_Weekend_Scale"]
        peaks_pub_we = [(h, a*scale_pub) for h, a in self._PEAKS["Public_Weekday"]]
        self.set_car_charging_prop(day=self.weekends, peaks=peaks_pub_we)

        # Residential – weekend
        self.set_car_charging_prop(day=self.weekends,
                                   peaks=self._PEAKS["Residential_Weekend"])
        # Residential – weekday (scaled)
        scale_res = self._PEAKS["Residential_Weekday_Scale"]
        peaks_res_wd = [(h, a*scale_res) for h, a in self._PEAKS["Residential_Weekend"]]
        self.set_car_charging_prop(day=self.weekdays, peaks=peaks_res_wd)

    # ──────────────────────────────────────────────────────────────────
    # 3)  PROFILE GENERATION (manual or default)
    # ──────────────────────────────────────────────────────────────────
    def set_car_charging_prop(self, *, day: list[str], peaks: list[tuple[float, float]],
                               base: float = 0.01, sigma: float = 2.0) -> None:
        """Create a 24‑value Gaussian‑mixture probability profile for *day*."""
        prof = [base]*24
        for mu, amp in peaks:
            for h in range(24):
                prof[h] += amp * math.exp(-((h - mu)**2)/(2*sigma*sigma))
        s = sum(prof)
        prof = [p/s for p in prof]
        for d in day:
            self.charging_profile[d] = prof.copy()

    # ──────────────────────────────────────────────────────────────────
    # 4)  PROFILE QUERIES (24 h → 48 × 30 min)
    # ──────────────────────────────────────────────────────────────────
    
    def get_weekly_profile(self) -> pd.DataFrame:
        """Return a DataFrame with columns [Day, Hour, ChargingPerc]."""
        records = []
        for day, profile in self.charging_profile.items():
            for hour, perc in enumerate(profile):
                records.append({"Day": day, "Hour": hour, "ChargingPerc": perc})
        return pd.DataFrame(records)

    def get_weekend_profile(self) -> pd.DataFrame:
        """Return aggregated weekday and weekend charging profiles."""
        df = self.get_weekly_profile()
        df["DayType"] = np.where(df["Day"].isin(self.weekends), "Weekend", "Weekday")
        grouped = (
            df.groupby(["DayType", "Hour"], as_index=False)["ChargingPerc"].mean()
        )
        return grouped
    
    def get_hourly_profile(self, day: str) -> list[float]:
        return self.charging_profile[day]

    def get_30min_profile(self, day: str) -> np.ndarray:
        y24 = np.array(self.charging_profile[day])
        t24 = np.arange(24)
        f   = interp1d(t24, y24, kind="linear", fill_value="extrapolate")
        t48 = np.linspace(0, 23.5, 48)
        y48 = np.maximum(f(t48), 0)
        return y48 / y48.sum()

    # ──────────────────────────────────────────────────────────────────
    # 5)  STATISTICAL SAMPLERS (MDPI Table‑2)
    # ──────────────────────────────────────────────────────────────────
    def sample_energy_kwh(self, source: str = "Residential") -> float:
        mu, sig = self._ENERGY_STATS[source]
        return max(0.0, np.random.normal(mu, sig))

    def sample_charging_duration_h(self, source: str = "Residential") -> float:
        mu, sig = self._DURATION_STATS[source]
        return max(0.01, np.random.normal(mu, sig)/60.0)

    def sample_frequency_per_day(self, source: str = "Residential") -> int:
        return np.random.poisson(self._FREQ_POISSON_LAMB[source])

    # ──────────────────────────────────────────────────────────────────
    # 6)  PROFILE SCALING (extra utility)
    # ──────────────────────────────────────────────────────────────────
    def scale_profile(self, days: list[str], factor: float) -> None:
        for d in days:
            scaled = [min(1.0, v*factor) for v in self.charging_profile[d]]
            s      = sum(scaled)
            self.charging_profile[d] = [v/s for v in scaled]

    # ──────────────────────────────────────────────────────────────────
    # 7)  EVENT SIMULATION
    # ──────────────────────────────────────────────────────────────────
    def _pick_slots(self, prob48: np.ndarray, n: int) -> np.ndarray:
        return np.random.choice(np.arange(48), size=n, p=prob48)

    def simulate(self, *, car_usage, efficiency_wh_per_km: float = 150.0,
                 start_date: datetime = None, n_days: int = 7,
                 source: str = "Residential") -> pd.DataFrame:
        if start_date is None:
            start_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        records: list[dict] = []
        for i in range(n_days):
            day_date = start_date + timedelta(days=i)
            day_name = day_date.strftime("%A")
            n_sessions = self.sample_frequency_per_day(source)
            if n_sessions == 0:  # skip if no sessions today
                continue
            prob48 = self.get_30min_profile(day_name)
            slots  = self._pick_slots(prob48, n_sessions)
            for slot in slots:
                ts  = day_date + timedelta(minutes=30*slot)
                records.append({
                    "Timestamp"  : ts,
                    "Day"        : day_name,
                    "Source"     : source,
                    "Energy_kWh" : self.sample_energy_kwh(source),
                    "Duration_h" : self.sample_charging_duration_h(source)
                })

        self.data = pd.DataFrame(records).sort_values("Timestamp").reset_index(drop=True)
        return self.data
    

def gaussian_profile(
    mu: float,  sigma_left: Union[int, float],
    n: int,     sigma_right: Union[int, float] = 0,
) -> np.ndarray:
    """
    Profil gaussien circulaire (24 h) éventuellement asymétrique.

    Parameters
    ----------
    mu : float
        Heure du pic (0 ≤ mu < 24). Toute valeur hors plage est repliée modulo 24.
    sigma_left : float
        Écart type, en heures, pour la partie *avant* le pic (côté gauche).
    n : int
        Nombre de cases (bins) sur 24 heures.
    sigma_right : float, optional
        Écart type pour la partie *après* le pic (côté droit).  
        Si 0 (valeur par défaut), on prend `sigma_right = sigma_left`
        ➜ profil symétrique.

    Returns
    -------
    np.ndarray
        Tableau de longueur ``n`` dont la somme vaut 1.

    Notes
    -----
    * La distance circulaire signée se calcule avec ::

          delta = ((t - mu + 12) % 24) - 12    # ∈ (-12, 12]

      Δ < 0 → côté gauche · Δ ≥ 0 → côté droit.
    """

    mu = mu % 24
    if sigma_right == 0:
        sigma_right = sigma_left
    if sigma_left <= 0 or sigma_right <= 0:
        raise ValueError("sigma_left et sigma_right doivent être > 0")

    t = np.linspace(0, 24, n, endpoint=False)
    delta = ((t - mu + 12) % 24) - 12           # (-12, 12]
    sigma = np.where(delta < 0, sigma_left, sigma_right)
    prof = np.exp(-(delta**2) / (2 * sigma**2))
    return prof / prof.sum()

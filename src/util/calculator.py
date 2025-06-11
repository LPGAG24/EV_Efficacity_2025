import numpy as np
import pandas as pd


def calculate_grid_power(
    car_count: int,
    efficiency_kwh_per_100km: float,
    distance_km: float,
    charging_time: float = 8,
    charging_profile: pd.DataFrame | None = None,
) -> float:
    """Return the total energy required to charge a fleet of vehicles.

    Parameters
    ----------
    car_count : int
        Number of electric vehicles.
    efficiency_kwh_per_100km : float
        Vehicle efficiency in kWh/100km.
    distance_km : float
        Average distance driven per day in km.
    charging_time : float, optional
        Unused for now but kept for backward compatibility.
    charging_profile : pandas.DataFrame, optional
        DataFrame with columns ``Day`` and ``ChargingPerc`` describing how
        charging is distributed across days/hours. If provided, energy is
        scaled by the sum of ``ChargingPerc`` for each day.

    Returns
    -------
    float
        Total energy required in kWh.
    """

    total_energy_needed = (
        car_count * float(efficiency_kwh_per_100km) * distance_km / 100.0
    )

    if charging_profile is None:
        return float(total_energy_needed)

    if not isinstance(charging_profile, pd.DataFrame):
        raise TypeError("charging_profile must be a pandas DataFrame")

    if not {"Day", "ChargingPerc"}.issubset(charging_profile.columns):
        raise ValueError("charging_profile must contain 'Day' and 'ChargingPerc' columns")

    total_charging_power = 0.0
    for day in charging_profile["Day"].unique():
        perc_sum = charging_profile.loc[charging_profile["Day"] == day, "ChargingPerc"].sum()
        total_charging_power += perc_sum * total_energy_needed

    return float(total_charging_power)


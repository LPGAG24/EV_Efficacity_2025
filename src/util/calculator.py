import numpy as np
import pandas as pd

def calculate_grid_power(car_count: int, efficiency_kwh_per_100km: float, distance_km: float, charging_time: float = 8, charging_profile: list[float]=None) -> float:
    """
    Calculate the total grid power needed for charging electric vehicles.

    :param car_count: Number of electric vehicles.
    :param efficiency_kwh_per_100km: Energy efficiency of the vehicle in kWh/100km.
    :param distance_km: Average distance driven per day in km.
    :param charging_profile: List of charging percentages for each hour of the day.
    :return: Total grid power needed in kWh.
    """
    total_energy_needed = car_count * np.float32(efficiency_kwh_per_100km) * distance_km / 100.0
    for day in charging_profile["Day"].unique():
        if day not in charging_profile["Day"].unique():
            raise ValueError(f"Charging profile for {day} not found in the provided profile.")
        total_charging_power = sum(charging_profile[charging_profile["Day"] == day]["ChargingPerc"].values) * total_energy_needed
    return total_charging_power[0] if isinstance(total_charging_power, np.ndarray) else total_charging_power


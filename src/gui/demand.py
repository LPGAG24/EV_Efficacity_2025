import numpy as np
import pandas as pd

from appHelper import compute_time_bins, circular_convolve, aggregate_power


def compute_charging_demand(categories, car_count, n_res, recharge_time):
    """Return charging demand statistics for the given scenario.

    Parameters
    ----------
    categories : list[dict]
        Charging category definitions containing share, profile and speed.
    car_count : int
        Number of vehicles considered.
    n_res : int
        Number of time slots in a day.
    recharge_time : float
        Average recharge time in hours.

    Returns
    -------
    dict
        Dictionary with DataFrames and summary statistics used by the app.
    """
    time_bins, n_slots, slot_len = compute_time_bins(n_res, recharge_time)

    cars_df = pd.DataFrame({"Time": time_bins})
    level_power_df = pd.DataFrame({"Time": time_bins})

    arrivals_list: list[np.ndarray] = []
    kernels_list: list[np.ndarray] = []
    level_arrivals: dict[str, list[np.ndarray]] = {}
    level_kernels: dict[str, list[np.ndarray]] = {}

    for cat in categories:
        arrivals = cat["share"] * car_count * cat["profile"]
        cars_df[cat["label"]] = circular_convolve(arrivals, np.ones(n_slots))
        arrivals_list.append(arrivals)
        kernels_list.append(np.full(n_slots, cat["speed"]))
        ratios = cat.get("ratios", (1.0, 0.0, 0.0))
        for i, kw in enumerate(cat["level_kW"]):
            level = f"Level {i+1}"
            level_arrivals.setdefault(level, []).append(arrivals * ratios[i])
            level_kernels.setdefault(level, []).append(np.full(n_slots, kw))

    labels = [c["label"] for c in categories]
    if labels:
        cars_df["Total_cars"] = cars_df[labels].sum(axis=1)
        cars_df["Total_thousands"] = cars_df["Total_cars"] / 1000
    else:
        cars_df["Total_cars"] = 0.0
        cars_df["Total_thousands"] = 0.0

    if arrivals_list:
        arrivals_mat = np.column_stack(arrivals_list)
        kernels_mat = np.column_stack(kernels_list)
        total_power = aggregate_power(arrivals_mat, kernels_mat)
    else:
        total_power = np.zeros(n_res)

    for level, arr_list in level_arrivals.items():
        arr_mat = np.column_stack(arr_list)
        kern_mat = np.column_stack(level_kernels[level])
        level_power_df[level] = aggregate_power(arr_mat, kern_mat)

    level_cols = [c for c in level_power_df.columns if c.startswith("Level ")]
    if level_cols:
        level_power_df["Agg_kW"] = level_power_df[level_cols].sum(axis=1)
    else:
        level_power_df["Agg_kW"] = 0.0
    level_power_df["Agg_kW"] = total_power

    if labels:
        cars_long = cars_df.melt(
            id_vars="Time", value_vars=labels, var_name="Source", value_name="Cars"
        )
        cars_long["Cars_thousands"] = cars_long["Cars"] / 1000
    else:
        cars_long = pd.DataFrame(columns=["Time", "Source", "Cars", "Cars_thousands"])

    level_vars = [c for c in level_power_df.columns if c.startswith("Level ")]
    if level_vars:
        power_long = level_power_df.melt(
            id_vars="Time", value_vars=level_vars, var_name="Charger Level", value_name="kW"
        )
    else:
        power_long = pd.DataFrame(columns=["Time", "Charger Level", "kW"])

    mean_cars = float(cars_df["Total_cars"].mean()) if labels else 0.0
    daily_energy_wh = float((level_power_df["Agg_kW"] * slot_len).sum() * 1000)
    if len(level_power_df):
        max_idx = level_power_df["Agg_kW"].idxmax()
        max_power_w = float(level_power_df.loc[max_idx, "Agg_kW"] * 1000)
        max_time = level_power_df.loc[max_idx, "Time"]
    else:
        max_power_w = 0.0
        max_time = None

    return {
        "cars_df": cars_df,
        "cars_long": cars_long,
        "power_long": power_long,
        "level_power_df": level_power_df,
        "slot_len": slot_len,
        "mean_cars": mean_cars,
        "daily_energy_wh": daily_energy_wh,
        "max_power_w": max_power_w,
        "max_time": max_time,
    }

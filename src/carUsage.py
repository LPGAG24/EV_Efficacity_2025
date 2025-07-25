import pandas as pd
import numpy as np
import re
import math
import util.normalizer as util
try:
    import streamlit as st  # optional dependency
except ModuleNotFoundError:  # pragma: no cover - streamlit not required for tests
    st = None
try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional for headless tests
    plt = None

class CarUsage:
    """CarUsage class for analyzing vehicle usage data.
    Calculates average distance traveled and recharge energy needed per day.
    """
    def __init__(self):
        self.data: pd.DataFrame = None
        self.weekdays: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends: list[str] = ["Saturday", "Sunday"]
        self.days: dict[str, pd.DataFrame] = {}
        self.averages: dict[str, float] = {}
        self.rechargeNeeded: dict[str, float] = {}

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for the CarUsage class.
        Expects `data` to have columns ["Day", "Distance_km"].
        """
        self.data = data
        for day in self.weekdays + self.weekends:
            self.days[day] = self.data[self.data["Day"] == day]

    def average_daily_distance(self) -> dict[str, float]:
        """Return a dict of {day_name: avg distance traveled (km)}."""
        avg = {}
        for day, df in self.days.items():
            avg[day] = df["Distance_km"].sum() / len(df) if not df.empty else 0.0
        return avg
    
    def set_average(self, averages: list|float, day: str|list[str] = None):
        """Set the average distance for a specific day or all days."""
        if isinstance(averages, list):
            if day is None:
                for i, d in enumerate(self.weekdays + self.weekends): self.averages[d] = averages[i]
            else:
                for d in day: self.averages[d] = averages[self.weekdays + self.weekends.index(d)]
        elif isinstance(averages, float):
            if day is None:
                for d in self.weekdays + self.weekends: self.averages[d] = averages
            else:
                for d in day: self.averages[d] = averages
        else: raise ValueError("Averages must be a list or float")

    def recharge_needed(self, efficiency_wh_per_km: float) -> dict[str, float]:
        """
        Compute how many kWh you need to recharge to replace avg daily usage.

        efficiency_wh_per_km: energy consumed by the car per km (Wh/km)
        """
        avg = self.average_daily_distance()
        recharge = {}
        for day, dist in avg.items():
            # Wh needed = dist * efficiency; convert to kWh
            recharge[day] = dist * efficiency_wh_per_km / 1000.0
        return recharge
    
    def set_recharge_needed(self, efficiency_wh_per_km: float, day: str) -> None:
        self.rechargeNeeded[day] = self.recharge_needed(efficiency_wh_per_km)[day]
        self.averages[day] = self.average_daily_distance()[day]
        print(f"Average distance on {day}: {self.averages[day]:.2f} km")
        print(f"Recharge needed on {day}: {self.rechargeNeeded[day]:.2f} kWh")



    def fetchData(self) -> pd.DataFrame:
        # ---------- CONFIGURATION -----------------------------------
        YEAR          = 2022                 # latest panel on page=2
        PROVINCES     = ["on", "qc", "ab", "bct", "ns", "nl", "pe", "mb", "sk", "nb"]  # NRCan 'juris' codes
        WEEKDAY_SHARE = 0.80                 # CVS-2009: 80 % of km Mon-Fri
        WEEKEND_SHARE = 0.20
        WKDAYS_2022   = 261                  # Mon-Fri days in 2022
        WKENDS_2022   = 104                  # Sat+Sun days in 2022
        SIGMA_LN      = 0.9                 # generic σ from CVS micro-data
        URL_TPL = ("https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/"
                   "showTable.cfm?juris={prov}&page=2&rn=21&sector=tran&type=CP&year={yr}")
        URL_CAN = ("https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/"
                   "showTable.cfm?juris=ca&type=CP&sector=tran&rn=32%20&year={yr}&page=2")

        # ---------- HELPERS -----------------------------------------
        def fetch_table(prov: str, yr: int = YEAR, URL: str = URL_TPL) -> pd.DataFrame:
            """Return the raw one-column DataFrame for NRCan Table 21 page 2."""
            return pd.read_html(URL.format(prov=prov, yr=yr), header=None)[0]

        def annual_km(df: pd.DataFrame, yr: int = YEAR) -> int:
            """Extract 'Average Distance Travelled per Year (km)' as an int."""
            col = str(yr)
            raw_val =df[col].iloc[4].iloc[0]
            # strip commas / blanks and cast
            return int(re.sub(r"[^\d]", "", str(raw_val)))

        def daily_split(km_year: int) -> tuple[float, float]:
            """Return (weekday_km, weekend_km) per car per day."""
            wkday = km_year * WEEKDAY_SHARE / WKDAYS_2022
            wkend = km_year * WEEKEND_SHARE / WKENDS_2022
            return wkday, wkend

        def mu_from_mean(mean: float, sigma: float = SIGMA_LN) -> float:
            """Convert arithmetic mean to μ for a log-normal."""
            return np.log(mean) - 0.5 * sigma**2
        

        # ---------- MAIN --------------------------------------------
        records = []                    # ← collect rows here  (list-of-dicts)

        for prov in PROVINCES:          # e.g. ["qc", "on", …]
            df          = fetch_table(prov)
            km_yr       = annual_km(df)
            wkday_km, wkend_km = daily_split(km_yr)

            records.append({            # each dict becomes one row
                "Province":    prov.upper(),     # keep province as a COLUMN
                "Year":        YEAR,
                "Annual_km":   km_yr,
                "Weekday_km":  wkday_km,
                "Weekend_km":  wkend_km,
                "Log_weekday": mu_from_mean(wkday_km),
                "Log_weekend": mu_from_mean(wkend_km),
                "Sigma_ln":    SIGMA_LN,
            })

        # ---------- Canada-wide line (optional) ----------
        df_ca          = fetch_table("ca", yr=YEAR, URL=URL_CAN)
        km_yr          = annual_km(df_ca)
        wkday_km, wkend_km = daily_split(km_yr)

        records.append({
            "Province":   "CA",
            "Year":       YEAR,
            "Annual_km":  km_yr,
            "Weekday_km": wkday_km,
            "Weekend_km": wkend_km,
            "Log_weekday": mu_from_mean(wkday_km),
            "Log_weekend": mu_from_mean(wkend_km),
            "Sigma_ln":    SIGMA_LN,
        })

        # ---------- list-of-dicts → DataFrame ----------
        self.data = pd.DataFrame.from_records(records)

        self.data = util.normalize_dataframe(self.data)
        
        #print("Car usage data fetched and processed.")
        #print("Average distance traveled per year (km):")
        #for prov in self.data["Province"]:
        #    print(f"  {prov}: {self.data.loc[self.data['Province'] == prov, 'Annual_km'].values[0]}")
        #print("Average distance traveled per weekday (km):")
        #for prov in self.data["Province"]:
        #    print(f"  {prov}: {self.data.loc[self.data['Province'] == prov, 'Weekday_km'].values[0]}")
        #print("Average distance traveled per weekend (km):")
        #for prov in self.data["Province"]:
        #    print(f"  {prov}: {self.data.loc[self.data['Province'] == prov, 'Weekend_km'].values[0]}")
        #print("Done")
        
        
    def __getitem__(self, key) -> pd.DataFrame:
        """
        Flexible selector on province/year table.

        Accepted keys
        -------------
        • "QC"                                   → all rows for Quebec
        • ["QC","ON"] or slice(...)              → several provinces
        • ("QC", 2022)                           → province + year
        • {'Province':"QC", "Year":2022}         → arbitrary column=value filter
        • callable                               → lambda df: ... (power-user hook)
        """
        df = self.data

        # 1. single province (string)
        if isinstance(key, str):
            return df[df["Province"] == key]

        # 2. several provinces (list/tuple/slice)
        if isinstance(key, (list, slice)):
            return df[df["Province"].isin(df["Province"].unique()[key])]

        # 3. hierarchical tuple: (province, year)
        if isinstance(key, tuple):
            cols = ["Province", "Year"]
            mask = pd.Series(True, index=df.index)
            for col, val in zip(cols, key):
                mask &= df[col] == val
            return df[mask]

        # 4. free-form dict {column: value, …}
        if isinstance(key, dict):
            mask = pd.Series(True, index=df.index)
            for col, val in key.items():
                mask &= df[col] == val
            return df[mask]

        # 5. callable
        if callable(key): return key(df)

        raise TypeError("Key must be str, list/slice, tuple, dict, or callable.")
    
    
    # def get_daily_driver_counts(self):
    #     """Fetch and attach daily driver counts."""
    #     df = fetch_statcan_daily_drivers()
    #     self.daily_drivers = df[df["Province"] == "Canada"]["Drivers"]
    #     return self.daily_drivers



import warnings
import pandas as pd
from stats_can.sc import table_to_df
try:  # StatsCan <3 compatibility
    from stats_can.sc import vectors_to_df_remote
except ImportError:  # pragma: no cover - old stats_can
    from stats_can.sc import vectors_to_df as vectors_to_df_remote
import pandas as pd
warnings.filterwarnings(
    "ignore",
    message=r"This function will be deprecated in the v3 release",
    category=FutureWarning,
    module=r"stats_can.sc",
)

# def fetch_statcan_distance():
#     """Fetch commute distance distribution table."""
#     t_part = table_to_df("45-10-0104-03")
#     return t_part

# def fetch_statcan(link: str) -> pd.DataFrame:
#     """Fetch a StatCan table by link, return DataFrame."""
#     tbl = table_to_df(link)
#     if tbl.empty:
#         raise ValueError(f"No data found for {link}")
#     return tbl




# ---- Daily private vehicle time distribution -------------------------------
# Source counts (number of people by one-way commute duration class)
COUNTS = {
    "<15": 4_178_570,
    "15-29": 4_546_195,
    "30-44": 2_487_010,
    "45-59": 910_610,
    "\u226560": 926_120,
}

# Midpoints for each duration class (minutes)
MIDPTS = {"<15": 7.5, "15-29": 22, "30-44": 37, "45-59": 52, "\u226560": 65}


def _to_percent(counts: dict[str, int]) -> dict[str, float]:
    """Convert absolute counts to percentage by category."""
    total = float(sum(counts.values()))
    return {k: v / total * 100.0 for k, v in counts.items()}


def _gaussian_from_percent(
    percent: dict[str, float], midpoints: dict[str, float], n_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return a gaussian curve matching the discrete percentage data."""
    keys = list(percent)
    p = np.array([percent[k] for k in keys]) / 100.0
    m = np.array([midpoints[k] for k in keys])
    mu = float((p * m).sum())
    sigma = float(np.sqrt((p * (m - mu) ** 2).sum()))
    x = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, n_points)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return x, y


# def time_to_distance(
#     x_time: np.ndarray, y_time: np.ndarray, speed_kmh: float = 70.0
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Convert a time distribution to distance (km) at constant speed."""
#     k = speed_kmh / 60.0
#     x_dist = x_time * k
#     y_dist = y_time / k  # change-of-variable scaling
#     return x_dist, y_dist


def gaussian_private_vehicle(
    province: str = "Canada", n_points: int = 200
) -> pd.DataFrame:
    """Return gaussian distribution for private vehicle time in minutes.

    The distribution is derived from national counts of commute durations and
    does not currently vary by *province*.
    """
    percent = _to_percent(COUNTS)
    x, y = _gaussian_from_percent(percent, MIDPTS, n_points=n_points)
    return pd.DataFrame({"Time": x, "Density": y})


# def plot_private_vehicle_gaussian(province: str = "Canada") -> None:
#     """Plot gaussian curve using matplotlib if available."""
#     if plt is None:
#         raise ImportError("matplotlib is required for plotting")
#     df = gaussian_private_vehicle(province)
#     plt.figure(figsize=(6, 4))
#     plt.plot(df["Time"], df["Density"], label=province)
#     plt.title(f"Private vehicle daily average time\n{province} - 2022")
#     plt.xlabel("Daily average time (minutes)")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def fetch_statcan_daily_drivers():
#     """Fetch daily driver counts per province."""
#     t_part = table_to_df("45-10-0104-03")
#     part = t_part[(t_part["Activity group"] == "Sleep and personal activities") &
#                   (t_part["Statistics"] == "Participation rate") &
#                   (t_part["Age group"] == "Total, 15 years and over") &
#                   (t_part["Gender"] == "Total, all persons")]
#     part = part.rename(columns={"GEO": "Province", "VALUE": "PartRate"})
#     part = part[["Province", "PartRate"]]
#     t_pop = table_to_df("17-10-0005-01")
#     t_pop["REF_DATE"] = pd.to_datetime(t_pop["REF_DATE"])
#     latest_date = t_pop["REF_DATE"].max()
#     pop = t_pop[(t_pop["REF_DATE"] == latest_date) &
#                 (t_pop["Gender"] == "Total - gender") &
#                 (t_pop["Age group"] == "All ages")]
#     pop = pop.rename(columns={"GEO": "Province", "VALUE": "Population"})
#     pop = pop[["Province", "Population"]]
#     pop = pop[pop["Province"] == "Canada"]
#     merged = part.merge(pop, on="Province", how="left")
#     merged["Drivers"] = (merged["Population"][0] * merged["PartRate"] / 100).round().astype(int)
#     merged = merged[["Province", "Drivers"]]
#     return merged


# def fetch_statcan_time_to_work():
#     """Placeholder for future StatCan commute time data."""
#     pass
    
    



if __name__ == "__main__":

    # sample data for testing set_data, averages and recharge calculations
    sample_df = pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Distance_km": [10, 20, 30, 40, 50, 60, 70],
    })

    car = CarUsage()

    print("Choose a test case:")
    print("1: test set_data and average_daily_distance")
    print("2: test set_average")
    print("3: test recharge_needed and set_recharge_needed")
    print("4: test fetchData and __getitem__")
    print("5: run all tests")
    choice = int(input("Enter case number: "))

    match choice:
        case 1:
            car.set_data(sample_df)
            print("average_daily_distance():", car.average_daily_distance())

        case 2:
            # set a single float average
            car.set_average(42.0)
            print("averages after float:", car.averages)
            # set a list of averages
            car.set_average([5, 6, 7, 8, 9, 10, 11])
            print("averages after list:", car.averages)

        case 3:
            car.set_data(sample_df)
            print("recharge_needed(150):", car.recharge_needed(150))
            car.set_recharge_needed(150, "Monday")

        case 4:
            car.fetchData()
            print("first rows of fetched data:\n", car.data.head())
            print("getitem by province 'QC':\n", car["QC"])
            print("getitem by (province, year):\n", car[("QC", 2022)])
            print("getitem by dict filter:\n", car[{"Province": "QC"}])
            # a custom callable
            filtered = car[lambda df: df[df["Annual_km"] > 20000]]
            print("callable filter >20000 Annual_km:\n", filtered)

        case 5:
            # run every function in sequence
            print("1) set_data & average_daily_distance")
            car.set_data(sample_df)
            print(car.average_daily_distance())

            print("\n2) set_average")
            car.set_average(50.0)
            car.set_average([1, 2, 3, 4, 5, 6, 7])
            print(car.averages)

            print("\n3) recharge_needed & set_recharge_needed")
            print(car.recharge_needed(120))
            car.set_recharge_needed(120, "Tuesday")

            print("\n4) fetchData & __getitem__")
            car.fetchData()
            print(car.data.head())
            print(car["ON"])
            print(car[("CA", 2022)])
            print(car[{"Year": 2022}])
            print(car[lambda df: df["Weekday_km"] > df["Weekend_km"]])

        case _:
            print("Invalid choice. Exiting.")


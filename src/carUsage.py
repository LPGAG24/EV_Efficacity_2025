import pandas as pd
import numpy as np
import re
import math

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
            avg[day] = df["Distance_km"].sum() / df["Distance_km"].nunique() \
                       if not df.empty else 0.0
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
        results = {}
        for prov in PROVINCES:
            df      = fetch_table(prov)
            km_yr   = annual_km(df)
            wkday_km, wkend_km = daily_split(km_yr)
    
            results[prov.upper()] = {
                "year":         YEAR,
                "annual_km":    km_yr,
                "wkday_km":     wkday_km,
                "wkend_km":     wkend_km,
                "μ_wkday":      mu_from_mean(wkday_km),
                "μ_wkend":      mu_from_mean(wkend_km),
                "σ":            SIGMA_LN,
            }

        df = fetch_table("ca", yr=YEAR, URL=URL_CAN)
        km_yr = annual_km(df)
        wkday_km, wkend_km = daily_split(km_yr)
        # Add Canada-wide data
        results["CA"] = {
            "year":         YEAR,
            "annual_km":    km_yr,
            "wkday_km":     wkday_km,
            "wkend_km":     wkend_km,
            "μ_wkday":      mu_from_mean(wkday_km),
            "μ_wkend":      mu_from_mean(wkend_km),
            "σ":            SIGMA_LN,
        }

        print("Car usage data fetched and processed.")
        print("Average distance traveled per year (km):")
        for prov, data in results.items():
            print(f"  {prov}: {data['annual_km']}")
        print("Average distance traveled per weekday (km):")
        for prov, data in results.items():
            print(f"  {prov}: {data['wkday_km']}")
        print("Average distance traveled per weekend (km):")
        for prov, data in results.items():
            print(f"  {prov}: {data['wkend_km']}")
        print("Done")



if __name__ == "__main__":
    car_usage = CarUsage()
    car_usage.fetchData()
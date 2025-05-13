import pandas as pd

class CarDistribution:
    # -------------------
    # 1) Initialization
    # -------------------
    def __init__(self, data: pd.DataFrame, Province: str = "Canada") -> None:
        self.initData = data
        self.data = pd.DataFrame.copy(data)
        self.Province: str = Province
        self.nb_of_vehicles = {}
        self.fuel_type = {}
        self.fuel_type_percent = {}
        self.fuel_type_percent_by_vehicle = {}
        self.remove_unwanted_rows()
        self.rename_CanStat_type()
        self.check_full_car_numbers()
        self.set_fuel_type_percent()
        self.set_fuel_type_percent_by_vehicle()

    def __repr__(self) -> str:
        return f"CarDistribution(Province={self.Province}, data_shape={self.data.shape})"


    def __getitem__(self, key):
        """
        Flexible selector.

        ─── Accepted keys ────────────────────────────────────────────
        • 'Ontario'                       → rows for one province
        • ['ON','QC'] or slice(...)       → rows for several provinces
        • ('Ontario', 'Subcompact')       → rows for province & vehicle type
        • ('Ontario', 'Subcompact', 'BEV')→ province + type + fuel
        • {'Province':'ON', 'Fuel':'BEV'} → arbitrary column/value filter
        • callable → passes the DataFrame to the func and returns the result
        ──────────────────────────────────────────────────────────────
        """
        df = self.data

        # 1. str  → single Province
        #    (e.g. 'Ontario', 'Quebec', 'Canada')
        if isinstance(key, str):
            return df[df["Province"] == key]

        # 2. list / slice  → multiple Provinces
        #    (e.g. ['Ontario', 'Quebec'], slice(0, 3))
        if isinstance(key, (list, slice)):
            return df[df["Province"].isin(df["Province"].unique()[key])]

        # 3. tuple  → hierarchical selector
        #    (e.g. ('Ontario', 'Subcompact'), ('Ontario', 'Subcompact', 'BEV'))
        if isinstance(key, tuple):
            cols = ["Province", "Vehicle Type", "Fuel Type"]
            mask = pd.Series(True, index=df.index)
            for lvl, value in zip(cols, key):
                mask &= df[lvl] == value
            return df[mask]

        # 4. dict  → free‑form selector {col: value}
        #    (e.g. {'Province': 'Ontario', 'Fuel Type': 'BEV'})
        if isinstance(key, dict):
            mask = pd.Series(True, index=df.index)
            for col, value in key.items():
                mask &= df[col] == value
            return df[mask]

        # 5. callable  → power‑user hook
        #    (e.g. lambda df: df[df['Province'] == 'Ontario'])
        #    (e.g. lambda df: df[df['Province'] == 'Ontario'].groupby('Vehicle Type'))
        #    (e.g. lambda df: df[df['Province'] == 'Ontario'].groupby('Vehicle Type').sum())
        #    (e.g. lambda df: df[df['Province'] == 'Ontario'].groupby('Vehicle Type').mean())
        #    (e.g. lambda df: df[df['Province'] == 'Ontario'].groupby('Vehicle Type').agg({'Vehicles nb': 'sum'}))
        if callable(key):
            return key(df)

        raise TypeError(
            "Key must be str, list, slice, tuple, dict, or callable."
        )

    # -------------------
    # 2) Data Cleaning
    # -------------------
    def remove_unwanted_rows(self) -> None:
        """
        Keeps only the target province’s non-total rows with 'All fuel types'.
        """
        df = self.data
        df = df[
            (~df["Vehicle Type"].str.contains("Total", na=False))
            & (df["Fuel Type"] == "All fuel types")
        ]
        self.data = df

    def rename_CanStat_type(self) -> None:
        """
        Normalizes 'Vehicle Type' strings into canonical efficiency classes.
        """
        df = self.data
        df["Vehicle Type"] = (
            df["Vehicle Type"].str.split(r"[:,]", n=1, expand=True)[0].str.strip()
        )
        mapping = {
            "Passenger cars": "Subcompact",
            "Pickup trucks": "Pickup truck",
            "Sport utility vehicles": "Sport utility vehicle",
            "Multi-purpose vehicles": "Minivan",
            "Vans": "Minivan",
            "Station wagons": "Station wagon",
            "Motorcycles and mopeds": "Two-seater",
            "Buses": "Other",
            "Class 7 vehicles": "Other",
            "Class 8 vehicles": "Other",
        }
        self.data["Vehicle Type"] = df["Vehicle Type"].map(mapping).fillna(
            df["Vehicle Type"]
        )

    def check_full_car_numbers(self) -> None:
        """
        Checks if sum of vehicles in the province exceeds the max across entire data.
        """
        if (
            self.data[self.data["Province"] == self.Province]["Vehicles nb"].sum()
            > self.initData["Vehicles nb"].max()
        ):
            raise ValueError("Count is greater than total number of vehicles")

    # -------------------
    # 3) Fuel Type Stats
    # -------------------
    def set_fuel_type_percent(self, Province: str = "Canada") -> None:
        """
        Computes and stores the percent share of each fuel type in the total fleet.
        """
        Province = Province or self.Province 
        selected_vehicles = self.data[
            (self.data["Province"] == Province)
            & (self.data["Vehicle Type"] == "Total, road motor vehicle registrations")
        ]
        self.nb_of_vehicles = selected_vehicles["Vehicles nb"]
        fuel_type_array = selected_vehicles["Fuel Type"].value_counts()
        for fuel_type, _ in fuel_type_array.items():
            if fuel_type != "All fuel types":
                self.fuel_type[fuel_type] = selected_vehicles[selected_vehicles["Fuel Type"] == fuel_type]["Vehicles nb"].sum()
                self.fuel_type_percent[fuel_type] = (self.fuel_type[fuel_type] / self.nb_of_vehicles.max() * 100)
        self.fuel_type_percent = dict(sorted(self.fuel_type_percent.items(), key=lambda x: x[1], reverse=True))
        print("Fuel type percent:")
        for fuel_type, percent in self.fuel_type_percent.items():
            print(f"{fuel_type}: {percent:.2f}%")

    def get_fuel_type(self) -> pd.DataFrame:
        """
        Returns a DataFrame of fuel‐type shares across the fleet.
        """
        return pd.DataFrame.from_dict(
            self.fuel_type_percent, orient="index", columns=["Percent"]
        )

    def set_fuel_type_percent_by_vehicle(self) -> None:
        """
        Computes per-vehicle-type shares and stores in 'fuel_type_percent_by_vehicle'.
        """
        df = self.data
        totals_by_type = (
            df[["Vehicle Type", "Vehicles nb"]]
            .groupby("Vehicle Type", as_index=False)["Vehicles nb"]
            .sum()
            .rename(columns={"Vehicles nb": "Count"})
        )
        totals_by_type["Pct of total"] = (
            totals_by_type["Count"] / totals_by_type["Count"].sum() * 100
        )
        self.fuel_type_percent_by_vehicle = (
            totals_by_type.set_index("Vehicle Type").to_dict()["Pct of total"]
        )

    def get_fuel_type_percent_by_vehicle(self) -> pd.DataFrame:
        """
        Returns per-vehicle-type percentage of the fleet as a DataFrame.
        """
        return pd.DataFrame.from_dict(
            self.fuel_type_percent_by_vehicle, orient="index", columns=["Percent"]
        )

    def switch_province(self, Province: str):
        self.Province = Province
        self.set_fuel_type_percent()
        self.set_fuel_type_percent_by_vehicle()



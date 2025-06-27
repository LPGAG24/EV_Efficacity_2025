import pandas as pd
from typing import TypedDict

"""
CarEfficiency class
class for:
   - get the car efficiency

Usage:
   - get the car efficiency by type
   - get the car efficiency by province
   - get the car efficiency by fuel type
   - get the car efficiency by year
   - get the car efficiency by make
   - get the car efficiency by model
   - get the car efficiency by trim
   - get the car efficiency by vehicle class
"""




class CarEfficiency:
    _SNOW_COEFF = 2  # Coefficient for snow conditions (2x consumption)


    def __init__(self, data: pd.DataFrame):
        """vehicle_class: ['Subcompact', 'Mid-size', 'Compact', 'Two-seater', 'Full-size',\n
       'Station wagon', 'Sport utility vehicle', 'Pickup truck',\n
       'Minicompact', 'Minivan']"""
        self.data: pd.DataFrame = data
        self.vehicleClass: pd.Series = self.data["Vehicle class"].unique()
        self.fuel_consumption: dict[str, float] = {}
        self.distance: dict[str, float] = {}
        
        self.efficiency_percent_by_vehicle_type: dict[str, float] = {} 
        self.set_efficiency_by_type()
 

    def set_efficiency_by_type(self) -> None:
        """Compute and store average combined consumption for each vehicle class.

        1. Collapse `Vehicle class` to the text before any “:”.
        2. Dynamically locate the column whose name contains both “Combined” and “Le”.
        3. Extract the leading numeric value from that column’s strings and convert to float.
        4. Group by `Vehicle class` and compute the mean of those float values.
        5. Store the result in `self.efficiency_by_vehicle_type` as a DataFrame
           with columns [`Vehicle class`, `<combined_column>`].

        Args:
            None

        Returns:
            None

        Side effects:
            Sets `self.efficiency_by_vehicle_type` to a pandas DataFrame:
            - Column 0: “Vehicle class” (str)
            - Column 1: the combined-consumption column (float)

        Example:
            >>> df = pd.DataFrame({
            ...     "Vehicle class": ["Compact: X","Compact: Y","Subcompact: Z"],
            ...     "Combined (Le/100 km)": ["5.6 (20.2 kWh/100 km)",
            ...                               "5.4 (19.5 kWh/100 km)",
            ...                               "4.8 (17.3 kWh/100 km)"]
            ... })
            >>> cd = CarDistribution(df)
            >>> cd.set_efficiency_by_type()
            >>> cd.efficiency_by_vehicle_type
               Vehicle class  Combined (Le/100 km)
            0       Compact                   5.50
            1     Subcompact                   4.80
        """
        df = self.data
        df["Vehicle class"] = df["Vehicle class"].str.split(":", n=1).str[0]
        mask = (
            df.columns.str.contains("Combined") &
            df.columns.str.contains("Le")
        )
        index_of_combined = df.columns[mask].tolist()[0]
        df[index_of_combined] = df[index_of_combined].str.split(" ").str[0].astype(float)

        # 2) now group and take the mean
        means = df.groupby("Vehicle class")[index_of_combined].mean()
        self.efficiency_by_vehicle_type = means.reset_index()
        self.efficiency_by_vehicle_type.columns = ["Vehicle class", index_of_combined]
        

    def get_efficiency_by_type(self, category: str = None) -> pd.DataFrame:
        """Return average combined consumption, optionally for one vehicle class.

        If `category` is provided, filters the `efficiency_by_vehicle_type`
        DataFrame to only that Vehicle class; otherwise returns the full table.

        Args:
            category (str, optional):  
                Name of the Vehicle class to filter by (e.g. "Compact", 
                "Sport utility vehicle"). If None (default), returns all classes.

        Returns:
            pd.DataFrame:  
                - If `category` is given: one-row DataFrame for that class.  
                - If `category` is None: full DataFrame with columns  
                  ["Vehicle class", "<combined-consumption column>"] and  
                  dtype float for the consumption values.

        Example:
            >>> # assume cd.efficiency_by_vehicle_type was set earlier
            >>> cd.get_efficiency_by_type("Compact")
              Vehicle class  Combined (Le/100 km)
            0       Compact                   5.50

            >>> # get the full table
            >>> cd.get_efficiency_by_type()
              Vehicle class  Combined (Le/100 km)
            0       Compact                   5.50
            1     Subcompact                  4.80
            2        Mid-size                 6.20
            ... 
        """
        df = self.efficiency_by_vehicle_type
        return df[df["Vehicle class"] == category] if category is not None else df

    def get_mean_efficiency(self, category: str | list[str] | None = None) -> float | pd.Series:
        """Must be able to return efficiency as a float or pandas Series. like

        >>> car_efficiency = CarEfficiency(data)
        >>> car_efficiency.get_mean_efficiency()
        Returns the mean efficiency of all vehicles in the dataset.
        
        >>> car_efficiency = CarEfficiency(data)
        >>> car_efficiency.get_mean_efficiency(["Compact", "Subcompact"])
        Returns the mean efficiency of specified vehicle classes. -> [0.56, 0.48]

        """
        category = None if isinstance(category, str) else category
        df = self.efficiency_by_vehicle_type
        return df[df["Vehicle class"].isin(category)].mean() if category else df.mean()

    def __call__(self) -> pd.DataFrame:
        return self.data

    def __getitem__(self, key) -> pd.DataFrame:
        """
        Flexible selector for the efficiency table.

        Accepted keys
        -------------
        • 'Kia'                                 → rows for one make
        • ['Kia','Hyundai'] or slice(...)       → several makes
        • ('Kia', 'EV6')                        → make + model
        • ('Kia', 'EV6', 2023)                  → make + model + year
        • ('Kia', 'EV6', 2023, 'BEV')           → + fuel type
        • {'Make':'Kia', 'Model year':2023}     → arbitrary column=value filter
        • callable                              → lambda df: ... (power-user hook)
        """
        df = self.data          # master DataFrame stored in __init__

        # 1 ── single make  -------------------------------------------------
        if isinstance(key, str):
            return df[df["Make"] == key]

        # 2 ── several makes (list/tuple/slice) -----------------------------
        if isinstance(key, (list, tuple, slice)):
            return df[df["Make"].isin(df["Make"].unique()[key])]

        # 3 ── hierarchical tuple (Make, Model, Year, Fuel) -----------------
        if isinstance(key, tuple):
            cols = ["Make", "Model", "Model year", "Fuel Type"]  # adjust if needed
            mask = pd.Series(True, index=df.index)
            for col, val in zip(cols, key):
                mask &= df[col] == val
            return df[mask]

        # 4 ── free-form dict {column: value, ...} --------------------------
        if isinstance(key, dict):
            mask = pd.Series(True, index=df.index)
            for col, val in key.items():
                mask &= df[col] == val
            return df[mask]

        # 5 ── callable hook  (lambda df: ...) ------------------------------
        if callable(key):
            return key(df)

        raise TypeError("Key must be str, list/tuple/slice, dict, or callable.")



class ChargerInfo:
    def __init__(self, data: pd.DataFrame):
        self.chargers = pd.DataFrame()
        self.chargers["Charger watts"] = {
            "Level 1" : 120*[12, 14, 16],
            "Level 2 (208)":208*[15,48,80],
            "Level 2 (240)":240*[15,48,80],
            "Level 3 (DC)": 480*[50, 100, 150],}
    
    def get_charger(self) -> pd.DataFrame:
        """
        Returns a DataFrame of charger information.
        """
        return self.chargers
    
    def get_charging_time(self, vehicle_class: str, battery_capacity_kwh: float) -> pd.DataFrame:
        """
        Calculate the charging time for a given vehicle class and battery capacity.
        
        Args:
            vehicle_class (str): The class of the vehicle (e.g., "Level 1", "Level 2 (208)", etc.).
            battery_capacity_kwh (float): The battery capacity in kWh.
        
        Returns:
            pd.DataFrame: A DataFrame with charging times for each charger type.
        """
        if vehicle_class not in self.chargers["Charger watts"].index:
            raise ValueError(f"Unknown charger type: {vehicle_class}")
        
        charger_watts = self.chargers.loc[vehicle_class, "Charger watts"]
        charging_times = battery_capacity_kwh * 1000 / charger_watts

        charging_times = charging_times.round(2)  # Round to 2 decimal places
        charging_times = charging_times.astype(float)

        return pd.DataFrame({
            "Charger Type": self.chargers.index,
            "Charging Time (hours)": charging_times
        })

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    # Example usage
    data = pd.DataFrame({
        "Vehicle class": ["Compact: X", "Compact: Y", "Subcompact: Z"],
        "Combined (Le/100 km)": ["5.6 (20.2 kWh/100 km)", "5.4 (19.5 kWh/100 km)", "4.8 (17.3 kWh/100 km)"]
    })
    
    car_efficiency = CarEfficiency(data)
    print(car_efficiency.get_efficiency_by_type())


    # Plotting the efficiency by vehicle type
    plt.bar(car_efficiency.efficiency_by_vehicle_type["Vehicle class"],
            car_efficiency.efficiency_by_vehicle_type["Combined (Le/100 km)"])
    plt.xlabel("Vehicle Class")
    plt.ylabel("Combined (Le/100 km)")
    plt.title("Efficiency by Vehicle Type")
    plt.xticks(rotation=45)
    plt.show()
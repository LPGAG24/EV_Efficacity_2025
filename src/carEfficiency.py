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
        self.set_battery_by_type()
 

    def set_efficiency_by_type(self, selected_types: list[str] | None = None) -> None:
        """
        Calcule la consommation moyenne combinée (Le/100 km) par type de véhicule.

        Étapes :
        1. Normalise la colonne "Vehicle class" en ne gardant que le texte avant le caractère ":".
        2. Repère dynamiquement la colonne contenant à la fois "Combined" et "Le".
        3. Extrait la partie numérique (valeur avant l’espace) de cette colonne et la convertit en float.
        4. [Optionnel] Si `selected_types` est fourni, garde seulement ces classes.
        5. Regroupe par "Vehicle class" et calcule la moyenne.
        6. Stocke le résultat dans `self.efficiency_by_vehicle_type`
        (colonnes : "Vehicle class", <combined_column>).

        Args:
            selected_types (list[str] | None): sous-ensemble de classes à conserver
                (ex. ["Compact", "SUV"]). Laisser `None` pour tout calculer.

        Returns:
            None
        """
        df = self.data.copy()

        # 1) Nettoie la classe
        df["Vehicle class"] = df["Vehicle class"].str.split(":", n=1).str[0]

        # 2) Colonne combinée (fonctionne même si le nom change : "Combined … Le…")
        mask = df.columns.str.contains("Combined") & df.columns.str.contains("Le")
        combined_col = df.columns[mask][0]

        # 3) Conversion texte → float
        df[combined_col] = df[combined_col].str.split(" ").str[0].astype(float)

        # 4) Filtre éventuel
        if selected_types:
            df = df[df["Vehicle class"].isin(selected_types)]

        # 5) Moyenne par classe
        means = (
            df.groupby("Vehicle class", sort=False)[combined_col]
            .mean()
            .reset_index()
        )

        # 6) Stockage
        self.efficiency_by_vehicle_type = means

    def set_battery_by_type(self, selected_types: list[str] | None = None) -> None:
        """Compute mean battery capacity (kWh) by vehicle class."""
        df = self.data.copy()
        df["Vehicle class"] = df["Vehicle class"].str.split(":", n=1).str[0]
        df["Range (km)"] = pd.to_numeric(df["Range (km)"], errors="coerce")
        df["Combined (kWh/100 km)"] = pd.to_numeric(
            df["Combined (kWh/100 km)"], errors="coerce"
        )
        df["Battery_kWh"] = df["Range (km)"] * df["Combined (kWh/100 km)"] / 100.0
        if selected_types:
            df = df[df["Vehicle class"].isin(selected_types)]
        means = (
            df.groupby("Vehicle class", sort=False)["Battery_kWh"].mean().reset_index()
        )
        self.battery_by_vehicle_type = means

    def get_battery_by_type(self, category: str | None = None) -> pd.DataFrame:
        df = self.battery_by_vehicle_type
        return df[df["Vehicle class"] == category] if category else df

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
    """Utility class providing charger power data and charge time estimates."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        # charger power levels in Watts for low/medium/high settings
        self.chargers = pd.DataFrame(
            {
                "Low":    [120 * 12, 208 * 15, 240 * 15, 480 * 50],
                "Medium": [120 * 14, 208 * 48, 240 * 48, 480 * 100],
                "High":   [120 * 16, 208 * 80, 240 * 80, 480 * 150],
            },
            index=["Level 1", "Level 2 (208)", "Level 2 (240)", "Level 3 (DC)"]
        )
    
    def get_charger(self) -> pd.DataFrame:
        """Return table of charger power levels in Watts."""
        return self.chargers.copy()

    def get_charging_time(self, charger_type: str, battery_capacity_kwh: float) -> pd.DataFrame:
        """Return charging times for ``charger_type`` at each power level.

        Parameters
        ----------
        charger_type : str
            Name of the charger type (e.g. ``"Level 1"``).
        battery_capacity_kwh : float
            Battery capacity in kilowatt-hours.
        """
        if charger_type not in self.chargers.index:
            raise ValueError(f"Unknown charger type: {charger_type}")

        watts = self.chargers.loc[charger_type]
        hours = (battery_capacity_kwh * 1000 / watts).round(2)

        return hours.to_frame(name="Charging Time (hours)")

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.DataFrame({
        "Vehicle class": ["Compact: X", "Compact: Y", "Subcompact: Z"],
        "Combined (Le/100 km)": ["5.6 (20.2 kWh/100 km)", "5.4 (19.5 kWh/100 km)", "4.8 (17.3 kWh/100 km)"]
    })

    car_efficiency = CarEfficiency(data)
    print(car_efficiency.get_efficiency_by_type())

    plt.bar(
        car_efficiency.efficiency_by_vehicle_type["Vehicle class"],
        car_efficiency.efficiency_by_vehicle_type["Combined (Le/100 km)"]
    )
    plt.xlabel("Vehicle Class")
    plt.ylabel("Combined (Le/100 km)")
    plt.title("Efficiency by Vehicle Type")
    plt.xticks(rotation=45)
    plt.show()

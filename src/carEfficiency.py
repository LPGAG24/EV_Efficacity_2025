import pandas as pd

#VEHICULE_CLASS = {['Subcompact', 'Mid-size', 'Compact', 'Two-seater', 'Full-size',
#       'Station wagon: Small', 'Sport utility vehicle: Standard',
#       'Sport utility vehicle: Small', 'Pickup truck: Standard',
#       'Minicompact', 'Station wagon: Mid-size', 'Minivan']}

class CarEfficiency:
    def __init__(self, data: pd.DataFrame):

        self.data = data
        self.vehicleClass = self.data["Vehicle class"].unique()
        self.fuel_consumption = {}
        self.distance = {}
        self.efficiency_percent_by_vehicle_type = {}
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
import pandas as pd

class CarDistribution:
    """Simple wrapper around a vehicle stock DataFrame."""

    def __init__(self, data: pd.DataFrame, Province: str = "Canada") -> None:
        self.data = data.copy()
        self.Province = Province

    def __getitem__(self, key):
        df = self.data
        if isinstance(key, str):
            return df[df["Province"] == key]
        if isinstance(key, (list, tuple)) and key and isinstance(key[0], str) and not isinstance(key, tuple):
            return df[df["Province"].isin(key)]
        if isinstance(key, tuple):
            cols = ["Province", "Vehicle Type", "Fuel Type"]
            mask = pd.Series(True, index=df.index)
            for col, val in zip(cols, key):
                mask &= df[col] == val
            return df[mask]
        if isinstance(key, dict):
            mask = pd.Series(True, index=df.index)
            for col, val in key.items():
                mask &= df[col] == val
            return df[mask]
        raise TypeError("Invalid key type")

    def get_fuel_type(self) -> pd.DataFrame:
        df = self[self.Province]
        tot = df.groupby("Fuel Type")["Vehicles nb"].sum()
        pct = (tot / tot.sum() * 100).round(2)
        return pct.reset_index(name="Percent")

    def get_fuel_type_percent_by_vehicle(self) -> pd.DataFrame:
        df = self[self.Province]
        tot = df.groupby("Vehicle Type")["Vehicles nb"].sum()
        pct = (tot / tot.sum() * 100).round(2)
        return pct.reset_index(name="Percent")

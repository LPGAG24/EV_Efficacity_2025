import pandas as pd

class CarUsage:
    """Minimal vehicle-usage helper used for the demo GUI."""

    def __init__(self) -> None:
        self.weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.weekends = ["Saturday", "Sunday"]
        self.data = pd.DataFrame({
            "Province": ["Canada"],
            "Weekday_km": [40],
            "Weekend_km": [30],
        })

    def fetchData(self) -> None:
        """Placeholder that would normally fetch data."""
        pass

    def __getitem__(self, key):
        df = self.data
        if isinstance(key, str):
            return df[df["Province"] == key]
        if isinstance(key, dict):
            mask = pd.Series(True, index=df.index)
            for col, val in key.items():
                mask &= df[col] == val
            return df[mask]
        raise TypeError("Invalid key type")

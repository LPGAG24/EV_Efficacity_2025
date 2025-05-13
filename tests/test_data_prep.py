from pathlib import Path

import pytest
import pandas as pd 
import src.carDistribution as cd
import src.data_prep_canada as dp
from src.carEfficiency import CarEfficiency
from src.carDistribution import CarDistribution

def test_km_to_kwh():
    assert dp.km_to_kwh(100) == 18

def test_car_distribution():
    df = pd.DataFrame({
        "Province": ["Canada", "Canada", "Canada"],
        "Vehicle Type": ["Passenger cars", "Pickup trucks", "Sport utility vehicles"],
        "Fuel Type": ["All fuel types", "All fuel types", "All fuel types"],
        "Vehicles nb": [100, 200, 300]
    })

    cd = CarDistribution(df)
    assert cd.get_fuel_type_percent() == 100
    assert cd.get_fuel_type_percent_by_vehicle()
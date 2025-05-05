from pathlib import Path

import pytest
import pandas as pd 
import data_prep as dp

def test_km_to_kwh():
    assert dp.km_to_kwh(100) == 18
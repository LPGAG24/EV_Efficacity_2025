from carDistribution import CarDistribution
from carEfficiency import CarEfficiency
from carRecharge import CarRecharge
from carUsage import CarUsage
from util import fetch_statcan_fleet, download_ckan_resource
from util import calculator

import pandas as pd
import numpy as np


class Merger:
    """Merger class to combine car distribution, efficiency, usage, and recharge data."""
    
    
    def __init__(self):
        #Start with the car distribution data
        self.car_distribution = CarDistribution()
        # car efficiency data for electric vehicles
        self.electric_efficiency = CarEfficiency(
            download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9")
        )
        ratio_by_type = self.car_distribution.get_fuel_type_percent_by_vehicle()
        
        self.electric_efficiency.get_mean_efficiency(self.electric_efficiency()["Vehicle class"].unique())
        
        
        

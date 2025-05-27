from data_prep_canada import *
from carDistribution import CarDistribution
from carEfficiency import CarEfficiency


def main():
    """Section 1: Car Distribution. Get an initial figure of the cars distribution in Canada and in each provinces"""
    # "23-10-0308-01" : Vehicle registrations, by type of vehicle and fuel type
    carDistribution = CarDistribution(fetch_statcan_fleet())


    """Section 2: Car Efficiency, get a general idea of the car efficiency either electric or hybrid"""
    # "026e45b4-eb63-451f-b34f-d9308ea3a3d9" : Electric vehicle registrations by province
    electricEfficiency = CarEfficiency(download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9"))
    # "8812228b-a6aa-4303-b3d0-66489225120d" : hybrid vehicle registrations by province and fuel type
    hybridEfficiency = CarEfficiency(download_ckan_resource("8812228b-a6aa-4303-b3d0-66489225120d"))
    print("Electric vehicle efficiency by type:")
    print(electricEfficiency.get_efficiency_by_type())
    print("Hybrid vehicle efficiency by type:")
    print(hybridEfficiency.get_efficiency_by_type())


    """Section 3: Car usage, get an idea of the car usage in Canada and in each provinces"""
    

main()
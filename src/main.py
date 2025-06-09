from data_prep_canada import *
from carDistribution import CarDistribution
from carEfficiency import CarEfficiency
from carRecharge import CarRecharge
from carUsage import CarUsage
import util
import pandas as pd
import numpy as np
import util.calculator




def main():
    """Section 1: Car Distribution. Get an initial figure of the cars distribution in Canada and in each provinces"""
    # "23-10-0308-01" : Vehicle registrations, by type of vehicle and fuel type
    car_distribution = CarDistribution(fetch_statcan_fleet())
    car_distribution.data = car_distribution()[["Province", "Vehicle Type", "Vehicles nb"]]
    
    # "026e45b4-eb63-451f-b34f-d9308ea3a3d9" : Electric vehicle registrations by province
    electric_efficiency = CarEfficiency(download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9"))
    electric_efficiency.data = electric_efficiency()[["Model year", "Make", "Combined (kWh/100 km)", "Recharge time (h)", "Vehicle class"]]
    
    # "8812228b-a6aa-4303-b3d0-66489225120d" : hybrid vehicle registrations by province and fuel type
    hybrid_efficiency = CarEfficiency(download_ckan_resource("8812228b-a6aa-4303-b3d0-66489225120d"))

    car_usage = CarUsage()
    car_usage.fetchData()


    car_recharge = CarRecharge(pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Distance_km": [10, 15, 20, 25, 30, 5, 10]
    }))
    
    
    #here, lets try to simulate the electric consumption if all vehicules where Kia EV6

    numbers_of_cars = car_distribution["Quebec", "Subcompact"].iloc[0]["Vehicles nb"]


    Tesla_3 = electric_efficiency[{"Make" : "Tesla", "Model" : "Model 3 Standard Range"}]
    Tesla_3_efficiency = Tesla_3.iloc[0]["Combined (kWh/100 km)"] if not Tesla_3.empty else 0.2
    Tesla_3_power = Tesla_3.iloc[0]["Recharge time (h)"] if not Tesla_3.empty else 8

    average_distance = car_usage[{"Province": "Quebec"}].iloc[0]["Weekday_km"]
    
    car_recharge = CarRecharge()
    

    total_consumption = util.calculator.calculate_grid_power(
        car_count=numbers_of_cars,
        efficiency_kwh_per_100km=Tesla_3_efficiency,
        distance_km=average_distance,
        charging_time=Tesla_3_power,
        charging_profile=car_recharge.get_weekly_profile()
    )

    print(f"Total number of cars: {numbers_of_cars}")
    print(f"Tesla Model 3 charging power: {Tesla_3_power} kWh")
    print(f"Average distance: {average_distance} km")
    print(f"Total consumption for all cars: {total_consumption:.2f} kWh")

    # plot the charging profile
    import matplotlib.pyplot as plt
    car_recharge.get_weekly_profile().pivot(index='Hour', columns='Day', values='ChargingPerc').plot(kind='bar', figsize=(12, 6))
    plt.title('Car Charging Profile')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Charging Probability')
    plt.legend(title='Day')
    plt.show()
    print("Car Distribution by fuel type:")


    print(car_distribution.get_fuel_type())
    """Section 3: Car usage, get an idea of the car usage in Canada and in each provinces"""
    

if __name__ == "__main__":
    # Uncomment the following line to enable logging
    main()
    

    
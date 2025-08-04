# EV_Efficacity_2025

Simulation toolkit and Streamlit dashboard for modelling the charging demand of electric vehicles across Canada. The project bundles data acquisition utilities, statistical models and an interactive UI to explore vehicle fleets, energy efficiency and charging behaviour.

## Architecture

- `CarDistribution` – summarises vehicle registrations by province and fuel type. It cleans StatCan data and exposes flexible selectors to retrieve subsets of the fleet (province, vehicle class, fuel) and helpers to compute fuel shares.
- `CarEfficiency` – processes the federal open dataset of vehicle efficiency to average consumption and battery capacity by vehicle class.
- `CarUsage` – retrieves NRCan tables of annual travel distances per province and transforms them into daily weekday/weekend usage and recharge energy needs.
- `CarRecharge` – builds default weekday/weekend charging probability profiles and provides statistical samplers for session energy, duration and frequency.
- `util.calculator.calculate_grid_power` – converts fleet size, efficiency and distance travelled into total energy demand, optionally weighted by a charging profile.

## Data sources

`data_prep_canada.py` centralises data downloads:

- `download_ckan_resource` pulls full tables from the open.canada.ca CKAN API and stores raw JSON for reproducibility.
- `fetch_statcan_fleet` uses the StatsCan Web Data Service to obtain the latest light‑duty vehicle fleet by province and fuel type.
- `CarUsage.fetchData` scrapes NRCan transport statistics to estimate daily travel distance per province.

These datasets feed the domain classes above and can be cached locally in `data/raw/`.

## Running the dashboard

The graphical interface lives in `src/gui/app.py` and is built with Streamlit.
To explore the model interactively:

```bash
streamlit run src/gui/app.py
```

## Development setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest -q
```

## Repository structure

```
src/
├── carDistribution.py
├── carEfficiency.py
├── carRecharge.py
├── carUsage.py
├── data_prep_canada.py
├── gui/app.py
└── util/
```

The repository contains additional helper modules and examples (see `src/main.py`) demonstrating how to combine the classes into end‑to‑end simulations.


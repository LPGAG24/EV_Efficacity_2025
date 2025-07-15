# EV_Efficacity_2025

# Modélisation charge VE

Ce dépôt contient le pipeline de données, les modèles et l’interface
graphique pour simuler la charge des véhicules électriques au Canada.

## Development Setup

Before running the test suite, install the development requirements:

```bash
pip install -r requirements-dev.txt
```

Then run the tests with:

```bash
pytest -q
```

## New features

- Weekend charging profiles can be enabled in the interface via the sidebar
  checkbox. When activated, home and work arrival distributions are edited
  specifically for weekend days.
- Weekend mode now has dedicated default values: later home arrival, later work
  arrival and weekend driving distances pulled from StatCan data.
- A calendar viewer is available ("Calendar" expander) which labels each day of
  the selected year as a weekday or weekend. Statutory holidays are treated as
  weekends.

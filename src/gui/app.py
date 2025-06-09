# app.py
import streamlit as st
import pandas as pd
import urllib.request, json, folium
from streamlit_folium import st_folium
import altair as alt
import numpy as np

from carDistribution import CarDistribution          # your classes
from carEfficiency   import CarEfficiency
from carRecharge     import CarRecharge
from carUsage        import CarUsage
from data_prep_canada import fetch_statcan_fleet, download_ckan_resource

st.set_page_config(page_title="EV & Hybrid Dashboard", layout="wide")

# ── GeoJSON helper ──────────────────────────────────────────────────────────
@st.cache_data
def load_canada_geojson():
    url = (
        "https://raw.githubusercontent.com/"
        "codeforgermany/click_that_hood/master/public/data/canada.geojson"
    )
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)

canada_geo = load_canada_geojson()

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.title("Filters")

# 1) Province chooser

@st.cache_data
def load_fleet():
    # slow: download or load data
    return fetch_statcan_fleet("23-10-0308-01")
fleet_df = load_fleet()

provinces = sorted(fleet_df["Province"].unique())
province  = st.sidebar.selectbox("Province", provinces, index=provinces.index("Canada"))

# 2) Vehicle‑class multiselect (for efficiency tables)
selected_types = []

# ── Data prep ───────────────────────────────────────────────────────────────
# Distribution object (for fleet tables)
@st.cache_resource
def get_dist(fleet_df, province):
    # fast: in-memory DataFrame processing
    return CarDistribution(fleet_df, Province=province)
dist = get_dist(fleet_df, province)

@st.cache_data
def load_electric_efficiency():
    return download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9")

@st.cache_data
def load_hybrid_efficiency():
    return download_ckan_resource("8812228b-a6aa-4303-b3d0-66489225120d")

electric_eff = CarEfficiency(load_electric_efficiency())
hybrid_eff   = CarEfficiency(load_hybrid_efficiency())


selected_types = st.sidebar.multiselect(
    "Vehicle class (efficiency)",
    options=electric_eff.efficiency_by_vehicle_type["Vehicle class"].tolist(),
)

# ── Layout ──────────────────────────────────────────────────────────────────
st.title(f"🚗 EV & Hybrid Dashboard — {province}")

# 1 & 2: tables and efficiency (your original code)
with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("1 · Fleet distribution")
        st.subheader("Fuel mix")
        st.table(dist.get_fuel_type())

        st.subheader("Vehicle‑type mix")
        st.table(dist.get_fuel_type_percent_by_vehicle())

    with col2:
        st.header("2 · Efficiency (kWh/100 km)")
        tabs = st.tabs(["⚡ Electric", "🔋 Hybrid"])

        with tabs[0]:
            df_e = electric_eff.efficiency_by_vehicle_type
            if selected_types:
                df_e = df_e[df_e["Vehicle class"].isin(selected_types)]
            st.dataframe(df_e, use_container_width=True)

        with tabs[1]:
            df_h = hybrid_eff.efficiency_by_vehicle_type
            if selected_types:
                df_h = df_h[df_h["Vehicle class"].isin(selected_types)]
            st.dataframe(df_h, use_container_width=True)

# --- Sidebar: user selects day and vehicle class
day_choices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Select Day of Week", day_choices)

available_classes = dist.data["Vehicle Type"].unique()
selected_class = st.sidebar.selectbox("Vehicle Class", available_classes)

# --- Get number of vehicles of that class in province
try:
    car_count = dist[(province, selected_class)]["Vehicles nb"].sum()
except Exception:
    car_count = 0

# --- Get efficiency for class (mean kWh/100km)
eff_row = electric_eff.get_efficiency_by_type(selected_class)
if not eff_row.empty:
    # Use the right column depending on your data (might be 'Combined (kWh/100 km)')
    eff_col = [c for c in eff_row.columns if "Combined (Le/100 km)" in c][0]
    eff = eff_row[eff_col].values[0]
else:
    eff = 18  # fallback value

# --- Get avg distance driven per day for that province (use CarUsage or fallback)
try:
    cu = CarUsage()
    cu.fetchData()
    dist_per_day = cu[{"Province": province}]  # returns DataFrame
    avg_distance = dist_per_day[f"{'Weekday' if selected_day in cu.weekdays else 'Weekend'}_km"].values[0]
except Exception:
    avg_distance = 30

# --- Charging profile for selected day (use CarRecharge or fallback)
usage_data = pd.DataFrame({"Day": [selected_day]*24, "Distance_km": [avg_distance]*24})
cr = CarRecharge(usage_data)
# Use some default profile or let user pick profile
cr.set_car_charging_prop(peaks=[(8, 0.25), (12, 0.25), (18, 0.5)], base=0.02)
profile_df = cr.get_charging_profile_df()
profile = profile_df[profile_df["Day"] == selected_day]["ChargingPerc"].values

# --- Interpolate hourly profile to 30-min intervals
profile_30min = np.repeat(profile, 2)
time_bins = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]

# --- Compute total energy required per 30 min
total_daily_energy = car_count * eff * avg_distance / 100  # in kWh
# The sum of profile_30min should be 1 (if profile was normalized)
energy_per_bin = profile_30min * total_daily_energy

# --- Plot
energy_df = pd.DataFrame({"Time": time_bins, "Energy_kWh": energy_per_bin})
chart = alt.Chart(energy_df).mark_bar().encode(
    x=alt.X('Time', sort=None, title='Time of Day'),
    y=alt.Y('Energy_kWh', title='Energy required (kWh)'),
    tooltip=['Time', 'Energy_kWh']
).properties(
    title=f"Total Charging Demand by 30-min Slot ({province}, {selected_class}, {selected_day})",
    width=900,
    height=350
)
st.altair_chart(chart, use_container_width=True)


# ── 3 · Canada EV‑share choropleth ──────────────────────────────────────────
st.header("3 · Provincial EV share (battery‑electric)")

# a) total fleet per province (light‑duty only, all fuels)
fleet_total = (
    fleet_df[
        (fleet_df["Fuel Type"] == "All fuel types") &
        (fleet_df["Vehicle Type"]        == "Light‑duty vehicle")
    ]
    .groupby("Province")["Vehicles nb"].sum()
)

# b) BEV stock per province (battery‑electric)
bev_total = (
    fleet_df[
        (fleet_df["Fuel Type"] == "Battery‑electric") &
        (fleet_df["Vehicle Type"]        == "Light‑duty vehicle")
    ]
    .groupby("Province")["Vehicles nb"].sum()
)

ev_share = (bev_total / fleet_total * 100).fillna(0).round(2)  # %

# inject into GeoJSON
for feat in canada_geo["features"]:
    prov_name = feat["properties"]["name"].replace("Province of ", "")
    feat["properties"]["ev_share"] = ev_share.get(prov_name, 0.0)



# build folium map
m = folium.Map(location=[56.3, -96], zoom_start=4, tiles="cartodbpositron")

folium.Choropleth(
    geo_data=canada_geo,
    name="EV share",
    data=ev_share.reset_index(),
    columns=["Province", "Vehicles nb"],
    key_on="feature.properties.name",
    fill_color="YlGn",
    nan_fill_color="#dddddd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Battery‑electric share of light‑duty fleet (%)",
).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=900, height=550)


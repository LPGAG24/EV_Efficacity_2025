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


def gaussian_profile(mu: float, sigma: float, n: int = 48) -> np.ndarray:
    """Return a normalised 30‑min profile centred on ``mu`` hours."""
    t = np.linspace(0, 24, n, endpoint=False)
    prof = np.exp(-((t - mu) ** 2) / (2 * sigma * sigma))
    prof[prof < 0] = 0
    return prof / prof.sum()

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


# ── Vehicle selection ──────────────────────────────────────────────────────
st.sidebar.header("Vehicle")
selected_types = st.sidebar.multiselect(
    "Vehicle class",
    options=electric_eff.efficiency_by_vehicle_type["Vehicle class"].tolist(),
)
makes = sorted(electric_eff.data["Make"].unique())
selected_make = st.sidebar.selectbox("Make", makes)
models = sorted(
    electric_eff.data[electric_eff.data["Make"] == selected_make]["Model"].unique()
)
selected_model = st.sidebar.selectbox("Model", models)

model_row = electric_eff[{"Make": selected_make, "Model": selected_model}]
if not model_row.empty:
    eff = float(model_row["Combined (kWh/100 km)"].iloc[0])
    selected_class = model_row["Vehicle class"].iloc[0]
else:
    eff = 18.0
    selected_class = None

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

# --- Get number of vehicles of that class in province
if selected_class:
    try:
        total_vehicles = dist[(province, selected_class)]["Vehicles nb"].sum()
    except Exception:
        total_vehicles = 0
else:
    total_vehicles = 0

ev_share = st.sidebar.number_input(
    "EV share (%)",
    min_value=0,
    max_value=100,
    value=10,
    step=1,
)
default_count = int(total_vehicles * ev_share / 100)
car_count = st.sidebar.number_input(
    "Number of EVs",
    min_value=0,
    value=default_count,
    step=1,
)

# 'eff' was already derived from selected model above

# --- Get avg distance driven per day for that province (use CarUsage or fallback)
try:
    cu = CarUsage()
    cu.fetchData()
    dist_per_day = cu[{"Province": province}]  # returns DataFrame
    avg_distance = dist_per_day[f"{'Weekday' if selected_day in cu.weekdays else 'Weekend'}_km"].values[0]
except Exception:
    avg_distance = 30

# --- Custom charging profiles -----------------------------------------------
st.sidebar.header("Charging profiles")
home_mu = st.sidebar.number_input(
    "Home peak hour",
    min_value=0.0,
    max_value=23.5,
    value=18.0,
    step=0.5,
)
home_sigma = st.sidebar.number_input(
    "Home σ",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
)
work_mu = st.sidebar.number_input(
    "Work peak hour",
    min_value=0.0,
    max_value=23.5,
    value=9.0,
    step=0.5,
)
work_sigma = st.sidebar.number_input(
    "Work σ",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
)

home_speed = st.sidebar.number_input("Home charger speed (kW)", value=7.2)
work_speed = st.sidebar.number_input("Work charger speed (kW)", value=11.0)

home_only = st.sidebar.checkbox("Home only", value=False)
work_only = st.sidebar.checkbox("Work only", value=False)
if home_only:
    home_share = 1.0
elif work_only:
    home_share = 0.0
else:
    home_share = st.sidebar.number_input(
        "Home charging %",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
    ) / 100

work_share = 1.0 - home_share

home_profile = gaussian_profile(home_mu, home_sigma)
work_profile = gaussian_profile(work_mu, work_sigma)

def profile_chart(prof, title):
    df = pd.DataFrame({
        "Time": [f"{i//2:02d}:{'30' if i%2 else '00'}" for i in range(48)],
        "Prob": prof
    })
    return alt.Chart(df).mark_bar().encode(x='Time', y='Prob').properties(title=title, width=700, height=250)

st.altair_chart(profile_chart(home_profile, "Home arrival distribution"))
st.altair_chart(profile_chart(work_profile, "Work arrival distribution"))

# --- Compute energy per 30-min slot -----------------------------------------
energy_per_car = avg_distance * eff / 100
time_bins = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

home_energy = home_share * car_count * energy_per_car * home_profile
work_energy = work_share * car_count * energy_per_car * work_profile

energy_df = pd.DataFrame({
    "Time": time_bins,
    "Home_kWh": home_energy,
    "Work_kWh": work_energy
})
energy_df["Total_kWh"] = energy_df["Home_kWh"] + energy_df["Work_kWh"]

energy_long = energy_df.melt(id_vars="Time", value_vars=["Home_kWh", "Work_kWh"], var_name="Source", value_name="kWh")
chart = alt.Chart(energy_long).mark_area(opacity=0.7).encode(
    x=alt.X('Time', sort=None),
    y=alt.Y('kWh', stack=None),
    color='Source'
).properties(
    title=f"Daily Charging Demand ({province}, {selected_model})",
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


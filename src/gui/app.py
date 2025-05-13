# app.py
import streamlit as st
import pandas as pd
import urllib.request, json, folium
from streamlit_folium import st_folium

from carDistribution import CarDistribution          # your classes
from carEfficiency   import CarEfficiency
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
fleet_df = fetch_statcan_fleet("23-10-0308-01")      # StatCan table, cached
provinces = sorted(fleet_df["Province"].unique())
province  = st.sidebar.selectbox("Province", provinces, index=provinces.index("Canada"))

# 2) Vehicle‑class multiselect (for efficiency tables)
selected_types = []

# ── Data prep ───────────────────────────────────────────────────────────────
# Distribution object (for fleet tables)
dist = CarDistribution(fleet_df, Province=province)

# Efficiency objects (electric & hybrid)
electric_eff = CarEfficiency(
    download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9")
)
hybrid_eff   = CarEfficiency(
    download_ckan_resource("8812228b-a6aa-4303-b3d0-66489225120d")
)

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

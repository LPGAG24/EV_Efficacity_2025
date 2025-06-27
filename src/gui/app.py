# app.py
import streamlit as st
import pandas as pd
import urllib.request, json, folium
from streamlit_folium import st_folium
import altair as alt
import numpy as np
import webbrowser

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
province  = st.sidebar.multiselect("Province", provinces, default=["Canada"])

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

@st.cache_resource(ttl=24 * 3600)       # ← 24 h de cache, ajuste si besoin
def load_car_usage() -> CarUsage:
    cu = CarUsage()
    cu.fetchData()              # charge les données de StatCan
    return cu                   # renvoie l’objet entier


electric_eff = CarEfficiency(load_electric_efficiency())
hybrid_eff   = CarEfficiency(load_hybrid_efficiency())


# ── Vehicle selection ──────────────────────────────────────────────────────
st.sidebar.header("Vehicle")
selected_types = st.sidebar.multiselect(
    "Vehicle class",
    options=electric_eff.efficiency_by_vehicle_type["Vehicle class"].tolist(),
    default=["Compact"]
)


show_vehicle_picker = st.sidebar.checkbox("Select vehicle")
if show_vehicle_picker:
    makes = sorted(electric_eff.data["Make"].unique())
    selected_make = st.sidebar.selectbox("Make", makes)
    models = sorted(
        electric_eff.data[electric_eff.data["Make"] == selected_make]["Model"].unique()
    )
    selected_model = st.sidebar.selectbox("Model", models, default=None)

    model_row = electric_eff[{"Make": selected_make, "Model": selected_model}]
    if not model_row.empty:
        eff = float(model_row["Combined (kWh/100 km)"].iloc[0])
        selected_class = model_row["Vehicle class"].iloc[0]
    else:
        eff = 18.0
        selected_class = None
    
    st.sidebar.write(f"🔋 {eff:.1f} kWh/100 km")
    if selected_class:
        st.sidebar.write(f"🚗 Class : {selected_class}")
else:
    #use mean of current vehicle class selected
    if selected_types:
        eff = electric_eff.efficiency_by_vehicle_type[
            electric_eff.efficiency_by_vehicle_type["Vehicle class"].isin(selected_types)
        ]
    else:
        eff = electric_eff.efficiency_by_vehicle_type

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
        st.table(dist.get_fuel_type_percent_by_vehicle(selected_types))

    with col2:
        st.header("2 · Efficiency (kWh/100 km)")
        tab1, tab2 = st.tabs(["⚡ Chart", "DataFrame"])

        df_e = electric_eff.efficiency_by_vehicle_type
        chart_df = (
            df_e[["Vehicle class", "Combined (Le/100 km)"]]
                .set_index("Vehicle class")          # x-axis
                .sort_index()                        # optional: alphabetic order
        )
        tab1.bar_chart(chart_df, use_container_width=True)
        tab2.dataframe(df_e, use_container_width=True)

# --- Sidebar: user selects day and vehicle class
day_choices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Select Day of Week", day_choices)

# --- Get number of vehicles of that class in province
if selected_types:
    try:
        total_vehicles = subset = dist[{"Province": province, "Vehicle Type": selected_types}]["Vehicles nb"].sum()
        #add percents of the vehicle types in total_vehicles
    except Exception:
        total_vehicles = 0
else:
    total_vehicles = 0

ev_share = st.sidebar.number_input(
    "EV share (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
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
    cu = load_car_usage()
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





# ── 1 · profils initiaux (gaussienne) ───────────────────────────────
home_prof_default = gaussian_profile(home_mu, home_sigma)   # ndarray 48×1
work_prof_default = gaussian_profile(work_mu, work_sigma)

# helper : fabrique un DataFrame Time/Prob pour l’éditeur
def profile_df(prof: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Time": [f"{i//2:02d}:{'30' if i%2 else '00'}" for i in range(48)],
        "Prob": prof,
    })

# ── 2 · Home — data_editor + renormalisation ───────────────────────
st.sidebar.subheader("Home arrival distribution (editable)")
home_df = profile_df(home_prof_default)

edited_home = st.data_editor(
    home_df,
    num_rows="fixed",
    column_config={
        "Prob": st.column_config.NumberColumn(step=0.01, min_value=0.0)
    },
    key="home_profile",            # clé pour conserver l’édition
    use_container_width=True,
)

edited_home["Prob"] = edited_home["Prob"].clip(lower=0)
edited_home["Prob"] = edited_home["Prob"] / edited_home["Prob"].sum()
home_profile = edited_home["Prob"].to_numpy()          # ← vecteur final

st.altair_chart(
    alt.Chart(edited_home)
        .mark_bar()
        .encode(x="Time", y="Prob")
        .properties(title="Home arrival distribution", width=700, height=250),
    use_container_width=True,
)

# ── 3 · Work — même logique ─────────────────────────────────────────
st.sidebar.subheader("Work arrival distribution (editable)")
work_df = profile_df(work_prof_default)

edited_work = st.data_editor(
    work_df,
    num_rows="fixed",
    column_config={
        "Prob": st.column_config.NumberColumn(step=0.01, min_value=0.0)
    },
    key="work_profile",
    use_container_width=True,
)

edited_work["Prob"] = edited_work["Prob"].clip(lower=0)
edited_work["Prob"] = edited_work["Prob"] / edited_work["Prob"].sum()
work_profile = edited_work["Prob"].to_numpy()

st.altair_chart(
    alt.Chart(edited_work)
        .mark_bar()
        .encode(x="Time", y="Prob")
        .properties(title="Work arrival distribution", width=700, height=250),
    use_container_width=True,
)






# --- Compute energy per 30-min slot -----------------------------------------
eff.loc[:, "Combined (Le/100 km)"] = (
    eff["Combined (Le/100 km)"] * avg_distance / 100
)
time_bins = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

#Home share * car_count * (energy_per_car @ percent of that same car type) * home profile

efficiency_total_number = 0.0
for vprov in provinces:
    for vtype in selected_types:
        eff_i  = eff.loc[eff["Vehicle class"] == vtype, "Combined (Le/100 km)"][0]
        dist_i = dist.data.loc[(dist.data["Vehicle Type"] == vtype) & (dist.data["Province"] == vprov)] ["Vehicles nb"].sum()
        efficiency_total_number += float(eff_i) * float(dist_i)
home_energy = home_share * efficiency_total_number * home_profile * ev_share
work_energy = work_share * efficiency_total_number * work_profile * ev_share

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
    title=f"Daily Charging Demand ({province}, "
    f"{selected_types if selected_types is not None else 'Average'} Vehicle)",
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
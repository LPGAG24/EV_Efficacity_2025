# app.py
import streamlit    as st
import pandas       as pd
import numpy        as np
import altair       as alt
import urllib.request, json, folium

from streamlit_folium   import st_folium
from aggregate_power    import aggregate_power
from carDistribution    import *
from carEfficiency      import *
from carRecharge        import *
from carUsage           import *
from appHelper          import *
from data_prep_canada   import fetch_statcan_fleet, download_ckan_resource


# resolution (number of slots in a 24 h day)
n_res = 48


st.set_page_config(page_title="EV & Hybrid Dashboard", layout="wide")

# ── GeoJSON helper (get number of cars)─────────────────────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_canada_geojson():
    url = (
        "https://raw.githubusercontent.com/"
        "codeforgermany/click_that_hood/master/public/data/canada.geojson"
    )
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)

canada_geo = load_canada_geojson()

# ── Sidebar controls ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")

SLOT_MINUTES_OPTIONS = [60, 30, 15, 10, 5, 2, 1]
N_RES_OPTIONS        = [1440 // m for m in SLOT_MINUTES_OPTIONS]   # [24,48,96,…]
N_RES_DEFAULT = 2

st.sidebar.selectbox("Durée d’un créneau (min)", SLOT_MINUTES_OPTIONS, index=N_RES_DEFAULT, key="slot_minutes", format_func=lambda x: f"{x:g} min", on_change=sync_from_slot)
st.sidebar.selectbox("Nombre de créneaux par jour", N_RES_OPTIONS, index=N_RES_DEFAULT, key="n_res", on_change=sync_from_nres)

slot_minutes = st.session_state.slot_minutes
n_res        = st.session_state.n_res    # ← voilà « n_res » pour la suite

# store number of custom profiles for dynamic editing
if "custom_profile_count" not in st.session_state:
    st.session_state.custom_profile_count = 1

st.sidebar.markdown(f"**Sélection actuelle** : {n_res} créneaux × {slot_minutes} min = 1 440 min")

@st.cache_data
def load_fleet():
    # slow: download or load data
    return fetch_statcan_fleet("23-10-0308-01")
fleet_df = load_fleet()

provinces = sorted(fleet_df["Province"].unique())
province  = st.sidebar.multiselect("Province", provinces, default=["Canada"])
if not province:
    province = ["Canada"]

# 2) Vehicle‑class multiselect (for efficiency tables)
selected_types = []

# ── Data prep ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Distribution object (for fleet tables)
@st.cache_resource
def get_dist(fleet_df, province):
    # fast: in-memory DataFrame processing
    if not province:
        province = ["Canada"]
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


# ── Vehicle selection ───────────────────────────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Vehicle")
selected_types = st.sidebar.multiselect(
    "Vehicle class",
    options=electric_eff.efficiency_by_vehicle_type["Vehicle class"].tolist(),
    default=["Compact"]
)


recharge_time = 8.0
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
        recharge_time = float(model_row["Recharge time (h)"].iloc[0])
        selected_class = model_row["Vehicle class"].iloc[0]
    else:
        eff = 18.0
        recharge_time = 8.0
        selected_class = None
    
    st.sidebar.write(f"🔋 {eff:.1f} kWh/100 km")
    if selected_class:
        st.sidebar.write(f"🚗 Class : {selected_class}")
    st.sidebar.write(f"⏱ {recharge_time} h recharge")
else:
    #use mean of current vehicle class selected
    if selected_types:
        eff = electric_eff.efficiency_by_vehicle_type[
            electric_eff.efficiency_by_vehicle_type["Vehicle class"].isin(selected_types)
        ]
    else:
        eff = electric_eff.efficiency_by_vehicle_type
    if selected_types:
        recharge_time = pd.to_numeric(
            electric_eff.data[electric_eff.data["Vehicle class"].isin(selected_types)][
                "Recharge time (h)"
            ],
            errors="coerce",
        ).mean()
    else:
        recharge_time = pd.to_numeric(
            electric_eff.data["Recharge time (h)"], errors="coerce"
        ).mean()

# ── Layout ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
st.title(f"🚗 EV & Hybrid Dashboard — {province}")

# 1 & 2: tables and efficiency (your original code)
with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("1 · Fleet distribution")
        st.subheader("Fuel mix")
        st.table(dist.get_fuel_type())

        st.subheader("Vehicle‑type mix")
        vt_df = dist.get_fuel_type_percent_by_vehicle(selected_types).reset_index()
        vt_df.columns = ["Vehicle Type", "Percent"]
        edited_vt = st.data_editor(vt_df, num_rows="dynamic")
        total_pct = edited_vt["Percent"].sum()
        if total_pct < 100:
            diff = 100 - total_pct
            mask = edited_vt["Vehicle Type"] == "Other"
            if mask.any():
                edited_vt.loc[mask, "Percent"] += diff
            else:
                new_row = pd.DataFrame([{"Vehicle Type": "Other", "Percent": diff}])
                edited_vt = pd.concat([edited_vt, new_row], ignore_index=True)
        elif total_pct > 100:
            st.warning("Vehicle type percentages exceed 100%")
        st.table(edited_vt)

    with col2:
        st.header("2 · Efficiency (kWh/100 km) and Battery size")
        tab1, tab2 = st.tabs(["⚡ Chart", "DataFrame"])
        electric_eff.set_efficiency_by_type(selected_types)
        electric_eff.set_battery_by_type(selected_types)
        df_e = electric_eff.efficiency_by_vehicle_type
        df_b = electric_eff.battery_by_vehicle_type
        df_merge = df_e.merge(df_b, on="Vehicle class")
        chart_df = (
            df_merge[["Vehicle class", "Combined (Le/100 km)", "Battery_kWh"]]
                .set_index("Vehicle class")          # x-axis
                .sort_index()                        # optional: alphabetic order
        )
        tab1.bar_chart(chart_df, use_container_width=True)
        tab2.dataframe(df_merge, use_container_width=True)

        st.sidebar.subheader("Battery size (kWh)")
        battery_inputs = {}
        for _, row in df_b.iterrows():
            battery_inputs[row["Vehicle class"]] = st.sidebar.number_input(
                row["Vehicle class"], value=float(row["Battery_kWh"]), step=1.0
            )

# --- Sidebar: user selects day and vehicle class
day_choices = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Select Day of Week", day_choices)

# --- Get number of vehicles of that class in selected provinces
total_vehicles = count_vehicles(dist, province, selected_types)

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


# Get avg distance driven per day for that province (use CarUsage or fallback)
try:
    cu = load_car_usage()
    dist_per_day = cu[{"Province": province}]  # returns DataFrame
    avg_distance = dist_per_day[f"{'Weekday' if selected_day in cu.weekdays else 'Weekend'}_km"].values[0]
except Exception:
    avg_distance = 30

# ─────── Custom charging profiles ───────────────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Charging profiles")
profile_mode = st.sidebar.radio(
    "Mode", ["Normal", "Custom"], horizontal=True, key="profile_mode"
)

if profile_mode == "Normal":
    home_share_input = st.sidebar.number_input(
        "Home charging %", min_value=0, max_value=100, value=60, step=1
    )
    work_share_input = st.sidebar.number_input(
        "Work charging %", min_value=0, max_value=100, value=30, step=1
    )

    total_share = home_share_input + work_share_input
    if total_share > 100:
        st.sidebar.warning("Home + Work share exceeds 100%")

    custom_share_display = max(0, 100 - total_share)
    st.sidebar.markdown(f"Custom charging %: {custom_share_display}")

    home_share = home_share_input / 100
    work_share = work_share_input / 100
    custom_share = custom_share_display / 100
else:
    home_share = 0.0
    work_share = 0.0
    custom_share = 1.0


# ────── Arrival distributions ────────────────────────────────────────────────
categories = []

if profile_mode == "Normal":
    home = arrival_profile_editor(
        "Home arrival distribution",
        n_slots=n_res,
        mu0=18.0,
        sigma0=2.0,
        speed0=7.2,
        key="home",
    )
    categories.append({
        "share": home_share,
        "profile": home["profile"],
        "speed": home["kW"],
        "label": "Level 1 Charger",
    })

    work = arrival_profile_editor(
        "Work arrival distribution",
        n_slots=n_res,
        mu0=9.0,
        sigma0=2.0,
        speed0=11.0,
        key="work",
    )
    categories.append({
        "share": work_share,
        "profile": work["profile"],
        "speed": work["kW"],
        "label": "Level 2 Charger",
    })
else:
    plus, minus = st.sidebar.columns(2)
    if plus.button("+"):
        st.session_state.custom_profile_count += 1
    if minus.button("-") and st.session_state.custom_profile_count > 1:
        st.session_state.custom_profile_count -= 1

    shares = []
    for i in range(st.session_state.custom_profile_count):
        name = st.sidebar.text_input(
            f"Name {i+1}", value=f"Profile {i+1}", key=f"name_{i}"
        )
        share = st.sidebar.number_input(
            f"{name} %", 0, 100, 100 // st.session_state.custom_profile_count,
            key=f"share_{i}"
        )
        prof = arrival_profile_editor(
            f"{name} arrival distribution",
            n_slots=n_res,
            mu0=12.0,
            sigma0=2.0,
            speed0=7.2,
            key=f"custom_{i}",
        )
        categories.append(
            {
                "share": share / 100,
                "profile": prof["profile"],
                "speed": prof["kW"],
                "label": name,
            }
        )
        shares.append(share)

    if sum(shares) > 100:
        st.sidebar.warning("Total share exceeds 100%")

# --- Compute charging cars and electric demand ------------------------------
time_bins, n_slots, slot_len = compute_time_bins(n_res, recharge_time)

cars_df = pd.DataFrame({"Time": time_bins})
power_df = pd.DataFrame({"Time": time_bins})
arrivals_list = []
kernels_list = []

for cat in categories:
    conv = np.convolve(cat["profile"], np.ones(n_slots), mode="same")
    cars = cat["share"] * car_count * conv
    cars_df[cat["label"]] = cars
    power_df[cat["label"]] = cars * cat["speed"]
    arrivals_list.append(cat["share"] * car_count * cat["profile"])
    kernels_list.append(np.full(n_slots, cat["speed"]))

cars_df["Total_cars"] = cars_df[[c["label"] for c in categories]].sum(axis=1)
power_df["Total_kW"] = power_df[[c["label"] for c in categories]].sum(axis=1)

arrivals_mat = np.column_stack(arrivals_list)
kernels = np.column_stack(kernels_list)
power_df["Agg_kW"] = aggregate_power(arrivals_mat, kernels)

value_vars = [c["label"] for c in categories]
cars_long = cars_df.melt(id_vars="Time", value_vars=value_vars,
                         var_name="Source", value_name="Cars")
chart_cars = (
    alt.Chart(cars_long)
    .mark_area(opacity=0.7)
    .encode(
        x=alt.X("Time", sort=None),
        y=alt.Y("Cars", stack=None),
        color=alt.Color("Source:N", title="Charging location"),
    )
    .properties(
        title=f"Daily number of cars state ({province}, "
        f"{selected_types if selected_types is not None else 'Average'} Vehicle)",
        width=900,
        height=350,
    )
)
st.altair_chart(chart_cars, use_container_width=True)

power_long = power_df.melt(id_vars="Time", value_vars=value_vars,
                           var_name="Source", value_name="kW")

area_chart = alt.Chart(power_long).mark_line(color="blue").encode(
    x=alt.X("Time", sort=None),
    y=alt.Y("kW", stack=None),
    color="Source",
)

line_chart = (
    alt.Chart(power_df)
    .mark_area(opacity=0.5)
    .encode(x="Time", y="Agg_kW")
)

chart_power = (
    area_chart + line_chart
).properties(
    title=f"Electric demand ({province}, "
    f"{selected_types if selected_types is not None else 'Average'} Vehicle)",
    width=900,
    height=350,
)
st.altair_chart(chart_power, use_container_width=True)
slot_hours = slot_len
power_df["Energy_kWh"] = power_df["Agg_kW"] * slot_hours
week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
week_frames = []
for day in week_days:
    df_day = power_df[["Time", "Energy_kWh"]].copy()
    df_day["Day"] = day
    week_frames.append(df_day)
weekly_profile = pd.concat(week_frames, ignore_index=True)

weekly_chart = (
    alt.Chart(weekly_profile)
    .mark_line()
    .encode(x=alt.X("Time", sort=None), y="Energy_kWh", color="Day")
    .properties(title="Weekly electric consumption", width=700, height=300)
)
st.altair_chart(weekly_chart, use_container_width=True)


# ── 3 · Canada EV‑share choropleth ──────────────────────────────────────────
st.header("3 · Provincial EV share (battery‑electric)")

fleet_total = (
    fleet_df[
        (fleet_df["Fuel Type"] == "All fuel types") &
        (fleet_df["Vehicle Type"]        == "Light‑duty vehicle")
    ]
    .groupby("Province")["Vehicles nb"].sum()
)

bev_total = (
    fleet_df[
        (fleet_df["Fuel Type"] == "Battery‑electric") &
        (fleet_df["Vehicle Type"]        == "Light‑duty vehicle")
    ]
    .groupby("Province")["Vehicles nb"].sum()
)

ev_share = (bev_total / fleet_total * 100).fillna(0).round(2)  # %

for feat in canada_geo["features"]:
    prov_name = feat["properties"]["name"].replace("Province of ", "")
    feat["properties"]["ev_share"] = ev_share.get(prov_name, 0.0)

m = folium.Map(location=[56.3, -96], zoom_start=4, tiles="cartodbpositron")
folium.Choropleth(
    geo_data=canada_geo, name="EV share", data=ev_share.reset_index(),
    columns=["Province", "Vehicles nb"], key_on="feature.properties.name",
    fill_color="YlGn", nan_fill_color="#dddddd", fill_opacity=0.7,
    line_opacity=0.2, legend_name="Battery‑electric share of light‑duty fleet (%)",
).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=900, height=550)
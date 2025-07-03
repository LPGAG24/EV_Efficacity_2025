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
home_share_input = st.sidebar.number_input(
    "Home charging %", min_value=0, max_value=100, value=60, step=1
)
work_share_input = st.sidebar.number_input(
    "Work charging %", min_value=0, max_value=100, value=30, step=1
)

total_share = home_share_input + work_share_input
if total_share > 100:
    st.sidebar.warning("Home + Work share exceeds 100%")

custom_share = max(0, 100 - total_share)
st.sidebar.markdown(f"Custom charging %: {custom_share}")

home_share = home_share_input / 100
work_share = work_share_input / 100
custom_share = custom_share / 100

# ────── Home and work arrival distributions ────────────────────────────────────────────────────────────────────────────────────
# ── Home arrival distribution ─────────────────────────────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Home arrival distribution (editable)")
    
    profile_mu_step: float = 0.1
    
    profile_type = st.radio("Type de profil", ["Symmetrical", "Asymmetrical"], horizontal=True)
    if profile_type == "Symmetrical":
        col1, col2, col3 = st.columns(3)
        home_mu =       col1.number_input("Home peak hour", 0.0, 23.5, value=18.0, step=0.5)
        home_sigma =    col2.number_input("Home σ (common)", 0.1, 10.0, value=2.0, step=profile_mu_step)
        home_speed =    col3.number_input("Home charger speed (kW)", value=7.2)
        home_prof_default = gaussian_profile(home_mu, home_sigma, n_res) # symétrique
    else:
        col1, col2, col3, col4 = st.columns(4)
        home_mu =           col1.number_input("Home peak hour", 0.0, 23.5, value=18.0, step=0.5)
        home_sigma_left =   col2.number_input("Home σ (left side)", 0.1, 10.0, value=2.0, step=profile_mu_step)
        home_sigma_right =  col3.number_input("Home σ (right side)", 0.1, 10.0, value=2.0, step=profile_mu_step)
        home_speed =        col4.number_input("Home charger speed (kW)", value=7.2)
        home_prof_default = gaussian_profile(home_mu, home_sigma_left, n_res, home_sigma_right) # asymétrique

    with st.expander("Afficher / masquer le DataFrame", expanded=False):
        home_df = profile_df(home_prof_default)
        edited_home = st.data_editor(
            home_df,
            num_rows="fixed",
            column_config={
                "Prob": st.column_config.NumberColumn(step=0.01, min_value=0.0)
            },
            key="home_profile",
            use_container_width=True,
        )
        edited_home["Prob"] = edited_home["Prob"].clip(lower=0)
        edited_home["Prob"] = edited_home["Prob"] / edited_home["Prob"].sum()
        home_profile = edited_home["Prob"].to_numpy()
        
    st.altair_chart(alt.Chart(edited_home).mark_bar().encode(x="Time", y="Prob").properties(title="Home arrival distribution", width=700, height=250),use_container_width=True,)

# ── Work arrival distribution ─────────────────────────────────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Work arrival distribution (editable)")
    work_profile_type = st.radio("Type de profil (Work)", ["Symmetrical", "Asymmetrical"],horizontal=True, key="work_profile_type")

    if work_profile_type == "Symmetrical":
        col1, col2, col3 = st.columns(3)
        work_mu =       col1.number_input("Work peak hour", 0.0, 23.5, value=9.0, step=0.5, key="work_mu")
        work_sigma =    col2.number_input("Work σ (common)", 0.5, 5.0, value=2.0, step=profile_mu_step, key="work_sigma")
        work_speed =    col3.number_input("Work charger speed (kW)", value=11.0, key="work_speed")
        work_prof_default = gaussian_profile(work_mu, work_sigma, n_res) # symétrique

    else:
        col1, col2, col3, col4 = st.columns(4)
        work_mu =           col1.number_input("Work peak hour", 0.0, 23.5, value=9.0, step=0.5, key="work_mu")
        work_sigma_left =   col2.number_input("Work σ (left side)", 0.5, 5.0, value=2.0, step=profile_mu_step,key="work_sigma_left")
        work_sigma_right =  col3.number_input("Work σ (right side)", 0.5, 5.0, value=2.0, step=profile_mu_step,key="work_sigma_right")
        work_speed =        col4.number_input("Work charger speed (kW)", value=11.0, key="work_speed")
        work_prof_default = gaussian_profile(work_mu, work_sigma_left, n_res, work_sigma_right) # asymétrique

    with st.expander("Afficher / masquer le DataFrame", expanded=False):
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

    # ── Graphe de distribution ────────────────────────────────────────────────
    st.altair_chart(alt.Chart(edited_work).mark_bar().encode(x="Time", y="Prob").properties(title="Work arrival distribution", width=700, height=250),use_container_width=True,)

# ── Custom arrival distribution ─────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Custom arrival distribution (editable)")
    custom_mu = st.number_input("Custom peak hour", 0.0, 23.5, value=12.0, step=0.5)
    custom_sigma = st.number_input("Custom σ", 0.1, 10.0, value=2.0, step=0.1)
    custom_speed = st.number_input("Custom charger speed (kW)", value=7.2)
    custom_default = gaussian_profile(custom_mu, custom_sigma, n_res)
    with st.expander("Afficher / masquer le DataFrame", expanded=False):
        custom_df = profile_df(custom_default)
        edited_custom = st.data_editor(
            custom_df,
            num_rows="fixed",
            column_config={"Prob": st.column_config.NumberColumn(step=0.01, min_value=0.0)},
            key="custom_profile",
            use_container_width=True,
        )
        edited_custom["Prob"] = edited_custom["Prob"].clip(lower=0)
        edited_custom["Prob"] = edited_custom["Prob"] / edited_custom["Prob"].sum()
        custom_profile = edited_custom["Prob"].to_numpy()

    st.altair_chart(
        alt.Chart(edited_custom).mark_bar().encode(x="Time", y="Prob").properties(
            title="Custom arrival distribution", width=700, height=250
        ),
        use_container_width=True,
    )

# --- Compute charging cars and electric demand ------------------------------
time_bins, n_slots, slot_len = compute_time_bins(n_res, recharge_time)

home_conv = np.convolve(home_profile, np.ones(n_slots), mode="same")
work_conv = np.convolve(work_profile, np.ones(n_slots), mode="same")
custom_conv = np.convolve(custom_profile, np.ones(n_slots), mode="same")

home_cars = home_share * car_count * home_conv
work_cars = work_share * car_count * work_conv
custom_cars = custom_share * car_count * custom_conv

cars_df = pd.DataFrame({
    "Time": time_bins,
    "Home_cars": home_cars,
    "Work_cars": work_cars,
    "Custom_cars": custom_cars,
})
cars_df["Total_cars"] = cars_df[["Home_cars", "Work_cars", "Custom_cars"]].sum(axis=1)

rename = {
    "Home_cars": "Level 1 Charger",
    "Work_cars": "Level 2 Charger",
    "Custom_cars": "Custom",
}
cars_long = cars_df.rename(columns=rename).melt(
    id_vars="Time",
    value_vars=["Level 1 Charger", "Level 2 Charger", "Custom"],
    var_name="Source", value_name="Cars"
)
chart_cars = (
    alt.Chart(cars_long)
    .mark_area(opacity=0.7)
    .encode(x=alt.X("Time", sort=None), y=alt.Y("Cars", stack=None), color=alt.Color("Source:N", title="Charging location"))
    .properties(title=f"Daily number of cars state ({province}, " f"{selected_types if selected_types is not None else 'Average'} Vehicle)", width=900, height=350)
)

st.altair_chart(chart_cars, use_container_width=True)

power_home = home_cars * home_speed
power_work = work_cars * work_speed
power_custom = custom_cars * custom_speed

power_df = pd.DataFrame({
    "Time": time_bins,
    "Home_kW": power_home,
    "Work_kW": power_work,
    "Custom_kW": power_custom,
})
power_df["Total_kW"] = power_df[["Home_kW", "Work_kW", "Custom_kW"]].sum(axis=1)

# ── Aggregate power demand ─────────────────────────────────────────────────────────────────────────────────────────────────────
arrivals_mat = np.column_stack([
    home_share * car_count * home_profile,
    work_share * car_count * work_profile,
    custom_share * car_count * custom_profile,
])
kernels = np.column_stack([
    np.full(n_slots, home_speed),
    np.full(n_slots, work_speed),
    np.full(n_slots, custom_speed),
])
power_df["Agg_kW"] = aggregate_power(arrivals_mat, kernels)

power_long = power_df.melt(id_vars="Time", value_vars=["Home_kW", "Work_kW", "Custom_kW"],
                           var_name="Source", value_name="kW")


area_chart = alt.Chart(power_long).mark_line(color='blue').encode(
    x=alt.X('Time', sort=None),
    y=alt.Y('kW', stack=None),
    color='Source'
)

line_chart = alt.Chart(power_df).mark_area(opacity=0.5).encode(x='Time', y='Agg_kW')

chart_power = (area_chart + line_chart).properties(title=f"Electric demand ({province}, "f"{selected_types if selected_types is not None else 'Average'} Vehicle)", width=900, height=350)
st.altair_chart(chart_power, use_container_width=True)

# --- Weekly energy graph -----------------------------------------------------
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
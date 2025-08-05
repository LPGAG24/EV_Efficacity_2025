# app.py
import streamlit    as st
import pandas       as pd
import numpy        as np
import altair       as alt
import urllib.request, json, folium

from streamlit_folium   import st_folium

from carDistribution    import *
from carEfficiency      import *
from carRecharge        import *
from carUsage           import *
from traitlets import default
from appHelper          import *
from data_prep_canada   import fetch_statcan_fleet, download_ckan_resource
from util.calendar      import build_calendar


def _get_shares(prefix: str, visible: bool = True) -> tuple[float, float, float]:
    """Return home, work and custom charging shares for the given prefix."""
    key_home = f"home_share_{prefix.lower()}"
    key_work = f"work_share_{prefix.lower()}"
    if visible:
        h_val = st.sidebar.number_input(
            f"Home charging % ({prefix})",
            0, 100, st.session_state.get(key_home, 60), step=1, key=key_home,
        )
        w_val = st.sidebar.number_input(
            f"Work charging % ({prefix})",
            0, 100, st.session_state.get(key_work, 30), step=1, key=key_work,
        )
    else:
        h_val = st.session_state.get(key_home, 60)
        w_val = st.session_state.get(key_work, 30)
    total = h_val + w_val
    scale = 100 / total if total > 100 else 1.0
    home_share = (h_val * scale) / 100
    work_share = (w_val * scale) / 100
    custom_share = 1.0 - home_share - work_share
    return home_share, work_share, custom_share


def _profile_from_state(key: str, mu0: float) -> dict:
    """Reconstruct arrival profile from session state."""
    mu = st.session_state.get(f"{key}_mu", mu0)
    sl = st.session_state.get(f"{key}_sl", 2.0)
    sr = st.session_state.get(f"{key}_sr", sl)
    lvl1 = st.session_state.get(f"{key}_lvl1", 100.0)
    lvl2 = st.session_state.get(f"{key}_lvl2", 0.0)
    lvl3 = st.session_state.get(f"{key}_lvl3", 0.0)
    p1 = st.session_state.get(f"{key}_p1", DEFAULT_CHARGER_KW[0])
    p2 = st.session_state.get(f"{key}_p2", DEFAULT_CHARGER_KW[1])
    p3 = st.session_state.get(f"{key}_p3", DEFAULT_CHARGER_KW[2])
    total = lvl1 + lvl2 + lvl3
    ratios = (
        (lvl1 / total) if total else 0.0,
        (lvl2 / total) if total else 0.0,
        (lvl3 / total) if total else 0.0,
    )
    kw = ratios[0] * p1 + ratios[1] * p2 + ratios[2] * p3
    prof = gaussian_profile(mu, sl, n_res, sr)
    return {"profile": prof, "kW": kw, "ratios": ratios, "kW_levels": (p1, p2, p3)}


def _build_categories(prefix: str, visible: bool = True) -> list[dict]:
    """Create charging categories for weekday or weekend."""
    weekend = prefix.lower() == "weekend"
    home_share, work_share, custom_share = _get_shares(prefix, visible)

    mu_home = 20.0 if weekend else 18.0
    if visible:
        home = arrival_profile_editor(
            f"{prefix} home arrival distribution",
            n_slots=n_res,
            mu0=mu_home,
            sigma0=2.0,
            ratio0=(100.0, 0.0, 0.0),
            key=f"home_{prefix.lower()}",
        )
    else:
        home = _profile_from_state(f"home_{prefix.lower()}", mu_home)
    categories = [
        {
            "share": home_share,
            "profile": home["profile"],
            "speed": home["kW"],
            "label": f"Home ({prefix})",
            "level_kW": home["kW_levels"],
            "ratios": home["ratios"],
        }
    ]

    mu_work = 10.0 if weekend else 9.0
    if visible:
        work = arrival_profile_editor(
            f"{prefix} work arrival distribution",
            n_slots=n_res,
            mu0=mu_work,
            sigma0=2.0,
            ratio0=(0.0, 100.0, 0.0),
            key=f"work_{prefix.lower()}",
        )
    else:
        work = _profile_from_state(f"work_{prefix.lower()}", mu_work)
    categories.append(
        {
            "share": work_share,
            "profile": work["profile"],
            "speed": work["kW"],
            "label": f"Work ({prefix})",
            "level_kW": work["kW_levels"],
            "ratios": work["ratios"],
        }
    )

    if custom_share > 0:
        if visible:
            other = arrival_profile_editor(
                f"{prefix} other arrival distribution",
                n_slots=n_res,
                mu0=12.0,
                sigma0=2.0,
                ratio0=(100.0, 0.0, 0.0),
                key=f"other_{prefix.lower()}",
            )
        else:
            other = _profile_from_state(f"other_{prefix.lower()}", 12.0)
        categories.append(
            {
                "share": custom_share,
                "profile": other["profile"],
                "speed": other["kW"],
                "label": f"Other ({prefix})",
                "level_kW": other["kW_levels"],
                "ratios": other["ratios"],
            }
        )

    return categories


def _compute_energy(categories: list[dict]) -> pd.DataFrame:
    """Return total power dataframe for given categories."""
    time_bins, n_slots, slot_len = compute_time_bins(n_res, recharge_time)
    level_power_df = pd.DataFrame({"Time": time_bins})
    arr_list = []
    kern_list = []
    for cat in categories:
        arr = cat["share"] * car_count * cat["profile"]
        arr_list.append(arr)
        kern_list.append(np.full(n_slots, cat["speed"]))
    if not arr_list:
        return level_power_df.assign(Agg_kW=0.0, Energy_Wh=0.0)
    arr_mat = np.column_stack(arr_list)
    kern_mat = np.column_stack(kern_list)
    total = aggregate_power(arr_mat, kern_mat)
    level_power_df["Agg_kW"] = total
    level_power_df["Energy_Wh"] = total * slot_len * 1000
    return level_power_df


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
selected_types = None

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
# hybrid_eff   = CarEfficiency(load_hybrid_efficiency())


# ── Vehicle selection ─────────────────────────────────────────────────────────
# Selection uses fleet distribution table; no sidebar multiselect

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
selected_eff = electric_eff

with st.container():
    st.header("1 · Fleet distribution")
    st.subheader("Vehicle fleet")
    # list of all vehicles with an editable "Percent" column and battery size
    vehicles_df = (
        electric_eff.data[
            [
                "Make",
                "Model",
                "Vehicle class",
                "Combined (kWh/100 km)",
                "Recharge time (h)",
                "Range (km)",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    vehicles_df["Battery (kWh)"] = (
        pd.to_numeric(vehicles_df["Range (km)"], errors="coerce")
        * pd.to_numeric(vehicles_df["Combined (kWh/100 km)"], errors="coerce")
        / 100.0
    ).pipe(lambda s: np.ceil(s / 5) * 5)
    # Percent column first, followed by computed battery size
    vehicles_df.insert(0, "Percent", 0.0)
    vehicles_df = vehicles_df[
        [
            "Percent",
            "Make",
            "Model",
            "Vehicle class",
            "Battery (kWh)",
            "Combined (kWh/100 km)",
            "Recharge time (h)",
            "Range (km)",
        ]
    ]
    edited_vehicles = st.data_editor(
        vehicles_df,
        height=300,
        use_container_width=True,
        key="fleet_editor",
        column_config={
            "Percent": st.column_config.NumberColumn(
                "Percent",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                default=0.0,
            )
        },
    )
    percent_rows = edited_vehicles[edited_vehicles["Percent"] > 0].copy()
    if not percent_rows.empty:
        percent_rows["Weight"] = percent_rows["Percent"] / 100.0
        merged = electric_eff.data.merge(
            percent_rows[["Make", "Model", "Weight"]],
            on=["Make", "Model"],
            how="inner",
        )
        weights = merged.pop("Weight")
        valid_rt = pd.to_numeric(merged["Recharge time (h)"], errors="coerce")
        if valid_rt.notna().any():
            recharge_time = np.average(
                valid_rt[valid_rt.notna()], weights=weights[valid_rt.notna()]
            )
        replicated = merged.loc[
            merged.index.repeat((weights * 100).round().astype(int))
        ].reset_index(drop=True)
        selected_eff = CarEfficiency(replicated)
    else:
        selected_eff = electric_eff



with st.container():
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Vehicle‑type mix")
        vt_df = dist.get_fuel_type_percent_by_vehicle(None).reset_index()
        vt_df.columns = ["Vehicle Type", "Percent"]
        edited_vt = st.data_editor(vt_df, num_rows="dynamic")
        selected_types = edited_vt["Vehicle Type"].tolist()
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

        vt_chart = (
            alt.Chart(edited_vt)
            .mark_bar()
            .encode(
                x=alt.X("Vehicle Type", title="Vehicle Type"),
                y=alt.Y("Percent", title="Fleet share (%)"),
            )
            .properties(title="Fleet distribution", width=450, height=300)
        )
        st.altair_chart(vt_chart, use_container_width=True)

    with col2:
        st.header("2 · Efficiency (kWh/100 km) and Battery size")
        tab1, tab2 = st.tabs(["⚡ Chart", "DataFrame"])
        selected_eff.set_efficiency_by_type(selected_types)
        selected_eff.set_battery_by_type(selected_types)
        df_e = selected_eff.efficiency_by_vehicle_type
        df_b = selected_eff.battery_by_vehicle_type
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
    min_value=0.0,  max_value=100.0,
    value=10.0,     step=1.0,
)
default_count = int(total_vehicles * ev_share / 100)
car_count = st.sidebar.number_input(
    "Number of EVs",    min_value=0,
    value=default_count,step=1,
)


# Get avg distance driven per day for that province (use CarUsage or fallback)
try:
    cu = load_car_usage()
    dist_per_day = cu[{"Province": province}]
    if weekend_mode:
        day_type = "Weekend"
    else:
        day_type = "Weekend" if selected_day in cu.weekends else "Weekday"
    avg_distance = dist_per_day[f"{day_type}_km"].values[0]
except Exception:
    avg_distance = 30

try:
    gauss_frames = []
    for prov in province:
        gdf = gaussian_private_vehicle(prov)
        gdf["Province"] = prov
        gauss_frames.append(gdf)
    gauss_df = pd.concat(gauss_frames, ignore_index=True)
    gauss_chart = (
        alt.Chart(gauss_df)
        .mark_line()
        .encode(
            x=alt.X("Time", title="Daily average time (minutes)", axis=alt.Axis(tickMinStep=5, tickCount=10)),
            y=alt.Y("Density"),
            color="Province",
        )
        .properties(title="Private vehicle daily time distribution")
    )
    st.altair_chart(gauss_chart, use_container_width=True)
except Exception:
    pass

# ─────── Custom charging profiles ───────────────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Charging profiles")
weekend_mode = st.sidebar.checkbox("Edit weekend profile", value=False, key="weekend_mode")
profile_mode = st.sidebar.radio(
    "Mode", ["Normal", "Custom"], horizontal=True, key="profile_mode"
)

if profile_mode == "Normal":
    categories_weekday = _build_categories("Weekday", not weekend_mode)
    categories_weekend = _build_categories("Weekend", weekend_mode)
    categories = categories_weekend if weekend_mode else categories_weekday
else:
    plus, minus = st.sidebar.columns(2)
    if plus.button("+"):
        st.session_state.custom_profile_count += 1
    if minus.button("-") and st.session_state.custom_profile_count > 1:
        st.session_state.custom_profile_count -= 1

    categories = []
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
            n_slots=n_res,  mu0=12.0,
            sigma0=2.0, ratio0=(100.0, 0.0, 0.0),
            key=f"custom_{i}",
        )
        categories.append(
            {
                "share": share / 100,
                "profile": prof["profile"],
                "speed": prof["kW"],
                "label": name,
                "level_kW": prof["kW_levels"],
                "ratios": prof["ratios"],
            }
        )
        shares.append(share)

    if sum(shares) > 100:
        st.sidebar.warning("Total share exceeds 100%")

    categories_weekday = categories
    categories_weekend = categories

# compute base power profiles for both day types
weekday_power_base = _compute_energy(categories_weekday)
weekend_power_base = _compute_energy(categories_weekend)

# --- Compute charging cars and electric demand ------------------------------
time_bins, n_slots, slot_len = compute_time_bins(n_res, recharge_time)

cars_df = pd.DataFrame({"Time": time_bins})
power_df = pd.DataFrame({"Time": time_bins})
level_power_df = pd.DataFrame({"Time": time_bins})

# accumulate arrival profiles and charging kernels for total demand
arrivals_list: list[np.ndarray] = []
kernels_list: list[np.ndarray] = []

# store arrivals and kernels for each charger level
level_arrivals: dict[str, list[np.ndarray]] = {}
level_kernels: dict[str, list[np.ndarray]] = {}

for cat in categories:
    arrivals = cat["share"] * car_count * cat["profile"]

    # cars charging simultaneously for this category
    cars = circular_convolve(arrivals, np.ones(n_slots))
    cars_df[cat["label"]] = cars

    # instantaneous power for this category
    power_df[cat["label"]] = circular_convolve(
        arrivals, np.full(n_slots, cat["speed"])
    )

    # record arrivals and kernels for total demand
    arrivals_list.append(arrivals)
    kernels_list.append(np.full(n_slots, cat["speed"]))

    # accumulate arrivals per charger level
    ratios = cat.get("ratios", (1.0, 0.0, 0.0))
    for i, kw in enumerate(cat["level_kW"]):
        level = f"Level {i+1}"
        level_arrivals.setdefault(level, []).append(arrivals * ratios[i])
        level_kernels.setdefault(level, []).append(np.full(n_slots, kw))

cars_df["Total_cars"] = cars_df[[c["label"] for c in categories]].sum(axis=1)
power_df["Total_kW"] = power_df[[c["label"] for c in categories]].sum(axis=1)

# total aggregated power across all categories
arrivals_mat = np.column_stack(arrivals_list)
kernels = np.column_stack(kernels_list)
total_power = aggregate_power(arrivals_mat, kernels)

# compute power demand by charger level
for level, arr_list in level_arrivals.items():
    arr_mat = np.column_stack(arr_list)
    kern_mat = np.column_stack(level_kernels[level])
    level_power_df[level] = aggregate_power(arr_mat, kern_mat)

level_cols = [c for c in level_power_df.columns if c.startswith("Level ")]
level_power_df["Agg_kW"] = level_power_df[level_cols].sum(axis=1)
level_power_df["Agg_kW"] = total_power

categ_vars = [c["label"] for c in categories]
cars_long = cars_df.melt(
    id_vars="Time", value_vars=categ_vars, var_name="Source", value_name="Cars"
)
cars_long["Cars_thousands"] = cars_long["Cars"] / 1000
cars_df["Total_thousands"] = cars_df["Total_cars"] / 1000

line_cars = (
    alt.Chart(cars_long)
    .mark_line()
    .encode(
        x=alt.X("Time", sort=None),
        y=alt.Y(
            "Cars_thousands",
            stack=None,
            title="Cars (thousands)",
            axis=alt.Axis(format="~s"),
        ),
        color=alt.Color("Source:N", title="Charging location"),
    )
)

area_total_cars = (
    alt.Chart(cars_df)
    .mark_area(opacity=0.5)
    .encode(
        x="Time",
        y=alt.Y(
            "Total_thousands",
            title="Cars (thousands)",
            axis=alt.Axis(format="~s"),
        ),
    )
)

chart_cars = (
    line_cars + area_total_cars
).properties(
    title=f"Cars charging over time ({', '.join(province)}, "
    f"{selected_types if selected_types is not None else 'Average'} vehicle)",
    width=900,
    height=350,
)
st.altair_chart(chart_cars, use_container_width=True)

level_vars = [c for c in level_power_df.columns if c.startswith("Level ")]
power_long = level_power_df.melt(
    id_vars="Time", value_vars=level_vars, var_name="Charger Level", value_name="kW"
)

line_levels = alt.Chart(power_long).mark_line().encode(
    x=alt.X("Time", sort=None),
    y=alt.Y(
        "kW",
        stack=None,
        title="Power (kW)",
        axis=alt.Axis(format="~s"),
    ),
    color=alt.Color("Charger Level", title="Charger level"),
)

area_total_power = (
    alt.Chart(level_power_df)
    .mark_area(opacity=0.5)
    .encode(
        x="Time",
        y=alt.Y("Agg_kW", title="Power (kW)", axis=alt.Axis(format="~s")),
    )
)

chart_power = (
    line_levels + area_total_power
).properties(
    title=f"Power demand over time ({', '.join(province)}, "
    f"{selected_types if selected_types is not None else 'Average'} vehicle)",
    width=900,
    height=350,
)
st.altair_chart(chart_power, use_container_width=True)
slot_hours = slot_len
weekday_power_base["Energy_Wh"] = weekday_power_base["Agg_kW"] * slot_hours * 1000
weekend_power_base["Energy_Wh"] = weekend_power_base["Agg_kW"] * slot_hours * 1000

with st.expander("Calendar", expanded=False):
    year = st.number_input("Year", min_value=2000, max_value=2100,
                           value=int(pd.Timestamp.today().year))
    cal_df = build_calendar(int(year))
    week_numbers = sorted(cal_df["Date"].apply(lambda d: d.isocalendar()[1]).unique())
    today_iso_week = pd.Timestamp.today().isocalendar()[1]
    default_week = today_iso_week if today_iso_week in week_numbers else week_numbers[0]
    week = st.number_input(
        "Week",
        min_value=int(min(week_numbers)),
        max_value=int(max(week_numbers)),
        value=int(default_week),
        step=1,)
    st.dataframe(cal_df, use_container_width=True)

week_df = cal_df[cal_df["Date"].apply(lambda d: d.isocalendar()[1]) == week]
week_days = [d.strftime("%a") for d in week_df["Date"]]
week_frames = []
for i, (_, row) in enumerate(week_df.iterrows()):
    src = weekend_power_base if row["Type"] == "Weekend" else weekday_power_base
    df_day = src[["Energy_Wh"]].copy()
    df_day["Hour"] = np.arange(len(src)) * slot_hours + i * 24
    df_day["Day"] = row["Date"].strftime("%a")
    week_frames.append(df_day)
weekly_profile = pd.concat(week_frames, ignore_index=True)

tick_vals = [i * 24 for i in range(len(week_days))]
weekly_chart = (
    alt.Chart(weekly_profile)
    .mark_line()
    .encode(
        x=alt.X(
            "Hour",
            scale=alt.Scale(domain=[0, 24 * len(week_days)]),
            axis=alt.Axis(
                values=tick_vals,
                labelExpr="['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][datum.value / 24]",
                title="Day",
            ),
        ),
        y=alt.Y(
            "Energy_Wh",
            title="Energy (Wh)",
            axis=alt.Axis(format="~s"),
        ),
        color="Day",
    )
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
    .groupby("Province",observed=True)["Vehicles nb"].sum()
)

bev_total = (
    fleet_df[
        (fleet_df["Fuel Type"] == "Battery‑electric") & (fleet_df["Vehicle Type"] == "Light‑duty vehicle")
    ].groupby("Province", observed=True)["Vehicles nb"].sum()
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

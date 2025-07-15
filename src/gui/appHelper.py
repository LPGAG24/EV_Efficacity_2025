import streamlit as st, altair as alt, numpy as np, pandas as pd
from carRecharge        import *
from carDistribution    import *

# Default charging speeds for the three charger levels (kW)
DEFAULT_CHARGER_KW = (
    1.9,    # Level 1
    7.2,    # Level 2
    50.0,   # Level 3
)

def profile_df(p):         # 24-slot → tidy DataFrame
    return pd.DataFrame({"Time": [f"{i//2:02d}:{'30' if i%2 else '00'}" for i in range(len(p))],
                         "Prob": p})
    
def compute_time_bins(resolution: int, recharge_time: float) -> tuple[list[str], int, float]:
    """Return time bin labels and slot parameters for a given resolution."""
    minutes_per_slot = 1440 // resolution
    time_bins = [
        f"{(i * minutes_per_slot) // 60:02d}:{(i * minutes_per_slot) % 60:02d}"
        for i in range(resolution)
    ]
    slot_len = 24 / resolution
    n_slots = max(1, int(recharge_time / slot_len))
    return time_bins, n_slots, slot_len

def sync_from_slot():
    """Quand l'utilisateur modifie slot_minutes, on met n_res à jour."""
    st.session_state["n_res"] = 1440 // st.session_state["slot_minutes"]

def sync_from_nres():
    """Quand l'utilisateur modifie n_res, on met slot_minutes à jour."""
    st.session_state["slot_minutes"] = 1440 // st.session_state["n_res"]
    
def count_vehicles(dist: CarDistribution, provinces: list[str], types: list[str] | None) -> int:
    """Return total vehicle stock for selected provinces and vehicle types."""
    df = dist.data
    mask = df["Province"].isin(provinces)
    if types:
        mask &= df["Vehicle Type"].isin(types)
    return int(df.loc[mask, "Vehicles nb"].sum())
    

    

def arrival_profile_editor(                 # compact UI helper
    title="Arrival dist.",
    n_slots=48,
    mu0=18.0,
    sigma0=2.0,
    ratio0=(100.0, 0.0, 0.0),
    power0=DEFAULT_CHARGER_KW,
    key="home",
):
    with st.expander(title, expanded=False):
        k = lambda s: f"{key}_{s}"             # unique Streamlit keys
        with st.container(border=True):
            st.subheader(f"{title} (editable)")
            asym = st.radio("Profil", ["Symmetric", "Asymmetric"],
                            horizontal=True, key=k("r")) == "Asymmetric"

            cols = st.columns(4 if asym else 3)
            mu = cols[0].number_input("Peak h", 0.0, 23.5, mu0, 0.5, key=k("mu"))
            sl = cols[1].number_input("σ left" if asym else "σ", 0.1, 10.0, sigma0, 0.1, key=k("sl"))
            sr = cols[2].number_input("σ right", 0.1, 10.0, sigma0, 0.1,
                                      key=k("sr")) if asym else sl
            ratio_cols = st.columns(3)
            lvl1 = ratio_cols[0].number_input(
                "Level 1 (%)", 0.0, 100.0, ratio0[0], step=1.0, key=k("lvl1")
            )
            lvl2 = ratio_cols[1].number_input(
                "Level 2 (%)", 0.0, 100.0, ratio0[1], step=1.0, key=k("lvl2")
            )
            lvl3 = ratio_cols[2].number_input(
                "Level 3 (%)", 0.0, 100.0, ratio0[2], step=1.0, key=k("lvl3")
            )

            power_cols = st.columns(3)
            p1 = power_cols[0].number_input(
                "Level 1 kW", 0.0, 350.0, power0[0], step=0.1, key=k("p1")
            )
            p2 = power_cols[1].number_input(
                "Level 2 kW", 0.0, 350.0, power0[1], step=0.1, key=k("p2")
            )
            p3 = power_cols[2].number_input(
                "Level 3 kW", 0.0, 350.0, power0[2], step=0.1, key=k("p3")
            )
            total = lvl1 + lvl2 + lvl3
            if total > 100:
                st.warning("Charger level ratios exceed 100%")
            ratios = (
                (lvl1 / total) if total else 0.0,
                (lvl2 / total) if total else 0.0,
                (lvl3 / total) if total else 0.0,
            )
            kw = ratios[0] * p1 + ratios[1] * p2 + ratios[2] * p3

            prof = gaussian_profile(mu, sl, n_slots, sr) if asym \
                   else gaussian_profile(mu, sl, n_slots)

            with st.expander("Données", False):
                df = st.data_editor(profile_df(prof),
                                    num_rows="fixed",
                                    column_config={"Prob": st.column_config.NumberColumn(step=0.01, min_value=0.0)},
                                    key=k("edit"),
                                    use_container_width=True)
                df["Prob"] = np.clip(df["Prob"], 0, None)
                df["Prob"] /= df["Prob"].sum()
                prof = df["Prob"].to_numpy()

            st.altair_chart(
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Time", title="Time"),
                    y=alt.Y("Prob", title="Probability"),
                )
                .properties(title=title, width=700, height=250),
                use_container_width=True,
            )

    return {
        "profile": prof,
        "mu": mu,
        "σ_L": sl,
        "σ_R": sr,
        "kW": kw,
        "ratios": ratios,
        "kW_levels": (p1, p2, p3),
    }



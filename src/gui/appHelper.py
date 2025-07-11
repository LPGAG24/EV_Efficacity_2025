import streamlit as st, altair as alt, numpy as np, pandas as pd
from carRecharge        import *
from carDistribution    import *

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
    speed0=7.2,
    key="home",
):
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
        kw = cols[-1].number_input("kW", value=speed0, key=k("kw"))

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
            alt.Chart(df).mark_bar().encode(x="Time", y="Prob")
            .properties(title=title, width=700, height=250),
            use_container_width=True,
        )

    return {"profile": prof, "mu": mu, "σ_L": sl, "σ_R": sr, "kW": kw}



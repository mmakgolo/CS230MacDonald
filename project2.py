"""
Name: Olorato MacDonald Makgolo    CS230: Section 6    Data: Top 2000 Global Companies
URL: https://cs230macdonald-ccruy8dswatyhy5jz6mzpn.streamlit.app/
Description: An interactive explorer of the Top2000 Global Companies
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Top2000 Global Companies Explorer",
    layout="wide"
)

# ─── Data Loading & DA1 ───────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)                                          # [PY3]
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    # [DA1] convert key columns to numeric (clean/manipulate via lambda)
    for col in ["Market Value ($billion)", "Sales ($billion)", "Profits ($billion)", 
                "Assets ($billion)", "Latitude", "Longitude"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # [DA1] split HQ location into City/State
    if "Headquarters Location" in df:
        df["City"]  = df["Headquarters Location"].apply(lambda s: s.split(",")[0].strip())
        df["State"] = df["Headquarters Location"].apply(lambda s: s.split(",")[1].strip() if "," in s else None)

    # [DA7] drop an unused column
    df = df.drop(columns=["Ticker"], errors="ignore")

    # [DA9] add normalized Market Value column (0–1 scale)
    mv = df["Market Value ($billion)"]
    df["MV_norm"] = (mv - mv.min()) / (mv.max() - mv.min())

    return df

df = load_data("Top2000CompaniesGlobally.csv")

# ─── Sidebar Filters & DA4/DA5 ────────────────────────────────────────
st.sidebar.header("Filters")
country   = st.sidebar.selectbox("Country",   ["All"] + sorted(df["Country"].dropna().unique()))  # [ST1]
continent = st.sidebar.selectbox("Continent", ["All"] + sorted(df["Continent"].dropna().unique()))  # [ST2]
n         = st.sidebar.slider("Top N companies", 5, 50, 10)                                     # [ST1]
bins      = st.sidebar.slider("Histogram bins",     5, 50, 20)                                 # [ST1]

def filter_df(data: pd.DataFrame) -> pd.DataFrame:
    if country != "All":
        data = data[data["Country"] == country]               # [DA4]
    if continent != "All":
        data = data[data["Continent"] == continent]           # [DA5]
    return data

filtered = filter_df(df)

# ─── Tabs ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Market Value Rankings", "Sales & Profits", "Global Map",
    "Leaderboard", "Profit Margin Dist.", "Stats & Pivot", "Key Insights"
])  # [ST4]

# ─── 1) Market Value Rankings & DA3 ─────────────────────────────────
with tabs[0]:
    st.subheader(f"Top {n} by Market Value")
    top_df = filtered.nlargest(n, "Market Value ($billion)")      # [DA3]
    # [DA2] also show sorted version
    sorted_df = filtered.sort_values("Market Value ($billion)", ascending=False)  # [DA2]
    fig = px.bar(top_df, x="Company", y="Market Value ($billion)",
                 text=top_df["Market Value ($billion)"].round(1),
                 labels={"Market Value ($billion)": "MV (B USD)"},
                 title="Market Value Rankings")
    fig.update_layout(xaxis_tickangle=-45, margin={"b":150})
    st.plotly_chart(fig, use_container_width=True)

# ─── 2) Sales & Profits Analysis ─────────────────────────────────────
with tabs[1]:
    st.subheader("Sales vs. Profits by Continent")
    scatter = filtered.dropna(subset=["Sales ($billion)","Profits ($billion)","Continent"])
    if scatter.empty:
        st.info("No data.")
    else:
        fig = px.scatter(scatter, x="Sales ($billion)", y="Profits ($billion)",
                         color="Continent", size="Market Value ($billion)",
                         hover_name="Company",
                         title="Sales & Profits Scatter")
        st.plotly_chart(fig, use_container_width=True)

# ─── 3) Global Map ───────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Global Company Locations")
    map_df = filtered.dropna(subset=["Latitude","Longitude"])
    if map_df.empty:
        st.info("No locations.")
    else:
        midpoint = (map_df["Latitude"].mean(), map_df["Longitude"].mean())
        deck = pdk.Deck(
            layers=[
                pdk.Layer("TileLayer", data=None,
                          get_tile_data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                pdk.Layer("ScatterplotLayer", data=map_df,
                          get_position=["Longitude","Latitude"],
                          get_fill_color=[34,139,34], get_radius=3, pickable=True)
            ],
            initial_view_state=pdk.ViewState(latitude=midpoint[0],
                                             longitude=midpoint[1], zoom=2)
        )
        st.pydeck_chart(deck)  # [MAP]

# ─── 4) Leaderboard ──────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Company Leaderboard")
    # [DA2] demonstrate sort_values again
    leaderboard = filtered.sort_values("Market Value ($billion)", ascending=False)
    st.dataframe(leaderboard.head(n))

# ─── 5) Profit Margin Distribution & DA9 ─────────────────────────────
with tabs[4]:
    st.subheader("Profit Margin Histogram")
    pm = filtered.dropna(subset=["Sales ($billion)","Profits ($billion)"]).copy()
    pm["Profit Margin (%)"] = (pm["Profits ($billion)"] / pm["Sales ($billion)"]) * 100  # [DA9]
    pm = pm[np.isfinite(pm["Profit Margin (%)"])]
    if pm.empty:
        st.info("No margins.")
    else:
        fig = px.histogram(pm, x="Profit Margin (%)", nbins=bins,
                           title="Profit Margin (%) Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ─── 6) Stats & Pivot Tables ─────────────────────────────────────────
with tabs[5]:
    st.subheader("Statistical Summary")
    stats_df = filtered[["Sales ($billion)","Profits ($billion)",
                         "Assets ($billion)","Market Value ($billion)"]]
    if stats_df.empty:
        st.info("No stats.")
    else:
        desc = stats_df.describe().loc[["mean","50%","std","min","max"]]
        desc.index = ["Mean","Median","Std","Min","Max"]
        st.table(desc)  # basic describe

        # [DA6] pivot: average profit margin by continent
        pivot = pd.pivot_table(filtered, index="Continent",
                               values="Profit Margin (%)", aggfunc="mean").round(2)
        st.markdown("**Avg Profit Margin by Continent**")
        st.table(pivot)

        # [DA7] group and sum Market Value by Continent
        grp = filtered.groupby("Continent")["Market Value ($billion)"]\
                      .sum().reset_index(name="Total MV (B USD)")  # [DA7]
        st.markdown("**Total Market Value by Continent**")
        st.table(grp)

# ─── 7) Key Insights & DA8 ────────────────────────────────────────────
with tabs[6]:
    st.subheader("Key Insights")
    # [DA8] iterate to find high‑norm companies
    high_norm = [row["Company"] for _, row in filtered.iterrows() if row["MV_norm"] > 0.9]
    if high_norm:
        st.write("Companies with MV_norm > 0.9:", ", ".join(high_norm))
    else:
        st.write("No company has MV_norm > 0.9.")

    # [DA5] filter two conditions: small MV & high profit margin
    niche = filtered[
        (filtered["Market Value ($billion)"] < 50) &
        (filtered["Profit Margin (%)"] > 15)
    ]  # [DA5]
    st.write("Small‑cap, high‑margin companies:")
    st.dataframe(niche[["Company","Market Value ($billion)","Profit Margin (%)"]])

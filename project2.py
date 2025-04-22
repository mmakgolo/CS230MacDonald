import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Data Explorer – Top2000 Global Companies",
    layout="wide"
)

# ─── Data Loading ────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "Market Value ($billion)",
        "Sales ($billion)",
        "Profits ($billion)",
        "Latitude",
        "Longitude"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data("Top2000CompaniesGlobally.csv")

# ─── Sidebar Filters ─────────────────────────────────────────────────
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Country", ["All"] + sorted(df["Country"].dropna().unique()))
continent = st.sidebar.selectbox("Continent", ["All"] + sorted(df["Continent"].dropna().unique()))
n = st.sidebar.slider("Top N companies", 5, 50, 10)
bins = st.sidebar.slider("Histogram bins", 5, 50, 20)

def filter_df(data: pd.DataFrame) -> pd.DataFrame:
    if country != "All":
        data = data[data["Country"] == country]
    if continent != "All":
        data = data[data["Continent"] == continent]
    return data

filtered = filter_df(df)

# ─── Tabs ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Bar Chart",
    "Interactive Scatter",
    "Interactive OSM Map",
    "Top List",
    "Interactive Histogram",
    "Summary Stats"
])

# ─── 1) Bar Chart (Plotly) ───────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Top {n} Companies by Market Value")
    top_df = filtered.nlargest(n, "Market Value ($billion)")
    fig = px.bar(
        top_df,
        x="Company",
        y="Market Value ($billion)",
        text=top_df["Market Value ($billion)"].round(1),
        labels={"Market Value ($billion)": "Market Value (B USD)"},
        title=f"Top {n} by Market Value"
    )
    fig.update_traces(
        marker_color="indianred",
        hovertemplate="<b>%{x}</b><br>Value: %{y:.1f} B"
    )
    fig.update_layout(xaxis_tickangle=-45, margin={"t":40,"b":150})
    st.plotly_chart(fig, use_container_width=True)

# ─── 2) Interactive Scatter (Plotly) ────────────────────────────────
with tabs[1]:
    st.subheader("Sales vs Profits (Interactive)")
    scatter_df = filtered.dropna(subset=["Sales ($billion)", "Profits ($billion)", "Continent"])
    if scatter_df.empty:
        st.info("No data to plot.")
    else:
        fig = px.scatter(
            scatter_df,
            x="Sales ($billion)",
            y="Profits ($billion)",
            color="Continent",
            size="Market Value ($billion)",
            hover_name="Company",
            labels={
                "Sales ($billion)": "Sales (B USD)",
                "Profits ($billion)": "Profits (B USD)",
                "Market Value ($billion)": "Market Value (B USD)"
            },
            title="Sales vs Profits by Continent"
        )
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
        st.plotly_chart(fig, use_container_width=True)

# ─── 3) Interactive OSM Map (PyDeck) ─────────────────────────────────
with tabs[2]:
    st.subheader("Company Locations (Stamen Terrain)")
    map_df = filtered.dropna(subset=["Latitude", "Longitude"])
    if map_df.empty:
        st.info("No location data to display.")
    else:
        midpoint = (map_df["Latitude"].mean(), map_df["Longitude"].mean())
        tile_layer = pdk.Layer(
            "TileLayer",
            data=None,
            get_tile_data="http://tile.stamen.com/terrain/{z}/{x}/{y}.png",
            tile_size=256,
            opacity=1.0
        )
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["Longitude", "Latitude"],
            get_radius=4,
            radiusUnits="pixels",
            pickable=True,
            get_fill_color=[34, 139, 34, 180]
        )
        view_state = pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=2,
            pitch=0
        )
        deck = pdk.Deck(
            layers=[tile_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{Company}</b><br>Value: {Market Value ($billion)} B",
                "style": {"backgroundColor": "black", "color": "white"}
            }
        )
        st.pydeck_chart(deck, use_container_width=True)

# ─── 4) Top List ─────────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Top {n} List")
    st.dataframe(filtered.nlargest(n, "Market Value ($billion)").reset_index(drop=True))

# ─── 5) Interactive Histogram (Plotly) ──────────────────────────────
with tabs[4]:
    st.subheader("Profit‑Margin Distribution")
    hist_df = filtered.dropna(subset=["Sales ($billion)", "Profits ($billion)"]).copy()
    hist_df = hist_df[hist_df["Sales ($billion)"] > 0]
    hist_df["Profit Margin (%)"] = (hist_df["Profits ($billion)"] / hist_df["Sales ($billion)"]) * 100
    hist_df = hist_df[np.isfinite(hist_df["Profit Margin (%)"])]
    if hist_df.empty:
        st.info("No valid profit‑margin data.")
    else:
        fig = px.histogram(
            hist_df,
            x="Profit Margin (%)",
            nbins=bins,
            title="Profit Margin (%)",
            labels={"count": "Frequency"}
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

# ─── 6) Summary Stats ────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Summary Statistics")
    stats_df = filtered[[
        "Sales ($billion)",
        "Profits ($billion)",
        "Assets ($billion)",
        "Market Value ($billion)"
    ]]
    if stats_df.empty:
        st.info("No data to summarize.")
    else:
        desc = stats_df.describe().loc[["mean", "50%", "std", "min", "max"]]
        desc.index = ["Mean", "Median", "Std", "Min", "Max"]
        st.table(desc)


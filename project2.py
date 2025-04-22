import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Data Explorer – Top2000 Global Companies",
    layout="wide"
)

# ─── Data loading ────────────────────────────────────────────────────
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

# ─── Sidebar filters ─────────────────────────────────────────────────
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
    "Financial Scatter",
    "Interactive OSM Map",
    "Top List",
    "Interactive Histogram",
    "Summary Stats"
])

# ─── 1) Bar Chart ─────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Top {n} by Market Value")
    top_df = filtered.nlargest(n, "Market Value ($billion)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top_df["Company"], top_df["Market Value ($billion)"])
    ax.set_ylabel("Market Value ($ B)")
    ax.set_xticklabels(top_df["Company"], rotation=90)
    for bar in ax.patches:
        ax.annotate(
            f"{bar.get_height():.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8
        )
    st.pyplot(fig)

# ─── 2) Financial Scatter ─────────────────────────────────────────────
with tabs[1]:
    st.subheader("Sales vs Profits")
    scatter_df = filtered.dropna(subset=["Sales ($billion)", "Profits ($billion)", "Continent"])
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    for i, cont in enumerate(scatter_df["Continent"].unique()):
        sub = scatter_df[scatter_df["Continent"] == cont]
        ax.scatter(
            sub["Sales ($billion)"],
            sub["Profits ($billion)"],
            label=cont,
            alpha=0.7,
            edgecolors="w",
            s=80,
            color=cmap(i)
        )
    ax.set_xlabel("Sales ($ B)")
    ax.set_ylabel("Profits ($ B)")
    ax.legend(title="Continent")
    st.pyplot(fig)

# ─── 3) Interactive OSM Map ───────────────────────────────────────────
with tabs[2]:
    st.subheader("Company Locations (OpenStreetMap Tiles)")
    map_df = filtered.dropna(subset=["Latitude", "Longitude"])
    if map_df.empty:
        st.info("No location data to display.")
    else:
        midpoint = (map_df["Latitude"].mean(), map_df["Longitude"].mean())

        # 1) OSM base tiles
        tile_layer = pdk.Layer(
            "TileLayer",
            data=None,
            get_tile_data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
            tile_size=256,
            opacity=1.0
        )

        # 2) Company points
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["Longitude", "Latitude"],
            get_radius=200000,             # meters
            pickable=True,
            get_fill_color=[220, 20, 60, 180]
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
                "html": "<b>{Company}</b><br/>Value: {Market Value ($billion)} B",
                "style": {"backgroundColor": "black", "color": "white"}
            }
        )
        st.pydeck_chart(deck, use_container_width=True)

# ─── 4) Top List ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Top {n} List")
    st.dataframe(
        filtered.nlargest(n, "Market Value ($billion)").reset_index(drop=True)
    )

# ─── 5) Interactive Histogram ────────────────────────────────────────
with tabs[4]:
    st.subheader("Profit‑Margin Histogram (Plotly)")
    hist_df = filtered.dropna(subset=["Sales ($billion)", "Profits ($billion)"])
    hist_df = hist_df[hist_df["Sales ($billion)"] > 0].copy()
    hist_df["Profit Margin (%)"] = (
        hist_df["Profits ($billion)"] / hist_df["Sales ($billion)"]
    ) * 100
    hist_df = hist_df[np.isfinite(hist_df["Profit Margin (%)"])]
    if hist_df.empty:
        st.info("No valid profit‑margin data.")
    else:
        fig = px.histogram(
            hist_df,
            x="Profit Margin (%)",
            nbins=bins,
            title="Profit Margin Distribution",
            labels={"count": "Frequency"}
        )
        fig.update_layout(
            xaxis_title="Profit Margin (%)",
            yaxis_title="Frequency",
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── 6) Summary Stats ─────────────────────────────────────────────────
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

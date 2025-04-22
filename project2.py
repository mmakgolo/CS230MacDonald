import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Interactive Data Explorer – Top2000 Global Companies",
    layout="wide"
)

@st.cache_data
def load_data(path):
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

# Sidebar filters
st.sidebar.header("Filters")
country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(df["Country"].dropna().unique())
)
continent = st.sidebar.selectbox(
    "Continent",
    ["All"] + sorted(df["Continent"].dropna().unique())
)
n = st.sidebar.slider("Top N companies", 5, 50, 10)
bins = st.sidebar.slider("Histogram bins", 5, 50, 20)

def filter_df(data):
    if country != "All":
        data = data[data["Country"] == country]
    if continent != "All":
        data = data[data["Continent"] == continent]
    return data

filtered = filter_df(df)

tabs = st.tabs([
    "Bar Chart",
    "Financial Scatter",
    "Country Map",
    "Top List",
    "Profit‑Margin Histogram",
    "Summary Statistics"
])

# 1) Bar Chart
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
            ha="center",
            va="bottom",
            fontsize=8
        )
    st.pyplot(fig)

# 2) Financial Scatter
with tabs[1]:
    st.subheader("Sales vs Profits")
    scatter_df = filtered.dropna(
        subset=["Sales ($billion)", "Profits ($billion)", "Continent"]
    )
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

# 3) Country Map
with tabs[2]:
    st.subheader("Company Locations")
    map_df = filtered.dropna(subset=["Latitude", "Longitude"])
    if map_df.empty:
        st.write("No location data to display.")
    else:
        map_df = map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        st.map(map_df[["lat", "lon"]])

# 4) Top List
with tabs[3]:
    st.subheader(f"Top {n} List")
    st.dataframe(
        filtered.nlargest(n, "Market Value ($billion)")
        .reset_index(drop=True)
    )

# 5) Profit‑Margin Histogram (with finite filtering)
with tabs[4]:
    st.subheader("Profit‑Margin Histogram")
    hist_df = filtered.dropna(
        subset=["Sales ($billion)", "Profits ($billion)"]
    )
    # only keep positive sales so the margin is finite
    hist_df = hist_df[hist_df["Sales ($billion)"] > 0]

    # compute margin and then drop any non-finite values
    hist_df["Profit Margin (%)"] = (
        hist_df["Profits ($billion)"] / hist_df["Sales ($billion)"]
    ) * 100
    pm = hist_df["Profit Margin (%)"]
    pm = pm[np.isfinite(pm)]

    if pm.empty:
        st.write("No valid profit‑margin data.")
    else:
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.hist(pm, bins=bins, orientation="horizontal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Profit Margin (%)")
        st.pyplot(fig)

# 6) Summary Statistics
with tabs[5]:
    st.subheader("Summary Statistics")
    stats_df = filtered[[
        "Sales ($billion)",
        "Profits ($billion)",
        "Assets ($billion)",
        "Market Value ($billion)"
    ]]
    if stats_df.empty:
        st.write("No data to summarize.")
    else:
        desc = stats_df.describe().loc[
            ["mean", "50%", "std", "min", "max"]
        ]
        desc.index = ["Mean", "Median", "Std", "Min", "Max"]
        st.table(desc)

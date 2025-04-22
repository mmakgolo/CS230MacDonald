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

# Page Config
st.set_page_config(
    page_title='Top2000 Global Companies Explorer',
    layout='wide'
)

# Data Loading & DA1
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
    
    # [DA1] convert key columns to numeric
    for col in ['Market Value ($billion)', 'Sales ($billion)', 'Profits ($billion)', 'Assets ($billion)', 'Latitude', 'Longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else df.get(col)

    # [DA1] split HQ location into City/State
    if 'Headquarters Location' in df.columns:
        df['City'] = df['Headquarters Location'].apply(lambda s: s.split(',')[0].strip())
        df['State'] = df['Headquarters Location'].apply(lambda s: s.split(',')[1].strip() if ',' in s else None)

    # [DA7] drop unused column
    df = df.drop(columns=['Ticker'], errors='ignore')

    # [DA9] add normalized Market Value and Profit Margin
    mv = df['Market Value ($billion)']
    df['MV_norm'] = (mv - mv.min()) / (mv.max() - mv.min())  # min-max normalization
    df['Profit Margin (%)'] = (df['Profits ($billion)'] / df['Sales ($billion)']) * 100
    df['Profit Margin (%)'].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

# load data
df = load_data('Top2000CompaniesGlobally.csv')

# Sidebar Filters & DA5
st.sidebar.header('Filters')
country = st.sidebar.selectbox('Country', ['All'] + sorted(df['Country'].dropna().unique()))
continent = st.sidebar.selectbox('Continent', ['All'] + sorted(df['Continent'].dropna().unique()))
max_mv = float(df['Market Value ($billion)'].max())
# two-value slider for profit margin range (min, max)
pm_min, pm_max = st.sidebar.slider(
    'Profit Margin (%) Range',
    0.0, 100.0,
    (0.0, 100.0),
    step=0.1
)
n = st.sidebar.slider('Top N companies', 5, 100, 10)
bins = st.sidebar.slider('Histogram bins', 5, 100, 20)

# market value threshold slider
min_mv = st.sidebar.slider('Min Market Value (B USD)', 0.0, max_mv, 0.0)

# DA5: combine filters
def filter_df(data: pd.DataFrame) -> pd.DataFrame:
    conds = []
    if country != 'All': conds.append(data['Country'] == country)
    if continent != 'All': conds.append(data['Continent'] == continent)
    conds.append(data['Market Value ($billion)'] >= min_mv)
    # filter within PM range
    conds.append(data['Profit Margin (%)'].between(pm_min, pm_max))
    if conds:
        mask = np.logical_and.reduce(conds)
        data = data[mask]
    return data

filtered = filter_df(df)

# Tabs
tabs = st.tabs([
    'Market Value Rankings', 'Sales & Profits', 'Global Map',
    'Leaderboard', 'Profit Margin Dist.', 'Stats & Pivot', 'Key Insights'
])

# 1) Market Value Rankings [DA3, DA2]
with tabs[0]:
    st.subheader(f'Top {n} by Market Value')
    top_df = filtered.nlargest(n, 'Market Value ($billion)')  # [DA3]
    fig = px.bar(
        top_df,
        x='Company',
        y='Market Value ($billion)',
        text=top_df['Market Value ($billion)'].round(1),
        title='Market Value Rankings'
    )
    fig.update_layout(xaxis_tickangle=-45, margin={'b':150})
    st.plotly_chart(fig, use_container_width=True)

# 2) Sales & Profits Analysis
with tabs[1]:
    st.subheader('Sales vs. Profits by Continent')
    scatter = filtered.dropna(subset=['Sales ($billion)', 'Profits ($billion)', 'Continent'])
    if scatter.empty:
        st.info('No data.')
    else:
        fig = px.scatter(
            scatter,
            x='Sales ($billion)',
            y='Profits ($billion)',
            color='Continent',
            size='Market Value ($billion)',
            hover_name='Company',
            title='Sales & Profits Scatter'
        )
        st.plotly_chart(fig, use_container_width=True)

# 3) Global Map [MAP]
with tabs[2]:
    st.subheader('Global Company Locations')
    map_df = filtered.dropna(subset=['Latitude', 'Longitude'])
    if map_df.empty:
        st.info('No locations.')
    else:
        mid_lat = map_df['Latitude'].mean()
        mid_lon = map_df['Longitude'].mean()
        deck = pdk.Deck(
            layers=[
                pdk.Layer('TileLayer', data=None, get_tile_data='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'),
                pdk.Layer('ScatterplotLayer', data=map_df, get_position=['Longitude','Latitude'], get_fill_color=[34,139,34], get_radius=3, pickable=True)
            ],
            initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=2)
        )
        st.pydeck_chart(deck)

# 4) Leaderboard
with tabs[3]:
    st.subheader('Company Leaderboard')
    lb = filtered.sort_values('Market Value ($billion)', ascending=False)
    st.dataframe(lb.head(n))

# 5) Profit Margin Distribution [DA9]
with tabs[4]:
    st.subheader('Profit Margin Histogram')
    pm = filtered.dropna(subset=['Profit Margin (%)']).copy()
    if pm.empty:
        st.info('No margins.')
    else:
        fig = px.histogram(
            pm,
            x='Profit Margin (%)',
            nbins=bins,
            title='Profit Margin (%) Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

# 6) Stats & Pivot Tables [DA6, DA7]
with tabs[5]:
    st.subheader('Statistical Summary')
    stats = filtered[['Sales ($billion)','Profits ($billion)','Assets ($billion)','Market Value ($billion)','Profit Margin (%)']]
    if stats.empty:
        st.info('No stats.')
    else:
        desc = stats.describe().loc[['mean','50%','std','min','max']]
        desc.index = ['Mean','Median','Std','Min','Max']
        st.table(desc)
        pivot = pd.pivot_table(filtered, index='Continent', values='Profit Margin (%)', aggfunc='mean').round(2)  # [DA6]
        st.markdown('**Avg Profit Margin by Continent**')
        st.table(pivot)
        grp = filtered.groupby('Continent')['Market Value ($billion)'].sum().reset_index(name='Total MV (B USD)')  # [DA7]
        st.markdown('**Total Market Value by Continent**')
        st.table(grp)

# 7) Key Insights & DA8
with tabs[6]:
    st.subheader('Key Insights')
    count_df = filtered['Continent'].value_counts().rename_axis('Continent').reset_index(name='Company Count')
    st.markdown('**Companies by Continent**')
    st.table(count_df)
    sum_df = filtered.groupby('Continent')['Market Value ($billion)'].sum().round(2).reset_index(name='Total Market Value (B USD)')
    st.markdown('**Total Market Value by Continent**')
    st.table(sum_df)
    high_norm = [row['Company'] for _, row in filtered.iterrows() if row['MV_norm'] > 0.9]  # [DA8]
    if high_norm:
        st.write('Companies with MV_norm > 0.9:', ', '.join(high_norm))
    else:
        st.write('No company has MV_norm > 0.9.')

"""
Name: Olorato MacDonald Makgolo    CS230: Section 6    Data: Top 2000 Global Companies
URL: https://cs230macdonald-ccruy8dswatyhy5kev.streamlit.app/
Description: An interactive Streamlit app showcasing Market Value Rankings, Sales vs Profits,
Global Locations map, Leaderboard table, Profit Margin distribution, Statistical Overview,
and Key Insights—plus DA1–DA9 pandas demonstrations embedded in code comments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

# Page Config
st.set_page_config(page_title='Top2000 Explorer', layout='wide')

# Load Data ([DA1])
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # convert numeric columns
    for col in ['Market Value ($billion)', 'Sales ($billion)', 'Profits ($billion)', 'Assets ($billion)', 'Latitude', 'Longitude']:
        df[col] = pd.to_numeric(df.get(col), errors='coerce')  # [DA1]
    # split HQ
    if 'Headquarters Location' in df.columns:
        df['City']  = df['Headquarters Location'].apply(lambda s: s.split(',')[0].strip())  # [DA1]
        df['State'] = df['Headquarters Location'].apply(lambda s: s.split(',')[1].strip() if ',' in s else '')
    # drop unused
    df.drop(columns=['Ticker'], errors='ignore', inplace=True)  # [DA7]
    # normalized MV
    mv = df['Market Value ($billion)']
    df['MV_norm'] = (mv - mv.min()) / (mv.max() - mv.min())  # [DA9]
    # profit margin
    df['Profit Margin (%)'] = (df['Profits ($billion)'] / df['Sales ($billion)']) * 100
    df['Profit Margin (%)'].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# Load data
df = load_data('Top2000CompaniesGlobally.csv')

# Sidebar Filters ([DA4],[DA5])
st.sidebar.header('Filters')
country = st.sidebar.selectbox('Country', ['All'] + sorted(df['Country'].dropna().unique()))
continent = st.sidebar.selectbox('Continent', ['All'] + sorted(df['Continent'].dropna().unique()))
n = st.sidebar.slider('Top N companies', 5, 50, 10)
bins = st.sidebar.slider('Histogram bins', 5, 50, 20)

# Filter function
def filter_df(data):
    if country != 'All':
        data = data[data['Country'] == country]  # [DA4]
    if continent != 'All':
        data = data[data['Continent'] == continent]  # [DA5]
    return data

filtered = filter_df(df)

# Tabs
tabs = st.tabs([
    'Market Value Rankings', 'Sales & Profits', 'Global Locations',
    'Company Leaderboard', 'Profit Margin Distribution',
    'Statistical Overview', 'Key Insights'
])

# 1) Market Value Rankings
with tabs[0]:
    st.subheader(f'Top {n} Companies by Market Value')
    top_df = filtered.nlargest(n, 'Market Value ($billion)')  # [DA3]
    fig = px.bar(
        top_df, x='Company', y='Market Value ($billion)', text=top_df['Market Value ($billion)'].round(1),
        labels={'Market Value ($billion)': 'Market Value (B USD)'},
        title='Market Value Rankings'
    )
    fig.update_layout(xaxis_tickangle=-45, margin={'t':40,'b':150})
    st.plotly_chart(fig, use_container_width=True)

# 2) Sales & Profits
with tabs[1]:
    st.subheader('Sales vs. Profits by Continent')
    scatter = filtered.dropna(subset=['Sales ($billion)', 'Profits ($billion)', 'Continent'])
    if scatter.empty:
        st.info('No data to plot.')
    else:
        fig = px.scatter(
            scatter, x='Sales ($billion)', y='Profits ($billion)',
            color='Continent', size='Market Value ($billion)',
            hover_name='Company',
            labels={'Sales ($billion)': 'Sales (B USD)', 'Profits ($billion)': 'Profits (B USD)'},
            title='Sales & Profits Analysis'
        )
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
        st.plotly_chart(fig, use_container_width=True)

# 3) Global Locations Map
with tabs[2]:
    st.subheader('Global Company Locations')
    map_df = filtered.dropna(subset=['Latitude', 'Longitude'])
    if map_df.empty:
        st.info('No location data.')
    else:
        midpoint = (map_df['Latitude'].mean(), map_df['Longitude'].mean())
        tile_layer = pdk.Layer(
            'TileLayer', data=None,
            get_tile_data='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
            tile_size=256, opacity=1.0
        )
        scatter_layer = pdk.Layer(
            'ScatterplotLayer', data=map_df,
            get_position=['Longitude', 'Latitude'], get_radius=50000, radius_units='meters',
            pickable=True, get_fill_color=[34, 139, 34, 180]
        )
        view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=2, pitch=0)
        deck = pdk.Deck(
            layers=[tile_layer, scatter_layer], initial_view_state=view_state,
            tooltip={'html': '<b>{Company}</b><br>MV: {Market Value ($billion)} B USD',
                     'style': {'backgroundColor': 'black', 'color': 'white'}}
        )
        st.pydeck_chart(deck, use_container_width=True)

# 4) Company Leaderboard
with tabs[3]:
    st.subheader(f'Top {n} Companies List')
    leaderboard = filtered.nlargest(n, 'Market Value ($billion)')  # [DA3]
    st.dataframe(leaderboard.reset_index(drop=True))

# 5) Profit Margin Distribution
with tabs[4]:
    st.subheader('Profit Margin Distribution')
    hist_df = filtered.dropna(subset=['Sales ($billion)', 'Profits ($billion)']).copy()
    hist_df = hist_df[hist_df['Sales ($billion)'] > 0]
    hist_df['Profit Margin (%)'] = (hist_df['Profits ($billion)'] / hist_df['Sales ($billion)']) * 100  # [DA9]
    hist_df = hist_df[np.isfinite(hist_df['Profit Margin (%)'])]
    if hist_df.empty:
        st.info('No valid profit‑margin data.')
    else:
        fig = px.histogram(
            hist_df, x='Profit Margin (%)', nbins=bins,
            title='Profit Margin Histogram', labels={'count':'Frequency'}
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

# 6) Statistical Overview
with tabs[5]:
    st.subheader('Statistical Overview')
    stats_df = filtered[[
        'Sales ($billion)', 'Profits ($billion)', 'Assets ($billion)',
        'Market Value ($billion)', 'Profit Margin (%)'
    ]]
    if stats_df.empty:
        st.info('No data to summarize.')
    else:
        desc = stats_df.describe().loc[['mean','50%','std','min','max']]
        desc.index = ['Mean','Median','Std','Min','Max']
        st.table(desc)

# 7) Key Insights
with tabs[6]:
    st.subheader('Key Insights by Continent')
    count_df = filtered['Continent'].value_counts().rename_axis('Continent').reset_index(name='Company Count')  # [DA8]
    st.table(count_df)
    sum_df = filtered.groupby('Continent')['Market Value ($billion)'].sum().round(2).reset_index(name='Total Market Value (B USD)')  # [DA7]
    st.table(sum_df)

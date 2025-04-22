"""
Name: Olorato MacDonald Makgolo    CS230: Section 6    Data: Top 2000 Global Companies
URL: https://cs230macdonald-ccruy8dswatyhy5kev.streamlit.app/
Description: An interactive Streamlit app showcasing Matplotlib, Seaborn, Plotly,
PyDeck, and Folium visualizations, plus DA1–DA9 pandas demonstrations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # [CHART1], [CHART2]
import seaborn as sns            # [SEA1], [SEA2]
import pydeck as pdk             # [MAP]
import folium                    # [FOLIUM1], [FOLIUM2]
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px

# Page Config
st.set_page_config(page_title='Top2000 Explorer', layout='wide')

# Load Data ([DA1])
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # convert columns to numeric
    for c in ['Market Value ($billion)','Sales ($billion)','Profits ($billion)','Assets ($billion)','Latitude','Longitude']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    # split location
    if 'Headquarters Location' in df.columns:
        df['City'] = df['Headquarters Location'].apply(lambda s: s.split(',')[0].strip())
        df['State'] = df['Headquarters Location'].apply(lambda s: s.split(',')[1].strip() if ',' in s else '')
    # drop unused
    df.drop(columns=['Ticker'], errors='ignore', inplace=True)
    # normalized MV
    df['MV_norm'] = (
        df['Market Value ($billion)'] - df['Market Value ($billion)'].min()
    ) / (
        df['Market Value ($billion)'].max() - df['Market Value ($billion)'].min()
    )
    # profit margin
    df['Profit Margin (%)'] = (df['Profits ($billion)'] / df['Sales ($billion)']) * 100
    df['Profit Margin (%)'].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

df = load_data('Top2000CompaniesGlobally.csv')

# Sidebar Filters ([DA4],[DA5])
st.sidebar.header('Filters')
country = st.sidebar.selectbox('Country', ['All'] + sorted(df['Country'].dropna().unique()))
continent = st.sidebar.selectbox('Continent', ['All'] + sorted(df['Continent'].dropna().unique()))
min_mv, max_mv = st.sidebar.slider(
    'Market Value Range (B USD)',
    float(df['Market Value ($billion)'].min()), float(df['Market Value ($billion)'].max()),
    (float(df['Market Value ($billion)'].min()), float(df['Market Value ($billion)'].max()))
)
pm_min, pm_max = st.sidebar.slider('Profit Margin Range (%)', 0.0, 100.0, (0.0, 100.0), step=0.5)
n = st.sidebar.slider('Top N Companies', 5, 100, 10)
bins = st.sidebar.slider('Histogram Bins', 5, 100, 20)

# Apply filters
def apply_filters(df):
    conds = []
    if country != 'All': conds.append(df['Country'] == country)
    if continent != 'All': conds.append(df['Continent'] == continent)
    conds.append(df['Market Value ($billion)'].between(min_mv, max_mv))
    conds.append(df['Profit Margin (%)'].between(pm_min, pm_max))
    if conds:
        return df[np.logical_and.reduce(conds)]
    return df

df_f = apply_filters(df)

# Tabs
tabs = st.tabs([
    'Matplotlib','Seaborn','Plotly','PyDeck Map','Folium Maps',
    'Stats & Pivot','Key Insights','DA Demo'
])

# Matplotlib Charts
with tabs[0]:
    st.subheader('Histogram of Market Value')
    plt.figure()
    df_f['Market Value ($billion)'].dropna().hist(bins=n)
    plt.title('Market Value Distribution')
    plt.xlabel('Market Value (B USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)  # [CHART1]

    st.subheader('Boxplot of Profit Margin (%)')
    plt.figure()
    plt.boxplot(df_f['Profit Margin (%)'].dropna())
    plt.title('Profit Margin Boxplot')
    plt.ylabel('Profit Margin (%)')
    st.pyplot(plt)  # [CHART2]

# Seaborn Charts
with tabs[1]:
    st.subheader('Avg Market Value by Continent')
    plt.figure(figsize=(8,4))
    sns.barplot(x='Continent', y='Market Value ($billion)', data=df_f)
    plt.xticks(rotation=45)
    st.pyplot(plt)  # [SEA1]

    st.subheader('MV_norm vs Profit Margin (%)')
    plt.figure(figsize=(8,4))
    sns.scatterplot(x='MV_norm', y='Profit Margin (%)', hue='Continent', data=df_f, alpha=0.7)
    st.pyplot(plt)  # [SEA2]

# Plotly
with tabs[2]:
    st.subheader('Sales vs Profits Scatter')
    fig = px.scatter(
        df_f, x='Sales ($billion)', y='Profits ($billion)',
        color='Continent', size='Market Value ($billion)',
        hover_name='Company', title='Sales vs Profits'
    )
    st.plotly_chart(fig, use_container_width=True)

# PyDeck Map [MAP]
with tabs[3]:
    st.subheader('Global Company Locations (PyDeck)')
    mdf = df_f.dropna(subset=['Latitude','Longitude'])
    view = pdk.ViewState(
        latitude=mdf['Latitude'].mean(), longitude=mdf['Longitude'].mean(), zoom=1
    )
    layer = pdk.Layer(
        'ScatterplotLayer', mdf,
        get_position=['Longitude','Latitude'], get_radius=50000,
        radius_units='meters', get_fill_color=[200,30,0,160], pickable=True
    )
    deck = pdk.Deck(layers=[layer], initial_view_state=view,
                    tooltip={'html':'<b>{Company}</b><br>MV: {Market Value ($billion)} B USD'})
    st.pydeck_chart(deck)

# Folium Maps
with tabs[4]:
    st.subheader('Folium: Point Map')
    m1 = folium.Map(location=[mdf['Latitude'].mean(), mdf['Longitude'].mean()], zoom_start=2)
    for _,r in mdf.iterrows():
        folium.CircleMarker(
            location=[r['Latitude'], r['Longitude']], radius=4,
            popup=r['Company'], color='blue', fill=True
        ).add_to(m1)
    st_folium(m1, width=700)  # [FOLIUM1]

    st.subheader('Folium: Heatmap')
    m2 = folium.Map(location=[mdf['Latitude'].mean(), mdf['Longitude'].mean()], zoom_start=2)
    heat = HeatMap(mdf[['Latitude','Longitude']].values.tolist(), radius=15)
    m2.add_child(heat)
    st_folium(m2, width=700)  # [FOLIUM2]

# Stats & Pivot
with tabs[5]:
    st.subheader('Statistical Summary')
    stats = df_f[['Sales ($billion)','Profits ($billion)','Assets ($billion)','Market Value ($billion)','Profit Margin (%)']]
    desc = stats.describe().loc[['mean','50%','std','min','max']]
    desc.index = ['Mean','Median','Std','Min','Max']
    st.table(desc)

    st.subheader('Avg Profit Margin by Continent')
    pivot = df_f.pivot_table(index='Continent', values='Profit Margin (%)', aggfunc='mean').round(2)
    st.table(pivot)

    st.subheader('Total Market Value by Continent')
    grp = df_f.groupby('Continent')['Market Value ($billion)'].sum().reset_index(name='Total MV (B USD)').round(2)
    st.table(grp)

# Key Insights
with tabs[6]:
    st.subheader('Key Insights')
    st.markdown('**Company Count by Continent**')
    cnt = df_f['Continent'].value_counts().rename_axis('Continent').reset_index(name='Count')
    st.table(cnt)

    st.subheader('High-Norm Companies (MV_norm > 0.9)')
    high_norm = [r['Company'] for _,r in df_f.iterrows() if r['MV_norm']>0.9]
    st.write(', '.join(high_norm) if high_norm else 'None')

# DA1–DA9 Demo
with tabs[7]:
    st.header('DA1–DA9 Demonstrations')
    demo = pd.DataFrame({
        'Operation': ['Clean','Sort','Top N','Filter','Filter Range','Pivot','Group','Iterrows','New Col'],
        'Tag': ['DA1','DA2','DA3','DA4','DA5','DA6','DA7','DA8','DA9']
    })
    st.table(demo)

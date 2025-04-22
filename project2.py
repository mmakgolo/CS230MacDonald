"""
Name: Olorato MacDonald Makgolo    CS230: Section 6    Data: Top 2000 Global Companies
URL: https://cs230macdonald-ccruy8dswatyhy5kev.streamlit.app/
Description: An interactive Streamlit app showcasing Matplotlib, Seaborn, Plotly,
PyDeck, and Folium visualizations, plus DA1–DA9 pandas demonstrations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for [CHART1], [CHART2]
import seaborn as sns            # for [SEA1], [SEA2]
import pydeck as pdk             # for [MAP]
import folium                    # for [FOLIUM1], [FOLIUM2]
from streamlit_folium import st_folium
import plotly.express as px

# Page Config
st.set_page_config(page_title='Top2000 Explorer', layout='wide')

# Load Data ([DA1])
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # [DA1]: convert key columns to numeric
    for c in ['Market Value ($billion)','Sales ($billion)','Profits ($billion)','Assets ($billion)','Latitude','Longitude']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    # [DA1]: split HQ location
    if 'Headquarters Location' in df.columns:
        df['City'] = df['Headquarters Location'].apply(lambda s: s.split(',')[0].strip())
        df['State'] = df['Headquarters Location'].apply(lambda s: s.split(',')[1].strip() if ',' in s else '')
    # [DA7]: drop unused column
    df.drop(columns=['Ticker'], errors='ignore', inplace=True)
    # [DA9]: compute normalized Market Value
    df['MV_norm'] = (
        df['Market Value ($billion)'] - df['Market Value ($billion)'].min()
    ) / (
        df['Market Value ($billion)'].max() - df['Market Value ($billion)'].min()
    )
    # [DA9]: compute Profit Margin and remove infinities
    df['Profit Margin (%)'] = (df['Profits ($billion)'] / df['Sales ($billion)']) * 100
    df['Profit Margin (%)'].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# Load
df = load_data('Top2000CompaniesGlobally.csv')

# Sidebar Filters (DA4, DA5)
st.sidebar.header('Filters')
country = st.sidebar.selectbox('Country',['All']+sorted(df['Country'].dropna().unique()))
continent = st.sidebar.selectbox('Continent',['All']+sorted(df['Continent'].dropna().unique()))
min_mv, max_mv = st.sidebar.slider(
    'Market Value Range (B USD)',
    float(df['Market Value ($billion)'].min()),
    float(df['Market Value ($billion)'].max()),
    (float(df['Market Value ($billion)'].min()), float(df['Market Value ($billion)'].max()))
)
pm_min, pm_max = st.sidebar.slider(
    'Profit Margin Range (%)', 0.0, 100.0, (0.0, 100.0), step=0.5
)

# Apply filters
conds = []
if country != 'All':
    conds.append(df['Country'] == country)
if continent != 'All':
    conds.append(df['Continent'] == continent)
conds.append(df['Market Value ($billion)'].between(min_mv, max_mv))
conds.append(df['Profit Margin (%)'].between(pm_min, pm_max))
if conds:
    mask = np.logical_and.reduce(conds)
    df_f = df[mask]
else:
    df_f = df.copy()

# Tabs
tabs = st.tabs([
    'Matplotlib','Seaborn','Plotly','PyDeck Map','Folium Maps','DA Demo'
])

# [CHART1]: Matplotlib Histogram of Market Value
with tabs[0]:
    plt.figure()
    df_f['Market Value ($billion)'].dropna().hist(bins=20)
    plt.title('Histogram of Market Value')
    plt.xlabel('Market Value (B USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # [CHART2]: Matplotlib Boxplot of Profit Margin
    plt.figure()
    plt.boxplot(df_f['Profit Margin (%)'].dropna())
    plt.title('Boxplot of Profit Margin (%)')
    plt.ylabel('Profit Margin (%)')
    st.pyplot(plt)

# [SEA1] & [SEA2]: Seaborn Charts
with tabs[1]:
    plt.figure()
    sns.barplot(x='Continent', y='Market Value ($billion)', data=df_f)
    plt.title('Avg Market Value by Continent')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    plt.figure()
    sns.scatterplot(
        x='MV_norm', y='Profit Margin (%)',
        hue='Continent', data=df_f, alpha=0.7
    )
    plt.title('Normalized MV vs Profit Margin')
    st.pyplot(plt)

# Plotly Scatter
with tabs[2]:
    fig = px.scatter(
        df_f, x='Sales ($billion)', y='Profits ($billion)',
        color='Continent', size='Market Value ($billion)',
        hover_name='Company', title='Sales vs Profits'
    )
    st.plotly_chart(fig, use_container_width=True)

# [MAP]: PyDeck Detailed Map with Tooltips
with tabs[3]:
    st.subheader('PyDeck: Company Scatter with Tooltips')
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=df_f['Latitude'].mean(),
            longitude=df_f['Longitude'].mean(),
            zoom=1
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_f.dropna(subset=['Latitude','Longitude']),
                get_position=['Longitude','Latitude'],
                get_fill_color=[200,30,0,160],
                get_radius=50000,
                radius_units='meters',
                pickable=True
            )
        ],
        tooltip={
            'html': '<b>Company:</b> {Company}<br/><b>MV (B USD):</b> {Market Value ($billion)}',
            'style': {'color': 'white'}
        }
    )
    st.pydeck_chart(deck)

# [FOLIUM1]: Folium Point Markers
with tabs[4]:
    m1 = folium.Map(
        location=[df_f['Latitude'].mean(), df_f['Longitude'].mean()],
        zoom_start=2
    )
    for _, r in df_f.dropna(subset=['Latitude','Longitude']).iterrows():
        folium.CircleMarker(
            location=[r['Latitude'], r['Longitude']],
            radius=4,
            popup=r['Company'],
            color='blue',
            fill=True
        ).add_to(m1)
    st_folium(m1, width=700)

# [FOLIUM2]: Folium Heatmap
with tabs[4]:
    from folium.plugins import HeatMap
    m2 = folium.Map(
        location=[df_f['Latitude'].mean(), df_f['Longitude'].mean()],
        zoom_start=2
    )
    heat_data = df_f[['Latitude','Longitude']].dropna().values.tolist()
    HeatMap(heat_data, radius=15).add_to(m2)
    st_folium(m2, width=700)

# DA Demo Tab [DA1–DA9]
with tabs[5]:
    st.header('DA1–DA9 Demonstrations')
    demo = pd.DataFrame({
        'Operation': [
            'Clean numeric', 'Sort', 'Top N', 'Filter', 'Filter range',
            'Pivot', 'Group', 'Iterrows', 'New column'
        ],
        'Tag': [
            'DA1','DA2','DA3','DA4','DA5','DA6','DA7','DA8','DA9'
        ]
    })
    st.table(demo)

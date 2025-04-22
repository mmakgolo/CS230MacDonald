"""
Name: Olorato MacDonald Makgolo    CS230: Section 6    Data: Top 2000 Global Companies
URL: https://cs230macdonald-ccruy8dswatyhy5kev.streamlit.app/
Description: An interactive explorer showcasing multiple chart types with Matplotlib, Seaborn, Plotly,
PyDeck, and Folium maps, plus DA1–DA9 pandas demos.
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
    # DA1: clean numeric columns
    for c in ['Market Value ($billion)','Sales ($billion)','Profits ($billion)','Assets ($billion)','Latitude','Longitude']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    # DA1: split location
    if 'Headquarters Location' in df.columns:
        df['City']=df['Headquarters Location'].apply(lambda s: s.split(',')[0])
        df['State']=df['Headquarters Location'].apply(lambda s: s.split(',')[1] if ',' in s else '')
    # DA7: drop unused
    df.drop(columns=['Ticker'], errors='ignore', inplace=True)
    # DA9: create new cols
    df['MV_norm']=(df['Market Value ($billion)']-df['Market Value ($billion)'].min())/
                  (df['Market Value ($billion)'].max()-df['Market Value ($billion)'].min())
    df['Profit Margin (%)']=(df['Profits ($billion)']/df['Sales ($billion)'])*100
    df['Profit Margin (%)'].replace([np.inf,-np.inf],np.nan,inplace=True)
    return df

df=load_data('Top2000CompaniesGlobally.csv')

# Sidebar Filters (DA4, DA5)
st.sidebar.header('Filters')
country=st.sidebar.selectbox('Country',['All']+sorted(df['Country'].dropna().unique()))
continent=st.sidebar.selectbox('Continent',['All']+sorted(df['Continent'].dropna().unique()))
min_mv, max_mv = st.sidebar.slider('Market Value Range (B USD)',
    float(df['Market Value ($billion)'].min()), float(df['Market Value ($billion)'].max()),
    (float(df['Market Value ($billion)'].min()),float(df['Market Value ($billion)'].max())))
pm_min, pm_max = st.sidebar.slider('Profit Margin Range (%)',0.0,100.0,(0.0,100.0),step=0.5)

# Apply filters
conds=[]
if country!='All':conds.append(df['Country']==country)
if continent!='All':conds.append(df['Continent']==continent)
conds.append(df['Market Value ($billion)'].between(min_mv,max_mv))
conds.append(df['Profit Margin (%)'].between(pm_min,pm_max))
if conds:
    mask=np.logical_and.reduce(conds)
    df_f=df[mask]
else:
    df_f=df.copy()

# Define layout
tabs=st.tabs(['Matplotlib','Seaborn','Plotly','PyDeck Map','Folium Maps','DA Demo'])

# [CHART1]: Matplotlib Histogram of Market Value
ax1=plt.figure();
with tabs[0]:
    plt.figure()
    df_f['Market Value ($billion)'].dropna().hist(bins=20)
    plt.title('Histogram of Market Value')
    plt.xlabel('Market Value (B USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)  # [CHART1]

# [CHART2]: Matplotlib Boxplot of Profit Margin
with tabs[0]:
    plt.figure()
    plt.boxplot(df_f['Profit Margin (%)'].dropna())
    plt.title('Boxplot of Profit Margin (%)')
    plt.ylabel('Profit Margin (%)')
    st.pyplot(plt)

# [SEA1]: Seaborn Barplot avg MV by Continent
with tabs[1]:
    plt.figure()
    sns.barplot(x='Continent',y='Market Value ($billion)',data=df_f)
    plt.title('Avg Market Value by Continent')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# [SEA2]: Seaborn Scatterplot MV_norm vs Profit Margin
with tabs[1]:
    plt.figure()
    sns.scatterplot(x='MV_norm',y='Profit Margin (%)',hue='Continent',data=df_f,alpha=0.7)
    plt.title('Normalized MV vs Profit Margin')
    st.pyplot(plt)

# Plotly: supplemental
with tabs[2]:
    fig=px.scatter(df_f,x='Sales ($billion)',y='Profits ($billion)',color='Continent',size='Market Value ($billion)',hover_name='Company',title='Sales vs Profits')
    st.plotly_chart(fig,use_container_width=True)

# [MAP]: PyDeck Detailed Map with tooltips\with tabs[3]:
    st.subheader('PyDeck Scatter with Tooltips')
    deck=pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=df_f['Latitude'].mean(),longitude=df_f['Longitude'].mean(),zoom=1),
        layers=[pdk.Layer('ScatterplotLayer',data=df_f.dropna(subset=['Latitude','Longitude']),get_position=['Longitude','Latitude'],get_radius=50000,radius_units='meters',get_fill_color='[200,30,0,160]',pickable=True)],
        tooltip={'text':'Company: {Company}\nMV: {Market Value ($billion)} B USD'}
    )
    st.pydeck_chart(deck)

# [FOLIUM1] Simple Folium Map
with tabs[4]:
    m1=folium.Map(location=[df_f['Latitude'].mean(),df_f['Longitude'].mean()],zoom_start=2)
    for _,r in df_f.dropna(subset=['Latitude','Longitude']).iterrows():
        folium.CircleMarker(location=[r['Latitude'],r['Longitude']],radius=4,popup=r['Company'],color='blue',fill=True).add_to(m1)
    st_folium(m1,width=700)

# [FOLIUM2] Heatmap Folium
with tabs[4]:
    from folium.plugins import HeatMap
    m2=folium.Map(location=[df_f['Latitude'].mean(),df_f['Longitude'].mean()],zoom_start=2)
    HeatMap(df_f[['Latitude','Longitude']].dropna().values.tolist(),radius=15).add_to(m2)
    st_folium(m2,width=700)

# DA Demo Tab
with tabs[5]:
    st.header('DA1–DA9 Demonstrations')
    da=pd.DataFrame({'Operation':['Clean','Sort','Top N','Filter1','Filter2','Pivot','Group','Iterrows','New Col'],
                     'Tag':['DA1','DA2','DA3','DA4','DA5','DA6','DA7','DA8','DA9']})
    st.table(da)

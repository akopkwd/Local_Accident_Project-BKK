import streamlit as st
import streamlit.components.v1 as stc

#from branca.tiles import OSM
from branca.colormap import LinearColormap
#from branca.layer import TileLayer, GeoJSON
from folium.plugins import MarkerCluster
#from folium import Map, LayerControl, ColorControl, FeatureGroup, MarkerCluster  # Import from folium directly

import folium
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from libpysal import weights
from esda.moran import Moran
from splot.esda import moran_scatterplot
import pydeck as pdk

# Importing Libraries

import pandas as pd
import geopandas as gpd
import numpy as np
from pysal.lib import weights
from splot.libpysal import plot_spatial_weights
from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot, plot_local_autocorrelation, lisa_cluster
import matplotlib.pyplot as plt
import folium
import matplotlib.colors as colors

import seaborn as sns

from streamlit_folium import folium_static


import warnings
warnings.filterwarnings("ignore")


#@st.cache_data
def plot_type_counts(dataframe):
    # นับจำนวนข้อมูลแต่ละประเภท
    type_counts = dataframe['type'].value_counts().head(5)

    # กำหนดขนาดของกราฟ
    plt.figure(figsize=(10, 8))

    # พล็อตกราฟแท่ง
    fig, ax = plt.subplots()
    type_counts.plot(kind='bar', ax=ax, color='#8B0000')
    ax.set_title('Type Counts', color = 'white')
    ax.set_xlabel('Type', color = 'white')
    ax.set_ylabel('Count', color = 'white')
    ax.set_xticklabels(type_counts.index, rotation=45)  # หมุนแกน x 45 องศาเพื่อให้ข้อความไม่ทับกัน
    ax.spines['top'].set_visible(False)  # ซ่อนกรอบด้านขวา
    ax.spines['right'].set_visible(False)  # ซ่อนกรอบด้านบน
    st.pyplot(fig)

#@st.cache_data
def create_map(dataframe):
    # สร้างแผนที่
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=10)

    # เพิ่ม circle marker บนแผนที่
    for index, row in dataframe.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,  # ขนาดของวงกลม
            color='blue',  # สีของวงกลม
            fill=True,
            fill_color='blue'  # สีในของวงกลม
        ).add_to(m)

    # แปลง Folium map เป็น HTML
    m = m._repr_html_()

    # แสดงแผนที่
    return m

#@st.cache_data
def process_acc_data(csv_path):
    """
    ฟังก์ชันสำหรับดำเนินการกับข้อมูลการเกิดอุบัติเหตุโดยการอ่านจากไฟล์ CSV
    
    Parameters:
    csv_path : str
        พาธไฟล์ของ CSV ที่มีข้อมูลการเกิดอุบัติเหตุ
    
    Returns:
    pd.DataFrame: DataFrame ที่ผ่านการประมวลผลแล้ว
    """
    # อ่านข้อมูลจากไฟล์ CSV
    acc_data = pd.read_csv(csv_path)
    
    # ลบคอลัมน์ที่ไม่ต้องการ
    acc_data.drop(['OBJECTID', 'pcode'], axis=1, inplace=True)
    
    # เปลี่ยนชื่อคอลัมน์
    acc_data = acc_data.rename(columns={'NUMPOINTS': 'acc_count'})
    
    # เรียงลำดับคอลัมน์ใหม่
    new_columns_order = ['dcode', 'dname', 'dname_e', 'acc_count', 'AREA', 'pname']
    acc_data = acc_data.reindex(columns=new_columns_order)
    
    return acc_data

#@st.cache_data
def process_data(filepath_acc_count):
    # อ่านข้อมูลจากไฟล์ CSV
    acc_data = pd.read_csv(filepath_acc_count)
    
    # ลบคอลัมน์ที่ไม่ต้องการ
    #acc_data.drop(['OBJECTID', 'pcode'], axis=1, inplace=True)
    
    # เรียงลำดับคอลัมน์ใหม่ตามลำดับที่กำหนด
    new_columns = ['dcode', 'dname', 'dname_e', 'acc_count', 'AREA', 'pname']
    acc_data = acc_data.reindex(columns=new_columns)
    
    return acc_data

#@st.cache_data
def display_map(accgdf):
    """
    ฟังก์ชันสำหรับแสดงแผนที่ Folium จากข้อมูลเชิงพื้นที่
    
    Parameters:
    accgdf : geopandas.GeoDataFrame
        ข้อมูลเชิงพื้นที่ที่ต้องการแสดงบนแผนที่ Folium
    
    Returns:
    folium.Map: แผนที่ Folium
    """
    # สร้างแผนที่ Folium
    m = folium.Map(location=[13.736717, 100.523186], zoom_start=10)

    # เพิ่มข้อมูลรูปร่างเขตแยกลงในแผนที่ Folium
    folium.GeoJson(accgdf).add_to(m)

    # แสดงแผนที่
    return m
#@st.cache_data
def count_acc_bkk_data(filepath_acc_gdf):
    """
    ฟังก์ชันเพื่อแสดงข้อมูลเขตแยกของกรุงเทพที่ผ่านการเรียงลำดับคอลัมน์แล้ว
    
    Parameters:
    filepath (str): ที่อยู่ของไฟล์ Shapefile
    
    Returns:
    None
    """
    # โหลดข้อมูลเขตแยกของกรุงเทพจากไฟล์ Shapefile
    accgdf = gpd.read_file(filepath_acc_gdf)

    # สร้างแผนที่ Folium
    m = folium.Map(location=[13.736717, 100.523186], zoom_start=10)

    # เพิ่มข้อมูลรูปร่างเขตแยกลงในแผนที่ Folium
    folium.GeoJson(accgdf).add_to(m)

    # แสดงแผนที่
    display(m)

    # รีอินเด็กซ์คอลัมน์ตามลำดับใหม่
    new_col = ['dcode', 'dname', 'dname_e','acc_count' , 'AREA','pname', 'geometry']
    accgdf = accgdf.reindex(columns=new_col)

    # แสดงข้อมูลหัวตาราง
    print(accgdf.head())

#@st.cache_data
def plot_acc_bkk_map(filepath_acc_gdf): #แยกจากอันข้างบน
    """
    ฟังก์ชันเพื่อแสดงแผนที่ข้อมูลเขตแยกของกรุงเทพ
    
    Parameters:
    filepath_acc_gdf (str): ที่อยู่ของไฟล์ Shapefile
    
    Returns:
    None
    """
    # โหลดข้อมูลเขตแยกของกรุงเทพจากไฟล์ Shapefile
    accgdf = gpd.read_file(filepath_acc_gdf)

    # สร้างแผนที่ Folium
    m = folium.Map(location=[13.736717, 100.523186], zoom_start=10)

    # เพิ่มข้อมูลรูปร่างเขตแยกลงในแผนที่ Folium
    folium.GeoJson(accgdf).add_to(m)

    # แสดงแผนที่
    display(m)

#@st.cache_data
def reorder_acc_gdf_columns(accgdf):
    """
    ฟังก์ชันเพื่อเรียงลำดับคอลัมน์ของ GeoDataFrame
    
    Parameters:
    accgdf (geopandas.GeoDataFrame): ข้อมูลเขตแยกของกรุงเทพ
    
    Returns:
    geopandas.GeoDataFrame: GeoDataFrame ที่เรียงลำดับคอลัมน์แล้ว
    """
    # รีอินเด็กซ์คอลัมน์ตามลำดับใหม่
    new_col = ['dcode', 'dname', 'dname_e','acc_count' , 'AREA','pname', 'geometry']
    accgdf = accgdf.reindex(columns=new_col)

    # แสดงข้อมูลหัวตาราง
    print(accgdf.head())
    
    return accgdf

#@st.cache_data
def create_choropleth_map(_geo_df, column, cmap='Oranges', legend_name=None):
    """
    สร้างแผนที่ choropleth map โดยใช้ Folium
    
    Parameters:
    _geo_df (geopandas.GeoDataFrame): ข้อมูลเขตแยกที่มีค่าข้อมูลตามพื้นที่
    column (str): ชื่อคอลัมน์ใน GeoDataFrame ที่ต้องการพล็อต
    cmap (str): ชื่อ colormap ที่ต้องการใช้ (ค่าเริ่มต้น 'Oranges')
    legend_name (str): ชื่อสำหรับคำอธิบายแผนที่ (ค่าเริ่มต้น None)
    
    Returns:
    folium.Map: แผนที่ Folium
    """
    # สร้างแผนที่ Folium
    m = folium.Map(location=[13.736717, 100.523186], zoom_start=10)

    # เพิ่มข้อมูลรูปร่างเขตแยกลงในแผนที่ Folium
    folium.GeoJson(_geo_df,
                    name='choropleth',
                    data=_geo_df,
                    columns=['dcode', column],
                    key_on='feature.properties.dcode',
                    fill_color=cmap,
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=legend_name).add_to(m)

    # แสดงแผนที่
    return m

############# global

#@st.cache_data
def calculate_and_plot_morans_I(_geo_df, _w):
    """
    ฟังก์ชันสำหรับคำนวณ Moran's I statistic และพล็อต scatter plot
    
    Parameters:
    _geo_df : geopandas.GeoDataFrame
        ข้อมูลเชิงพื้นที่
    _w : libpysal.weights.W object
        วัตถุน้ำหนักพื้นที่
    
    Returns:
    None
    """
    # กำหนดตัวแปรตามที่ต้องการ
    y_acc_count = _geo_df['acc_count']
    
    # คำนวณ Moran's I statistic
    moran = Moran(y_acc_count, _w)
    
    # คืนค่า Moran's I statistic และ p-value
    return moran.I, moran.p_sim
############# local

#@st.cache_data
def plot_local(y_data, _moran_obj):
    """
    ฟังก์ชันเพื่อพล็อต Local Moran's I scatterplot
    
    Parameters:
    y_data (pandas.Series): ข้อมูลที่ใช้ในการคำนวณ Local Moran's I
    _moran_obj (esda.Moran_Local): อ็อบเจกต์ที่คำนวณจาก Moran_Local
    
    Returns:
    None
    """
    # กำหนดขนาดของกราฟ
    plt.figure(figsize=(10, 8))

    # Plotting Local Moran's I scatterplot
    fig, ax = moran_scatterplot(_moran_obj, p=0.05)

    # เพิ่มข้อความลงในกราฟ
    plt.text(5, 1, 'HH', fontsize=25)
    plt.text(5, -1.0, 'HL', fontsize=25)
    plt.text(-1.5, 1, 'LH', fontsize=25)
    plt.text(-1.5, -1, 'LL', fontsize=25)

    # กำหนดแกน y ให้แสดงทีละ 0.1
    plt.yticks([i/10 for i in range(-10, 12, 2)], fontsize=7.5, rotation=0, ha='right')

    # แสดงกราฟใน Streamlit
    st.pyplot(fig)

#@st.cache_data
def local_classification(_accgdf, _moran_obj):
    """
    ฟังก์ชันเพื่อเพิ่มคอลัมน์ Local Moran's I classification และ p-value เข้าสู่ GeoDataFrame
    
    Parameters:
    accgdf (geopandas.GeoDataFrame): ข้อมูล GeoDataFrame
    moran_obj (esda.Moran_Local): อ็อบเจกต์ที่คำนวณจาก Moran_Local
    
    Returns:
    geopandas.GeoDataFrame: GeoDataFrame ที่เพิ่มคอลัมน์แล้ว
    """
    # สร้างคอลัมน์กำหนดของความสำคัญทางสถิติ Local Moran's I classification
    _accgdf['acc_count_local_moran'] = _moran_obj.q

    # Dict เพื่อแมปรหัสการจัดหมวดหมู่ Local Moran's I classification
    local_moran_classification = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}

    # แมปรหัสการจัดหมวดหมู่ Local Moran's I classification
    _accgdf['acc_count_local_moran'] = _accgdf['acc_count_local_moran'].map(local_moran_classification)

    # คำนวณ p-value สำหรับแต่ละคู่ข้อมูลในบริเวณเคียง
    _accgdf['acc_count_local_moran_p_sim'] = _moran_obj.p_sim

    # ถ้า p-value > 0.05 แสดงว่าไม่มีความสำคัญทางสถิติ
    _accgdf['acc_count_local_moran'] = np.where(_accgdf['acc_count_local_moran_p_sim'] > 0.05, 'ns', _accgdf['acc_count_local_moran'])

    return _accgdf

#@st.cache_data
def get_dname_by_moran_class(_accgdf):
    """
    ฟังก์ชันเพื่อรับชื่อเขตในคอลัมน์ "dname" ของ GeoDataFrame ตามค่าในคอลัมน์ "acc_count_local_moran"
    
    Parameters:
    accgdf (geopandas.GeoDataFrame): GeoDataFrame ที่มีคอลัมน์ "dname" และ "acc_count_local_moran"
    
    Returns:
    dict: ค่าเขตตามคลาสของ Moran
    """
    # สร้าง dictionary เพื่อเก็บชื่อเขตตามค่า "acc_count_local_moran"
    dname_by_moran_class = {'HH': [], 'HL': [], 'LH': [], 'LL': []}
    
    # วนลูปผ่านแต่ละค่าของ "acc_count_local_moran"
    for moran_class in dname_by_moran_class.keys():
        # กรองข้อมูลเขตที่มีค่า "acc_count_local_moran" ตรงกับค่าที่กำหนด
        filtered_accgdf = _accgdf[_accgdf['acc_count_local_moran'] == moran_class]
        
        # เพิ่มชื่อเขตลงในรายการของค่า "acc_count_local_moran" นั้น
        dname_by_moran_class[moran_class].extend(filtered_accgdf['dname'].tolist())
    
    return dname_by_moran_class

##################################################################################################################################


st.set_page_config(
    page_title="Traffic_bkk",
    page_icon="🚦",
)

##################################################################################################################################

tab1, tab2, tab3 = st.tabs(["Information", "Event Map", "Local Autocorrelation for Accident in BKK"])

with tab1:
   st.header("2020 Accident Information", divider = "red")

   st.subheader("Top 5 types of accidents")
   df = pd.read_csv(r'https://github.com/akopkwd/Local_Accident_Project-BKK/blob/main/event2020_withd_2.csv',encoding='ISO-8859-1')
   df.drop(['OBJECTID', 'pcode'], axis=1, inplace=True)
   plot_type_counts(df)

   st.subheader("Type Description")
   info = pd.read_csv(r'https://github.com/akopkwd/Local_Accident_Project-BKK/blob/main/event_info.csv')
   st.dataframe(info)


##################################################################################################################################

with tab2:
    
   st.title("Event Data Map")
   # Load geographical data
   accgdf_2020 = gpd.read_file('E:/studyyy/year4/senior pj/test/web_test/count_2020_1.shp')
   accgdf_2020 = accgdf_2020.rename(columns={'NUMPOINTS': 'acc_count'})
   new_col = ['dcode', 'dname', 'dname_e','acc_count' , 'AREA','pname', 'geometry']
   accgdf_2020 = accgdf_2020.reindex(columns=new_col)

   # Fillna 
   accgdf_2020['acc_count'].fillna(0, inplace=True)

   # Create folium map
   m = folium.Map(location=[13.7563, 100.5018], zoom_start=10)

   # Add choropleth layer to the map
   folium.Choropleth(
       geo_data=accgdf_2020,
       data=accgdf_2020,
       columns=['dcode', 'acc_count'],
       key_on='feature.properties.dcode',
       fill_color='Reds',
       fill_opacity=0.7,
       line_opacity=0.1,
       legend_name='Accident Count'
   ).add_to(m)

   # Add popup for each district
   for _, row in accgdf_2020.iterrows():
       folium.Marker(
           location=[row.geometry.centroid.y, row.geometry.centroid.x],
           popup=folium.Popup(row['dname_e'], parse_html=True),
    ).add_to(m)

   # Render folium map using st.write
   folium_static(m)

   
   st.title("Information for each District")
   acc_no2020 = process_acc_data("E:/studyyy/year4/senior pj/test/web_test/count_acc_d2020.csv")
   st.dataframe(acc_no2020)


##################################################################################################################################

with tab3:
   st.header("Global Autocorrelation")
   
   ##########global

   # Building spatial weights object
   w = weights.contiguity.Queen.from_dataframe(accgdf_2020)

   # Plotting to visualize spatial weights
   fig, ax = plt.subplots(figsize=(10, 5))
   plot_spatial_weights(w, accgdf_2020, ax=ax)
   #st.pyplot(fig)

   # Transforming weights into binary (if it's 1 = is neighbor, 0 = not neighbor)
   w.transform = "B"

   # คำนวณและเก็บค่า Moran's I statistic และ p-value
   morans_I, p_value = calculate_and_plot_morans_I(accgdf_2020, w)

   # แสดงค่า Moran's I statistic และ p-value
   st.write("Moran's I statistic:", morans_I)
   st.write("p-value:", p_value)

   # พล็อต scatter plot
   fig, ax = plt.subplots(figsize=(8, 6))
   moran = Moran(accgdf_2020['acc_count'], w)
   moran_scatterplot(moran, ax=ax)
   plt.title('Moran Scatterplot')
   plt.xlabel('Values')
   plt.ylabel('Spatial Lag of Values')
   st.pyplot(fig)

   st.write("--------------------------------------------------------------------------------------")

 ##########local

   st.header("Local Autocorrelation")

   y_acc_count = accgdf_2020['acc_count']

   # Local Moran's I
   acc_count_local_moran = Moran_Local(y_acc_count, w)

   local2020 = plot_local(y_acc_count, acc_count_local_moran)

   # เพิ่มคอลัมน์ Local Moran's I classification และ p-value เข้าสู่ GeoDataFrame
   accgdf_2020 = local_classification(accgdf_2020, acc_count_local_moran)

   # อ่านไฟล์ HTML
   with open("E:/studyyy/year4/senior pj/test/web_test/local_moran_map2020.html", "r") as file:
       html_code = file.read()

   # แสดง HTML บน Streamlit
   st.components.v1.html(html_code, width=800, height=500)

   # รับข้อมูลชื่อเขตตามคลาสของ Moran
   dname_by_moran_class = get_dname_by_moran_class(accgdf_2020)
   # แสดงผลใน Streamlit
   for moran_class, dnames in dname_by_moran_class.items():
       st.write(f"{moran_class} : {', '.join(dnames)}")

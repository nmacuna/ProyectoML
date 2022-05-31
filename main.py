# -*- coding: utf-8 -*-
"""
SCRIPT PRINCIPAL PARA EJECUTAR MODELOS DE MACHINE LEARNING DEL PROYECTO FINAL
DEL CURSO "ANALISIS CON MACHINE LEARNING"

AUTORES:
    - NAYLED ACUÑA
    - CARLOS OLIVEROS
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import pandas as pd
import numpy as np
import itertools

import sys
import os
import os.path as osp

# Librerías adicionales
import streamlit as st
pip install joblib
from joblib import load
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Entrenamiento del modelo de K-means
from sklearn.cluster import KMeans, DBSCAN

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#Entrenamiento del modelo lineal
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# CARGAR FUNCIONES DE LIBRERÍA PERSONAL
from Preprocessing import Preprocesamiento
from Preprocessing import ToPolynomial

from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# 
# =============================================================================

# Configuración básica de la página
st.set_page_config(
  page_title="InfraPlanner", layout="wide",
)

# Logo
col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.write("")

with col3:
    logo=Image.open("Img\Asset_1.png")
    st.image(logo, width=200)

# Header
st.header("Welcome to InfraPlanner")


# Subheader
st.subheader("A planning tool for the infrastructure investment in the departments of Colombia.")

# Clases proporcionadas por el negocio
classes = {
    1: "Desempeño bajo - continentaL.",
    2: "Categoría alto - continental",
    3: "Categoría intermedio . continental",
    4: "Categoría intermedia - isleña"
}

# Al tener que realizar la conexión una vez, utilizamos el decorador para no hacer varios llamados a la vez
@st.experimental_singleton
def upload():
    models = {
    'BestModel_classifier':{
        "name":"Regresion",
        "model": None
        },
    'BestModel_regressor':{
        "name":"Clasificacion",
        "model": None
        },     
    'EscaladoAgrupacion':{
        "name":"Escalado y normalizacion",
        "model": None
        },  
    }
    
    for name, values in models.items():
        values["model"] = load(f"Models/{name}.pkl")

    return models


models = upload() # Cargamos nuestros modelos


# CARGAR DF DE LINEA BASE
df_Base = pd.read_pickle( 'df_Base.pkl' )
df_Base = df_Base.reset_index()
df_Base = df_Base.drop('index',axis=1)

# =============================================================================
# GENERACIÓN DE FUNCIONES DE REGRESIÓN Y DE CLASIFICACIÓN
# =============================================================================

def PredictYClasific(df_entrada,df_Base,dep):
    
    # ------------------------------------------------------------------------------
    # Se realiza la predicción a partir de los deltas de infraestructura registrados
    # 1. Se identifica el departamento de análisis
    df_Dpt_0 = df_Base[ df_Base['Unnamed: 0']==df_Base['Unnamed: 0'][dep] ]
    df_Dpt = df_Base[ df_Base['Unnamed: 0']==df_Base['Unnamed: 0'][dep] ]
    
    # ------------------------------------------------------------------------------
    # 2. Adición de columnas acumulables
    arr_ColsAcum = ['cent_elect','subest','pnts','tnls',
                    'prts','aero','biblio','coleg','uni',
                    'carc','salud','lg_viap','lg_via4',
                    'lg_oleod','lg_lt220','lg_lt500']
    
    df_Dpt_sum = df_Dpt[arr_ColsAcum].add(df_entrada[arr_ColsAcum],fill_value=0)
    df_Dpt_sum = df_Dpt_sum.sum(axis=0).to_frame().T
    
    df_Dpt[arr_ColsAcum] = df_Dpt_sum.loc[0,arr_ColsAcum] 
    
    # ------------------------------------------------------------------------------
    # 3. Reemplazo de columnas reemplazables
    df_Dpt['Poblacion_1000'] = df_entrada['Poblacion_1000'][0]
    
    # ------------------------------------------------------------------------------
    # 4. Preprocesamiento de ambos data frames
    df_Dpt_0_in = Preprocesamiento.Transformar(df_Dpt_0)
    df_Dpt_0_in = df_Dpt_0_in.drop(['PIB per cap'],axis=1)
    
    df_Dpt_in = Preprocesamiento.Transformar(df_Dpt)
    df_Dpt_in = df_Dpt_in.drop(['PIB per cap'],axis=1)
    
    # ------------------------------------------------------------------------------
    # 5. Ejecución de modelos
    df_Dpt_0_out = models['BestModel_regressor']["model"].predict( df_Dpt_0_in )
    df_Dpt_out = models['BestModel_regressor']["model"].predict( df_Dpt_in )
    # ------------------------------------------------------------------------------
    # 6. Cálculo de "impacto"
    dbl_Impacto = df_Dpt_out/df_Dpt_0_out
    # ------------------------------------------------------------------------------
    # 7. Reclasificar
    # Re-transformar con PIB
    df_Dpt_in = Preprocesamiento.Transformar(df_Dpt)
    # Se re-ajusta en PIB base con el incremento esperado
    df_Dpt_in['PIB per cap'] = df_Dpt_in['PIB per cap']*(dbl_Impacto)
    # Escalar/normalizar:
    
    df_Dpt_in_vals = models['EscaladoAgrupacion']["model"].transform( df_Dpt_in.values )
    df_Dpt_in = pd.DataFrame( df_Dpt_in_vals , columns = df_Dpt_in.columns )
    # Calcular clase nueva

    dbl_Clase = models['BestModel_classifier']["model"].predict( df_Dpt_in )
    
    # ------------------------------------------------------------------------------
    # 8. Escribir resultados
    #st.markdown('La inversión en infraestructura tiene el siguiente desempeño:')
    # Impacto
    #st.markdown(f'Un impacto del *{np.round_(dbl_Impacto*100,decimals=2)}* en la economía departamental')
    # Categoría
    #st.markdown(f'El departamento se clasificaría en desempeño en infraestructura como *{classes[dbl_Clase]}*')
    
    st.write("The economic impact will be", float(np.round_((dbl_Impacto-1)*100,decimals=2)),'%')
    st.write('It will be classified on infrastructure development as', classes[dbl_Clase[0]+1])
    if dbl_Clase[0]+1==1:
        img = Image.open("Img\Map_C1.png")
        st.image(img, width=500)
    elif dbl_Clase[0]+1==2:
        img = Image.open("Img\Map_C2.png")
        st.image(img, width=500)
    elif dbl_Clase[0]+1==3:
        img = Image.open("Img\Map_C3.png")
        st.image(img, width=500)
    else:
        img = Image.open("Img\Map_C4.png")
        st.image(img, width=500)
    st.write("Showing Departments with similar infrastructure development and PIB")

# =============================================================================
# GENERACIÓN DE DATAFRAME DE ENTRADA CON BARRAS DESLIZABLES DE INTERFAZ
# =============================================================================

# 1. CREACIÓN DE VENTANAS DE ENTRADA
# - Departamento
# Selection box
 
# first argument takes the titleof the selectionbox
# second argument takes options
list_n = ['Amazonas', 'Antioquia', 'Arauca', 'Atlántico','Bolívar', 'Boyacá','Caldas','Caquetá','Casanare',
 'Cauca','Cesar','Chocó','Cundinamarca','Córdoba','Guainía','Guaviare','Huila','La Guajira','Magdalena','Meta','Nariño',
 'Norte de Santander', 'Putumayo', 'Quindío', 'Risaralda','San Andrés islas','Santander','Sucre', 'Tolima','Valle del Cauca',
 'Vaupés', 'Vichada']
DTO = st.selectbox("Departments of Colombia: ",
                     list_n)
 
# print the selected hobby
st.write("Selected Department: ", DTO)

# INICIALIZAMOS 

# - CENTRALES ELÉCTRICAS
CE = 0
# - SUBESTACIONES
SE = 0
# - PUENTES
PT = 0
# - TÚNELES
TN = 0
# - PUERTOS
PR = 0
# - AEROPUERTOS
AE = 0
# - BIBLIOTECAS
BB = 0
# - COLEGIOS
CO = 0
# - UNIVERSIDADES
UN = 0
# - CÁRCELES
CA = 0
#- SALUD
SA = 0
# - VIAS PRINCIPALES
VP = 0
# - VÍAS 4G
V4 = 0
# - OLEODUCTOS
OL = 0
# - LONG CABLES DE TRANSMISIÓN 220
T2 = 0
# - LONG CABLES DE TRANSMISIÓN 500
T5 = 0

## ENTRADAS

CE = st.slider("Select number of new power stations:", 0, 20)
st.text('Selected: {}'.format(CE))

SE = st.slider("Select number of new subestations:", 0, 20)
st.text('Selected: {}'.format(SE))

PT = st.slider("Select number of new bridges:", 0, 20)
st.text('Selected: {}'.format(PT))

TN= st.slider("Select number of new tunnels:", 0, 20)
st.text('Selected: {}'.format(TN))

PR= st.slider("Select number of new ports:", 0, 15)
st.text('Selected: {}'.format(PR))

AE= st.slider("Select number of new airports:", 0, 15)
st.text('Selected: {}'.format(AE))

BB= st.slider("Select number of new libraries:", 0, 200)
st.text('Selected: {}'.format(BB))

CO= st.slider("Select number of new schools:", 0, 1000)
st.text('Selected: {}'.format(CO))

UN= st.slider("Select number of new universities:", 0, 400)
st.text('Selected: {}'.format(UN))

CA= st.slider("Select number of new prisions:", 0, 400)
st.text('Selected: {}'.format(CA))

SA= st.slider("Select number of new health centers:", 0, 2000)
st.text('Selected: {}'.format(SA))

VP= st.slider("Select number of new kilometers of principal roads:", 0, 5000)
st.text('Selected: {}'.format(VP))

V4= st.slider("Select number of new kilometers of roads 4G:", 0, 5000)
st.text('Selected: {}'.format(V4))

OL= st.slider("Select number of new kilometers of oil pipeline:", 0, 5000)
st.text('Selected: {}'.format(OL))

T2= st.slider("Select number of new kilometers of electric transmission line 220 kv:", 0, 5000)
st.text('Selected: {}'.format(T2))

T5= st.slider("Select number of new kilometers of electric transmission line 500 kv:", 0, 5000)
st.text('Selected: {}'.format(T5))

PO= st.slider("Expected population (in thousands):", 1, 10000)
st.text('Selected: {}'.format(PO))
# ------------------------------------------------------------------------------
# 2. COMPILACIÓN DE DATOS EN DATAFRAME DE ENTRADA:
df_entrada = list(zip(list([DTO]),
                      list([0]),  # Este no se modifica -> PIB p.c. (Aún no lo conocemos)
                      list([CE]),
                      list([SE]),
                      list([PT]),
                      list([TN]),
                      list([PR]),
                      list([AE]),
                      list([BB]),
                      list([CO]),
                      list([UN]),
                      list([CA]),
                      list([SA]),
                      list([VP]),
                      list([V4]),
                      list([OL]),
                      list([T2]),
                      list([T5]),
                      list([0]), # Este no se modifica -> Área departamental
                      list([PO]),
                      ))

df_entrada = pd.DataFrame( df_entrada , columns = df_Base.columns )
          
# ------------------------------------------------------------------------------
# 3. BOTÓN PARA EJECUTAR:        

# Create a button, that when clicked, shows a text
if(st.button("Calculate performance of the infrastructure investment")):
    PredictYClasific(df_entrada, df_Base,list_n.index(DTO))


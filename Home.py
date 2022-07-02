"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
from RegresionLineal import * 
from RegresionPolinomial import *
from ClasificadorGaussiano import *


#importo otra page
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Welcome Proyecto 2 - 201408419')

st.subheader('Configuracion de visualizacion')
uploaded_file = st.file_uploader(label = "Sube tu archivos.",type=['csv','xls','xlsx','json'])

df = ""
if uploaded_file is not None:
  #si no es nullo
  print(uploaded_file)
  print(uploaded_file.type)
  print('hello')
  try:
    if uploaded_file.type == 'text/csv':
      print('csv')
      st.text("csv")
      df = pd.read_csv(uploaded_file)
      #settitlepage2(df)
    if uploaded_file.type == 'application/vnd.ms-excel':
      print('xls')
      st.text("xls")
      #instalar una libreria para reconocer xls
      #pip install xlrd
      df = pd.read_excel(uploaded_file)
    if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
      print("xlsx")
      st.text("xlsx")
      df = pd.read_excel(uploaded_file)
    if uploaded_file.type == 'application/json':
      print("json")
      st.text("json")
      df = pd.read_json(uploaded_file)
    #df = pd.read_json(uploaded_file)
    #st.write(df)
  except Exception as e: 
    print(e)
    st.subheader('Error al cargar archivo: ',e)

#dropbutton
option = st.selectbox(
     'Algoritmos',
     ('Seleccione una opcion','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Clasificador de árboles de decisión','Redes neuronales'))

st.write('You selected:', option)


if option == 'Regresión lineal':
  RL(df)
elif option == 'Regresión polinomial':
  RPol(df)
elif option == 'Clasificador Gaussiano':
  ClGaussiano(df)
  #param = st.text_input('Ingrese parametro de aproximacion','I')
  #data_top = df.columns.values
  #listaa = data_top.tolist()

  #print("Clasificador Gaussiano")
  #ClGaussiano(df)
elif option == 'Clasificador de árboles de decisión':
  print("Clasificador de árboles de decisión")
elif option == 'Redes neuronales':
  print("Redes neuronales")









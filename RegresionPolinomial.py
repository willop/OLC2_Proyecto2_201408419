import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def RPol(_info):
    st.title("Regresion Polinomial")
    st.subheader('Informacion')
    st.write(_info)
    st.subheader('Parametros')
    col1,col2,col3 = st.columns(3)
    with col1:
        paramx = st.text_input('Ingrese parametro X','NO')
    #st.write(paramx)
    with col2:
        paramy = st.text_input('Ingrese parametro Y','A')
    #st.write(paramy)
    with col3:
        grado = st.text_input('Ingrese el grado de la ecuacion','2')

    co1,co2,co3 = st.columns(3)
    with co2:
        prediccion = st.text_input('Ingrese valor para aproximar','100')

    x = _info[paramx]
    y = _info[paramy]

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[:,np.newaxis]
    y = y[:,np.newaxis]

    #plot si se quiere
    #fig1, ax = plt.subplots()
    #ax.scatter(x,y)
 
    #---------------------Paso 2-------------------------------------------
    #---------------------Preparacion de la informacion--------------------
    nb_degree = int(grado)
    polinomial_feature = PolynomialFeatures(degree = nb_degree)
    X_TRANSF = polinomial_feature.fit_transform(x)

    #---------------------Paso 3-------------------------------------------
    #---------------------Definir y entrenar el modelo---------------------
    model = LinearRegression()
    model.fit(X_TRANSF,y)

    #---------------------Paso 4-------------------------------------------
    #---------------------Calcular bayesiana-------------------------------
    Y_NEW = model.predict(X_TRANSF)
    
    rmse = np.sqrt(mean_squared_error(y,Y_NEW))
    r2 = r2_score(y, Y_NEW)

    #---------------------Paso 5-------------------------------------------
    #---------------------Prediccion---------------------------------------
    x_new_min = 0.0
    x_new_max = float(prediccion)  #para el calculo de la prediccion

    X_NEW = np.linspace(x_new_min,x_new_max,50)
    X_NEW = X_NEW[:,np.newaxis]

    X_NEW_TRANSF = polinomial_feature.fit_transform(X_NEW)
    Y_NEW = model.predict(X_NEW_TRANSF)

    fig, ax = plt.subplots()
    plt.plot(X_NEW, Y_NEW, color='blue',linewidth=3)
    plt.grid()
    plt.xlim(x_new_min, x_new_max)
    title = 'Degree = {}; RMSE = {}; R2={};'.format(nb_degree,rmse,r2)
    plt.title("Regresion polinomial\n"+title ,fontsize = 10)
    plt.xlabel(paramx)
    plt.ylabel(paramy)

    #---------------------Paso 6-----------------------------------------
    #---------------------Impresion de resultados------------------------
    st.subheader('Resultados')
    d = {'RMSE' : rmse,'R2':  r2,'Prediccion':Y_NEW[Y_NEW.size-1]}
    dresult = pd.DataFrame(data = d)
    st.dataframe(dresult)

    with st.expander("Ver grafica regresion polinomial"):
        st.pyplot(fig)
    
    with st.expander("Ver grafica de Puntos"):
        fig2,ax2 = plt.subplots()
        ax2.scatter(x,y, color='black')
        st.pyplot(fig2)

    

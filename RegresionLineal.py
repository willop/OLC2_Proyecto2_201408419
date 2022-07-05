import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def RL(_info):
    print("Hola desde regresion lineal")
    st.title('Regresion lineal')
    st.subheader('Informacion')
    st.write(_info)
    st.subheader('Parametros')
    col1,col2 = st.columns(2)
    with col1:
        paramx = st.text_input('Ingrese parametro X','NO')
    #st.write(paramx)
    with col2:
        paramy = st.text_input('Ingrese parametro Y','A')
    #st.write(paramy)

    #inicio de la regresion lineal
    x = np.asarray(_info[paramx]).reshape(-1,1)
    y = _info[paramy]
    
    regr = linear_model.LinearRegression()
    regr.fit(x,y)

    y_pred = regr.predict(x)
    regresion = regr.coef_


    #plt.scatter(x,y, color='black')
    #plt.plot(x,y_pred, color='blue', linewidth=3)
    #st.pyplot(plt.show())

    st.subheader('Resultados')
    color = st.select_slider(
     'Seleccione un color para la recta',
     options=['black','red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    color2 = st.select_slider(
     'Seleccione un color para los puntos',
     options=['black','red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    fig, ax = plt.subplots()
    ax.scatter(x,y, color=color2)
    ax.plot(x,y_pred,color = color)
    plt.title('Regresio lineal\nCoeficiente de regresion: '+str(regresion))#,'  con un error cuadratico: ',mean_squared_error(y,y_pred))
    plt.xlabel(paramx)
    plt.ylabel(paramy)
    plt.grid()
    #st.pyplot(fig)
    texto = str(round(regr.coef_[0],2))+"X+"+str(round(regr.intercept_,2))
    print("Ecuacion ",texto)
    d = {'Coeficiente de regresion': [regresion], 'Error cuadratico':[mean_squared_error(y,y_pred)], 'Coeficinte de determinacion':[r2_score(y,y_pred)],'Ecuacion lineal Ax + B':texto}
    dresult = pd.DataFrame(data=d)
    st.dataframe(dresult)

    with st.expander("Ver grafica regresion lineal"):
        st.pyplot(fig)
    
    with st.expander("Ver grafica de Puntos"):
        fig2,ax2 = plt.subplots()
        ax2.scatter(x,y, color='black')
        st.pyplot(fig2)

#aproximacion
    c1,c2,c3 = st.columns(3)
    with c2:
        #print(texto)
        #st.subheader(texto)
        calcular = st.text_input('Valor para aproximacion','0')
        variable = regr.predict([[int(calcular)]])
        st.text(variable)
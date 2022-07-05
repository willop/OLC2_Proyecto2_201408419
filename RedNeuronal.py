import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def RedNeuronal(_info):
    st.title('Redes neuronales')
    st.subheader('Informacion')
    st.write(_info)
    st.subheader('Parametrizacion de redes neuronales')
    param = st.text_input('Ingrese parametro de aproximacion','I')

    data_top = _info.columns.values
    listaa = data_top.tolist()

    #eliminar una columna no deseada
    listaaux = ["Seleccionar"]+listaa
    eliminarcolumna =st.multiselect('Eliminar una columna?',listaaux,['Seleccionar'])
    _aux = _info
    _info = _info.drop(str(param),axis=1)

    # elimino de la tabla el parametro predecir
    if eliminarcolumna[0] != 'Seleccionar':
        for i in eliminarcolumna:
            aux = str(i)
            print('Valor a eliminar')
            print(aux)
            _info = _info.drop(aux,axis=1)
    # ahora con el eliminado buscarlo y guardarlo
    result = _aux[param]

    x=_info[:].astype(int)
    y = result.astype(int)

    st.text('Parametro en x')
    st.write(x)
    st.text('Parametro en y')
    st.write(y)

    nn = MLPClassifier(hidden_layer_sizes=(3,3,3),max_iter=1000)
    st.text('Parametro nn')
    st.write(nn)

    lasso = linear_model.Lasso()
    scores = cross_val_score(nn,x,y)
    st.text('Scores')
    st.write(scores)
    info = 'Score/puntuacion '+str(scores.mean())+' -Std. Desviacion '+str(scores.std())
    st.subheader('Score')
    st.text(info)

    ## no incluir float ni palabras
    st.title('Predecir un valor')
    textomostrar = 'Ingrese '+str(len(x.columns))+' parametros separados por coma (,)'
    prediccion = st.text_input(textomostrar,'')
    if prediccion != '':
        entrada = prediccion.split(",")
        map_obj = list(map(int,entrada))
        map_obj = np.array(map_obj)
        print('arreglo de entrada: ',entrada)
        nn.fit(x,y)
        st.header('Resultado:')
        st.subheader(nn.predict([map_obj]))

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:48:13 2021

@author: josem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


ab =pd.read_excel(r"C:\Users\josem\OneDrive\Escritorio\Regresión Lineal\Libro1.xlsx")

#tomo el valor de la columna de edad
#Esta es la variable independiente 
x= ab.iloc[:,1].values
#Se reforma el x
x=x.reshape(-1,1)
# Selecciono la presión como la variable dependiente
#si no coloco el values la resultante es una serie
y= ab.iloc[:,2].values
x_train,x_test,y_train,y_test=train_test_split(x, y)
# llamo un objeto tipo regresión lineal
Lin_reg=LinearRegression()
#Adecuamos los valores  al mismo modelo, es decir entrenamos el modelo 
Lin_reg.fit(x_train,y_train)
# #Se predice la precion de la sangre
y_pred=Lin_reg.predict(x_test)
# #Si el score es > a 0.5 la predición es 
print(Lin_reg.score(x_test,y_test))
print(Lin_reg.predict([[20]]))

#Grafica del conjunto de datos de entrenamiento

plt.scatter(x_train,y_train)
#Grafica de datos para evaluar modelo
plt.scatter(x_test,y_test)
#Grafica linea de regresión
plt.plot(x_train,Lin_reg.predict(x_train),color='r')

plt.figure(2)


#Grafica conjunto de datos de evaluación

plt.scatter(x_test,y_test)
plt.plot(x_test,Lin_reg.predict(x_test),color='r')
plt.xlabel("age")
plt.ylabel("BP")
plt.title("train dataset")
plt.show()
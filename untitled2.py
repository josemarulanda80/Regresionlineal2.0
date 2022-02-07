# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:14:46 2021

@author: josem
"""


#k Nearest Neighbor Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_excel(r'C:\Users\josem\OneDrive\Escritorio\Regresión Lineal\Libro2.xlsx')
df.head()
scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))

sc_feat=scaler.transform(df.drop('TARGET CLASS', axis=1))

df_feat=pd.DataFrame(sc_feat,columns=df.columns[:-1])
df_feat.head()
x=df_feat
y=df['TARGET CLASS']
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#usando knn

from sklearn.neighbors import KNeighborsClassifier

#Creo el modlo con n =1

knn=KNeighborsClassifier(n_neighbors=1)

#Entreno el modelo

knn.fit(x_train,y_train)

#predición y evaluación
pred = knn.predict(x_test)
print(confusion_matrix(y_test,pred))

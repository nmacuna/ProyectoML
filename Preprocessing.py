# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:07:57 2022

@author: 
    - NAYLED ACUÑA
    - CARLOS OLIVEROS
"""

import pandas as pd
import numpy as np
import itertools

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class Preprocesamiento():
    
    # Función final
    def Transformar(X,y=None):
        
        X_var =  pd.DataFrame(X).copy()
        
        def NormalizarPorAreasyPob(X,y=None):
            
            X_var = pd.DataFrame(X).copy()
            
            # Por población:
            X_var['cent_elect'] = X_var.apply(lambda ii: ii['cent_elect']/(ii['Poblacion_1000']/1000), axis=1)
            X_var['subest'] = X_var.apply(lambda ii: ii['subest']/(ii['Poblacion_1000']/1000), axis=1)
            X_var['biblio'] = X_var.apply(lambda ii: ii['biblio']/(ii['Poblacion_1000']/1000), axis=1)
            X_var['coleg'] = X_var.apply(lambda ii: ii['coleg']/(ii['Poblacion_1000']/1000), axis=1)
            X_var['uni'] = X_var.apply(lambda ii: ii['uni']/(ii['Poblacion_1000']/1000), axis=1)
            X_var['carc'] = X_var.apply(lambda ii: ii['carc']/(ii['Poblacion_1000']/1000), axis=1)
            X_var['salud'] = X_var.apply(lambda ii: ii['salud']/(ii['Poblacion_1000']/1000), axis=1)
            
            
            # Por área:
            X_var['pnts'] = X_var.apply(lambda ii: ii['pnts']/(ii['Área_km2']/1000), axis=1)
            X_var['tnls'] = X_var.apply(lambda ii: ii['tnls']/(ii['Área_km2']/1000), axis=1)
            X_var['prts'] = X_var.apply(lambda ii: ii['prts']/(ii['Área_km2']/1000), axis=1)
            X_var['aero'] = X_var.apply(lambda ii: ii['aero']/(ii['Área_km2']/1000), axis=1)
            X_var['lg_viap'] = X_var.apply(lambda ii: ii['lg_viap']/(ii['Área_km2']/1000), axis=1)
            X_var['lg_via4'] = X_var.apply(lambda ii: ii['lg_via4']/(ii['Área_km2']/1000), axis=1)
            X_var['lg_oleod'] = X_var.apply(lambda ii: ii['lg_oleod']/(ii['Área_km2']/1000), axis=1)
            X_var['lg_lt220'] = X_var.apply(lambda ii: ii['lg_lt220']/(ii['Área_km2']/1000), axis=1)
            X_var['lg_lt500'] = X_var.apply(lambda ii: ii['lg_lt500']/(ii['Área_km2']/1000), axis=1)
            
            return X_var
        
        def AcumularVariables(X,y=None):
            
            X_var = pd.DataFrame(X).copy()
            
            X_var['lg_viap'] = X_var['lg_viap'] + X_var['lg_via4']
            X_var.drop('lg_via4', inplace = True , axis=1)
                    
            return X_var
        
        
        X_var = NormalizarPorAreasyPob(X_var)
        X_var = AcumularVariables(X_var)
        X_var.drop('Área_km2', inplace = True , axis=1)
        X_var.drop('Poblacion_1000', inplace = True , axis=1)
        X_var.drop('Unnamed: 0', inplace = True , axis=1)
                
        return X_var
    

class ToPolynomial():
    def __init__(self,k=2):
        self.k = k

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        columns = X.columns
        X_train_pol = pd.concat([X**(i+1) for i in range(self.k)],axis=1) #Polinomios sin interacciones
        X_train_pol.columns = np.reshape([[i+' '+str(j+1) for i in columns] for j in range(self.k)],-1)
        temp = pd.concat([X[i[0]]*X[i[1]] for i in list(itertools.combinations(columns, 2))],axis=1) #Combinaciones sólo de grado 1
        temp.columns = [' '.join(i) for i in list(itertools.combinations(columns, 2))]
        X_train_pol = pd.concat([X_train_pol,temp],axis=1)
        return X_train_pol

# Importo librerias a usar

# Librerias comunes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
# Librerias de sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
# Librerias de keras 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from keras import regularizers
from keras.layers import Dropout
from keras import optimizers
from tensorflow.python.keras.models import load_model

##############################################################################################################

def plot_CM(cm, classes):
    
    ''' Esta funcion grafica la matriz de confusion que se le pase como parametro, con las clases indicadas'''
    
    fig, ax = plt.subplots(figsize = (15,10))
    im = ax.imshow(np.array(cm))

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, np.array(cm)[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Valor Predicho')
    ax.set_ylabel('Valor Real')
    fig.tight_layout()
    plt.show()
    
##############################################################################################################

def corr_elimination(X_train, max_corr = 0.9):
    
    """
    Esta funcion calcula la correlacion entre las variables de entrada. 
    Luego va eliminando de a una empezando por la que tiene mas variables 
    correlacionadas. 
    """
    
    corr = X_train.corr()
    
    cor_var = []
    while True:
        corr[(corr>max_corr) | (corr<-max_corr)] = 1
        corr[(corr<max_corr) & (corr>-max_corr)] = 0
        n_corr = corr.sum()-1
        m = max(n_corr) # m = min(n_corr[n_corr > 0], default=0) # esta opcion elimina mas variables
        if m>=1:
            M = n_corr[n_corr==m].index[0]
            cor_var.append(M)
            corr.loc[M,:] = 0
            corr.loc[:,M] = 0
        else:
            break
        
    print('Se eliminaron',len(cor_var),'de',len(corr.columns),'variables.')

    return cor_var

##############################################################################################################

def forest_elimination(X_train,y_train, total = 10):

    """
    Esta funcion elimina variables de entrada utilizando un random forest.
    Lo que hace es agregar una variable randon y entrenar un modelo de
    random forest varias veces. Luego se calcula la importancia de las 
    variables para clasificar los datos y se eliminan todas ellas que tengan
    un nivel de importancia menor a la variable random.
    """
        
    X_train['random'] = [np.random.randn() for i in range(len(X_train))]
    
    rank = np.zeros(len(X_train.columns))
    
    for i in range(total):
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, verbose=3, n_estimators=50)
        rf.fit(X_train, y_train)
        rank += rf.feature_importances_
        
    ranking = np.zeros(len(rank))
    for i in range(len(rank)):
        ranking[rank==max(rank)] = i
        rank[rank==max(rank)] = 0

    ranking = pd.DataFrame({'feacture': X_train.columns, 'ranking': ranking}).sort_values(['ranking']).reset_index(drop=True)
    print(ranking)
    
    m = ranking[ranking['feacture']=='random']['ranking'].values[0]
    rf_var = list(ranking.loc[m+1:,'feacture'].values)   
    print('Se eliminaron',len(rf_var),'de',len(X_train.columns)-1,'variables.')
    
    return rf_var

##############################################################################################################

def Class_nn_model(X_train,NumL1, NumL2, NumL3):
    
    """
    Modelo de red neuronal
    """
    
    model = Sequential()

    model.add(Dense(NumL1,
                    kernel_initializer=keras.initializers.he_normal(seed=10),
                    bias_initializer='zeros',
                    input_dim=len(X_train.columns),
                    activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)))
    
    model.add(Dropout(0.5))
    
    if NumL2 != None:
        model.add(Dense(NumL2,
                        kernel_initializer=keras.initializers.he_normal(seed=10),
                        bias_initializer='zeros',
                        activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)))

        model.add(Dropout(0.5))
    
    if NumL3 != None:
        model.add(Dense(NumL3,
                        kernel_initializer=keras.initializers.he_normal(seed=10),
                        bias_initializer='zeros',
                        activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)))

        model.add(Dropout(0.5))

    model.add(Dense(4,
                    kernel_initializer=keras.initializers.he_normal(seed=10),
                    bias_initializer='zeros',
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(0.001)))

    ADAM = optimizers.Adam(lr=0.0002)
    model.compile(loss='binary_crossentropy',
                  optimizer=ADAM,
                  metrics=['accuracy'])
    
    return model
##############################################################################################################

def ROC_curve(Class_NNmodel, X_train, y_train):
    
    """
    En esta funcion creamos la curva ROC y calculamos el valor de AUC promedio
    para todas las curvas.
    """
    
    fpr_train = []
    tpr_train = []
    thresholds_train = []
    y_train_pred     = Class_NNmodel.predict(X_train)

    AUC = 0
    for i in range(4):
        fpr, tpr, thresholds = roc_curve(np.array(y_train)[:,i], y_train_pred[:,i], pos_label=1)
        AUC += roc_auc_score(np.array(y_train)[:,i], y_train_pred[:,i])/4.0
        fpr_train.append(fpr)
        tpr_train.append(tpr)
        thresholds_train.append(thresholds)

    fig = plt.figure(figsize=(6,5))
    for i in range(4):
        plt.plot(fpr_train[i],tpr_train[i])
    plt.plot([0,1],[0,1],'b--',lw=2)
    plt.legend(['0','1','2','3'])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.grid()
    plt.text(0.2, 0.0, 'AUC = {:.2f}'.format(AUC), fontsize=20)
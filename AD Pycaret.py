# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pycaret.anomaly import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import psycopg2
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from datetime import datetime as dt, timedelta
import time
import matplotlib.pyplot as plt  
import seaborn as sns 
import pytz
import os
import math 



print('pass n days data create mean median and standardize')
def standardization_function(db,metric_list):
    df=db.copy()
    for metric in metric_list:
        average=df[metric].mean()
        standard_deviation=df[metric].std()
        df['standardized']=(df[metric]-average)/standard_deviation
        df.rename(columns={'standardized':metric+'_standardized'},inplace=True)
    return df

print('get probability of the points')
def get_probability(start_value, end_value, eval_points, kd):
    N = eval_points                                      
    step = (end_value - start_value) / (N - 1)  
    x = np.linspace(start_value, end_value, N)[:, np.newaxis]  
    kd_vals = np.exp(kd.score_samples(x)) 
    probability = np.sum(kd_vals * step)  
    return probability.round(4)


print('fit kernel function')
def fit_kernel_functions(db,metric_list):
    for metric in metric_list:
        array=np.array(db[metric]).reshape(-1, 1)
        array_min=array.min()-1.5
        kd = KernelDensity(kernel='gaussian',bandwidth=0.5).fit(array)
        db['probability']=db[metric].apply(lambda x: get_probability(array_min,x,100,kd))
        db.rename(columns={'probability':metric+'_probability'},inplace=True)
        
    return db


def predict_probability(db,train_time,metric_list,time_col):
    db=standardization_function(db,metric_list)
    train_db=db[db[time_col]<=train_time]
    test_db=db[db[time_col]>train_time]
    column_list=[i+'_standardized' for i in metric_list]
    train_db_probs, test_db_probs=fit_kernel_functions(train_db,test_db,column_list)
    return train_db_probs,test_db_probs



from sklearn.neighbors import KernelDensity
import gc


######PYCARET ANOMALY DETECTION MODELS ########
def anomaly_models(df,model_list,metric,time_col,train_time):
    model_db=df.copy()
    for i in model_list:
        data=df.copy()
        print(i)
        train_data=data[data[time_col]<train_time]
        test_data=data[data[time_col]>=train_time]
        print(len(test_data))
        s = setup(train_data, session_id = 123,silent=True)
        model_name = create_model(i, fraction = 0.1)
        model_results_train = assign_model(model_name)
        model_results_train.drop(columns=metric,inplace=True)

        model_results_test = predict_model(model_name,test_data)
        model_results_test.drop(columns=metric,inplace=True)

        
        model_results=pd.concat([model_results_train,model_results_test])
        del model_results_train
        del model_results_test
        
        
        model_results.rename(columns={'Anomaly':'Anomaly_'+i,'Anomaly_Score':'Anomaly_Score_'+i},inplace=True)
        model_db=model_db.merge(model_results,on=[time_col])
    return model_db

from sklearn.neighbors import KernelDensity
import gc





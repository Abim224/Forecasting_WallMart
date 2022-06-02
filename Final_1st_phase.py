# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 00:59:49 2022

@author: Abinash.m
"""
import datetime as dt
import streamlit as st
import pandas as pd
from ThymeBoost import ThymeBoost as tb
import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller,acf, pacf
from math import sqrt
import base64
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from statsmodels.tsa.stattools import kpss
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import statsmodels.api as sm
import itertools
from ThymeBoost import ThymeBoost as tb
from pmdarima.arima import AutoARIMA
from datetime import datetime

#----------------------------------------GETTING DATA------------------------------------------------
def get_df(data):
  custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
  extension = data.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(data,index_col=0,squeeze=True,parse_dates=True,
                     date_parser=custom_date_parser,
                     infer_datetime_format=True,dayfirst=True)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(data, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(data) 
  return df

#-----------------------------------MODEL1-------------------------------------------------------


def arima(df,train,test):
    model=auto_arima(train,start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=50)
    prediction = pd.DataFrame(model.predict(n_periods = len(test)),index=test.index)
    prediction.columns = ['predicted_data']
    test['predicted_data'] = prediction
    rmse= sqrt(mean_squared_error(test.iloc[:,0], test['predicted_data']))
    r2_scor =r2_score(test.iloc[:,0],test['predicted_data'])
    r2_scor =r2_scor*100
    mae =mean_absolute_error(test.iloc[:,0],test['predicted_data'])
    mape =mean_absolute_percentage_error(test.iloc[:,0],test['predicted_data'])
    mape = mape*100
    #st.write(rmse)
    #st.write(r2_scor)
    #st.write(mae)
    #st.write(mape)
    
    
    x =df.index[-1]
    rng = pd.date_range(x, periods=25, freq='M')
    pred = pd.DataFrame(model.predict(n_periods = 25),index=rng)
    #future= model.predict(n_periods=43, typ='linear')
    #pred = pd.DataFrame({ 'Date': rng, 'ARIMA': future})
    #st.table(pred)
    
    return rmse,r2_scor,mae,mape,pred


#-----------------------------------MODEL2---------------------------------------------------------



def sarima(df):
    train = df[:int(len(df)*.75)]
    test = df[int(len(df)*.75):]
    p = d = q = range(0, 1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    order =[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results = mod.fit()
                order.append((param,param_seasonal,results.aic))
                #print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
                #st.write(results.aic)
            except: 
                continue
    order_df = pd.DataFrame(order,columns=['Order','Seasonal_order','AIC'])
    order_df =order_df.sort_values('AIC')
    pdq_order = order_df['Order'].iloc[0]
    seasonal = order_df['Seasonal_order'].iloc[0]
    mod = sm.tsa.statespace.SARIMAX(df,
                                order=(0, 0, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    sarima = mod.fit()
    pred = sarima.get_prediction(start=len(train), dynamic=False)
    pred_ci = pred.conf_int()
    y_pred = pred.predicted_mean
    mse = mean_squared_error(test, y_pred)
    rmse =sqrt(mse)
    r2_scor =r2_score(test,y_pred)
    
    mae =mean_absolute_error(test,y_pred)
    mape =mean_absolute_percentage_error(test, y_pred)
    mape = mape*100
    #pred_uc = sarima.get_forecast(steps=12)
    #pred_ci = pred_uc.conf_int()
    x =df.index[-1]
    rng = pd.date_range(x, periods=25, freq='M')
    pred_uc = sarima.get_forecast(steps=25)
    
    pred_ci = pred_uc.conf_int()
    forecast = pred_uc.predicted_mean
    #pred = pd.DataFrame(forecast,index=rng)
    pred = pd.DataFrame({ 'Date': rng, 'ARIMA': forecast})

    
    #st.write(pred_ci)
    #st.write(rmse)
    #st.write(r2_scor)
    #st.write(mae)
    #st.write(mape)
    #st.write(pred)
    return rmse,r2_scor,mae,mape,pred
#---------------------------------------MODEL 3---------------------------------------------------------
def auto_model(df):
    size = round(int(len(df)*.80))
    train, test = model_selection.train_test_split(df.iloc[:,0], train_size=size)
    
    # Let's create a pipeline with multiple stages... the Wineind dataset is
    # seasonal, so we'll include a FourierFeaturizer so we can fit it without
    # seasonality
    pipe = pipeline.Pipeline([
        ("fourier", ppc.FourierFeaturizer(m=12, k=4)),
        ("arima", AutoARIMA(stepwise=True, trace=1, error_action="ignore",
                                  seasonal=False,  # because we use Fourier
                                  suppress_warnings=True))
    ])
    
    pipe.fit(df)
    x =df.index[-1]
    rng = pd.date_range(x, periods=25, freq='M')
    preds, conf_int = pipe.predict(n_periods=25, return_conf_int=True)
    
    auto_pred=pd.DataFrame(preds)
    m=pd.DataFrame(conf_int)
    auto_pred['Upper Bound'] = m.iloc[:,1]
    auto_pred['Lower Bound'] = m.iloc[:,0]
    pred = pd.DataFrame({ 'Date': rng, 'Forecast value': preds})
    pred = pd.DataFrame(pred)
    pred = pred.set_index('Date')
    return pred



#---------------------------------------MODEL 4----------------------------------------------------------
def enhanced_auto_ml_model(df):
    
    if df is not None:
        df.columns=['ds','y']
        df['ds'] = pd.to_datetime(df['ds'],errors='coerce') 
        
        
        
        max_date = df['ds'].max()
        #st.write(max_date)
    
    periods_input = 25
    
    if df is not None:
        m = Prophet()
        m.fit(df)

    if df is not None:
        future = m.make_future_dataframe(periods=periods_input,freq ='M')
        
        forecast = m.predict(future)
        fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
        fcst_filtered =  fcst[fcst['ds'] > max_date]    
        #st.write(fcst_filtered)
        metric_df = fcst.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
       
        mse = mean_squared_error(metric_df.y, metric_df.yhat)
        rmse =sqrt(mse)
        r2_scor =r2_score(metric_df.y, metric_df.yhat)
        
        mae =mean_absolute_error(metric_df.y, metric_df.yhat)
        mape =mean_absolute_percentage_error(metric_df.y, metric_df.yhat)
        mape = mape*100
        pred = pd.DataFrame(fcst_filtered)
        pred = pred.set_index('ds')
        
    
    return rmse,r2_scor,mae,mape,pred
#---------------------------------------MODEL 5-------------------------------------------------------
def enhanced_arima_model(df,train,test):
    boosted_model = tb.ThymeBoost(verbose=0)
    model = boosted_model.autofit(train.iloc[:,0],
                               seasonal_period=0)
    predicted_output = boosted_model.predict(model, forecast_horizon=len(train))
    mse = mean_squared_error(train.iloc[:,0], predicted_output['predictions'])
    rmse =sqrt(mse)
    r2_scor =r2_score(train.iloc[:,0], predicted_output['predictions'])
    
    mae =mean_absolute_error(train.iloc[:,0], predicted_output['predictions'])
    mape =mean_absolute_percentage_error(train.iloc[:,0], predicted_output['predictions'])
    mape = mape*100
    boosted_model1 = tb.ThymeBoost(verbose=0)
    model1 = boosted_model1.autofit(df.iloc[:,0],
                               seasonal_period=12)
    x =df.index[-1]
    rng = pd.date_range(x, periods=25, freq='M')
    predicted_output1 = boosted_model1.predict(model1, forecast_horizon=len(rng))

    return rmse,r2_scor,mae,mape,predicted_output1
#---------------------------------------DOWNLOAD THE FILE----------------------------------------
def download(df):
    csv_exp = df.to_csv(index=True)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
    #st.table(df)
#---------------------------------------MAIN BLOCK-------------------------------------------------------  
def main():
    st.header('Upload the data with date column')
    data = st.file_uploader("Upload file", type=['csv' ,'xlsx','pickle'])
    if not data:
      st.write("Upload a .csv or .xlsx file to get started")
      return
    df =get_df(data)
    #pro = get_df1(data)
    df =pd.DataFrame(df)
    cols = st.selectbox(
       'Please select a column',df.columns.tolist())
    df = df[cols]
    #pro = pro[cols]
    df =pd.DataFrame(df)
    #pro = df.copy()
    #df= pd.to_datetime(df.index,infer_datetime_format=True,format='%Y-%m-%d',exact=True)
    pred = df.copy()
    pred = pred.reset_index()
    #pred= pd.to_datetime(pred.iloc[:,0],infer_datetime_format=True,format='%Y-%m-%d',exact=True)
    train = df[:int(len(df)*.75)]
    test = df[int(len(df)*.75):]
    model1 = arima(df,train,test)
    model2 =sarima(df)
    model3 =enhanced_auto_ml_model(pred)
    model5=enhanced_arima_model(df, train,test)
    model4 = auto_model(df)
    my_dict ={'RMSE':[model1[0],model2[0],model3[0],model5[0]],
              'R2_SCORE':[model1[1],model2[1],model3[1],model5[1]],
              'MAE':[model1[2],model2[2],model3[2],model5[2]],
              'MAPE':[model1[3],model2[3],model3[3],model5[3]]
             }
    my_df=pd.DataFrame(my_dict,index=['ARIMA','SARIMA','ENHANCED AUTO ML MODEL','ENHANCED ARIMA MODEL'])
    st.subheader('EVALUATION METRICS')
    st.table(my_df)
    
    #st.subheader('Auto_arima')
    #model(df, train, test)
    #st.subheader('SARIMAX')
    #auto(df)
    #st.subheader('MODEL 4')
    #prophet(pred)
    #st.subheader('MODEL 1')
    #st.table(model1[4])
    #st.subheader('MODEL 2')
    #st.table(model2[4])
    #st.table(df)
    if model5[3]<float(20):
        st.write('ENHANCED ARIMA MODEL is the best model for the dataset ')
        #csv_exp = model5[4].to_csv(index=True)
        download(model5[4])
        st.line_chart(model5[4].iloc[:,0])
        st.balloons()
        
    elif model3[3]<float(20):
        st.write('ENHANCED AUTO ML MODEL is the best model for the dataset ')
        download(model3[4])
        
        st.line_chart(model3[4].iloc[:,0])
        st.balloons()

    elif model2[3]<float(20):
        st.write('SARIMA MODEL is the best model for the dataset ')
        download(model2[4])
        st.line_chart(model2[4].iloc[:,1])
        st.balloons()
    elif model1[3]<float(20):
        st.write('ARIMA MODEL is the best model for the dataset ')
        download(model1[4])
        st.line_chart(model1[4].iloc[:,0])
        st.balloons()
        
    else:
        st.write('ARIMA MODEL WITH PIPELINE   is the best model for the dataset ')
        download(model4[4]) 
        st.line_chart(model4.iloc[:,0])
        st.balloons()
        
        
main()
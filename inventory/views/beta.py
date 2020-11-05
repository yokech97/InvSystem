from django.shortcuts import render
import pandas as pd
# from sklearn.compose import ColumnTransformer 
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np 
# import the BaseEstimator
import matplotlib.pyplot as plt

import matplotlib
import datetime as dt
from django.contrib.auth.decorators import login_required
from inventory.models import *
# import plotly.plotly as py
from plotly.offline import plot
import plotly.graph_objects as go
# import plotly.express as px
from inventory.forms import *

@login_required
def index(request):

    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'k'
    form = HomeForm(request.POST)
    data = ''
    if request.method == "POST":
            if form.is_valid():
                data=form.cleaned_data.get('post')
    if data=='':
        figure = go.Figure()
        figure.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
        fig= plot(figure,output_type='div')
        figure1 = go.Figure()
        figure1.update_layout(title='Prediction of Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
        fig1= plot(figure1,output_type='div')        
        context ={
        'fig':fig,
        'fig1':fig1,
        'form': form,
        'data':data,
        
        }
        return render(request, 'inv/home.html',context)
    sales=pd.DataFrame.from_records(sales_record.objects.filter(item_code=data).values('record_id','item_quantity_before_sales','item_quantity_sold','item_quantity_after_sales','date_sold','item_code_id'))
    cols = ['record_id','item_code_id']
    sales.drop(cols, axis=1, inplace=True)

    sales['date_sold'] = pd.to_datetime(sales['date_sold'],format='%Y/%m/%d')
    

    sales = sales.reset_index()
    sales = sales.set_index('date_sold')
    
    # print(sales.index)
    y = sales

    x=y['2014-1':'2014-12']
    w=y['2015-1':'2015-12']
    z=y['2016-1':'2016-12']
    # h = sales['item_quantity_sold'].resample('MS').mean()
    
    # layout=go.Layout(title={
    #     'text': "Timestamp of item sold",
    #     'xanchor': 'center',
    #     'yanchor': 'top'},
    #     height=400,
    #     xaxis_title="Timestamp",
    #     yaxis_title="Item Quantity Sold",)
    figure = go.Figure()
    scatter = go.Scatter(x=y.index, y=y['item_quantity_sold'], mode='lines+markers')
    figure.add_trace(scatter)
    figure.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
    fig= plot(figure,output_type='div')



    # plt.figure(figsize=(14, 8))
    # plt.plot(y.index, y['item_quantity_sold'], 'b-', label = '1046')
    # plt.xlabel('Date'); plt.ylabel('Item Quantity Sold'); plt.title('Sales of 1046')
    # plt.show()


    train=pd.DataFrame({'date_sold':x.index,'item_quantity_sold':x['item_quantity_sold'],'item_quantity_before_sales':x['item_quantity_before_sales'],'item_quantity_after_sales':x['item_quantity_after_sales']})
    validation=pd.DataFrame({'date_sold':w.index,'item_quantity_sold':w['item_quantity_sold'],'item_quantity_before_sales':w['item_quantity_before_sales'],'item_quantity_after_sales':w['item_quantity_after_sales']})
    test=pd.DataFrame({'date_sold':z.index,'item_quantity_sold':z['item_quantity_sold'],'item_quantity_before_sales':z['item_quantity_before_sales'],'item_quantity_after_sales':z['item_quantity_after_sales']})
    
    X_train = train.drop(columns=['item_quantity_sold'])
    y_train = train['item_quantity_sold'].values
    X_train['date_sold']=X_train['date_sold'].map(dt.datetime.toordinal)


    X_test = test.drop(columns=['item_quantity_sold'])
    y_test = test['item_quantity_sold'].values
    X_test['date_sold']=X_test['date_sold'].map(dt.datetime.toordinal)


    X_valid = validation.drop(columns=['item_quantity_sold'])
    y_valid = validation['item_quantity_sold'].values
    X_valid['date_sold']=X_valid['date_sold'].map(dt.datetime.toordinal)



    # print('\n\nBuilding Pipeline\n\n')


    # print('Fitting the pipeline with the training data from 2014')

    model_pipeline=RandomForestRegressor(n_estimators=5000, oob_score=False, random_state=100)

    # print(y_train)
    mp=model_pipeline.fit(X_train,y_train)
    # pred_train_rf = mp.predict(X_train)
    # print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
    # print(r2_score(y_train, pred_train_rf))
    # print(pred_train_rf)
    # # predict target values on the training data

    # print('\n\nPredict target on the validation data in 2015')
    # model_pipeline.fit(X_valid,y_valid)
    pred_valid_rf = mp.predict(X_valid)
    # print(pred_valid_rf)
    # print(np.sqrt(mean_squared_error(y_valid,pred_valid_rf)))
    # print(r2_score(y_valid, pred_valid_rf))



    # print('\n\nPredict target on the test data in 2016')

    pred_test_rf=mp.predict(X_test)
    # print(pred_test_rf)
    # print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
    # print(r2_score(y_test, pred_test_rf))
    # print(y_test)
    pred=test['date_sold']+ pd.DateOffset(years=1)
    observe=test['date_sold']


    # plt.plot(observe,y_test,label='2016 data')
    # plt.plot(pred,pred_test_rf,label='predicted')
    # plt.ylabel('Item Quantity Sold')
    # plt.xlabel('Date Sold')
    # plt.show()

    pred1=test['date_sold']+ pd.DateOffset(years=1)
    observe1=test['date_sold']
    figure1 = go.Figure()
    scatter1 = go.Scatter(x=observe1 , y=y_test, mode='lines+markers')
    scatter2= go.Scatter(x=pred1, y=pred_test_rf,mode='lines+markers')
    figure1.add_trace(scatter1)
    figure1.add_trace(scatter2)
    figure1.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
    fig1= plot(figure1,output_type='div')
    context ={
        'fig':fig,
        'fig1':fig1,
        'form': form,
        'data':data,
    }
    
    return render(request, 'inv/home.html',context)



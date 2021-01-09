from django.shortcuts import render
import pandas as pd
import math
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
import datetime 
from django.contrib.auth.decorators import login_required
from inventory.models import *
# import plotly.plotly as py
from plotly.offline import plot
import plotly.graph_objects as go
# import plotly.express as px
from inventory.forms import *


@login_required
def index(request):
    form = HomeForm(request.POST)
    data = ''
    if request.method == "POST":
            if form.is_valid():
                data=form.cleaned_data.get('post')

    # datem = datetime.datetime.now().strftime ("%Y-%m-%d")
    # datem='/'.join(datem.split('-')[0:2])
    datem=datetime.datetime.now()
    year=datem.year
    month=datem.month
    last_month = datem.month-1 if datem.month > 1 else 12
    last_year = year if datem.month > 1 else year-1
    # thismonth =datetime.now().month
    solditem=pd.DataFrame.from_records(sales_record.objects.filter(date_sold__year=year,date_sold__month=month).values('item_quantity_sold','item_code'))
    item_name=pd.DataFrame.from_records(item_status.objects.select_related().values('item_code','item_name'))
    lastmonthsolditem=pd.DataFrame.from_records(sales_record.objects.filter(date_sold__year=last_year,date_sold__month=last_month).values('item_quantity_sold','item_code'))
    item=pd.DataFrame.from_records(item_status.objects.select_related().values('item_code','retail_price'))
    stockitem=pd.DataFrame.from_records(item_status.objects.select_related().values('item_code','stock_price'))
    quantity_reorder=pd.DataFrame.from_records(reorder.objects.filter(date_reorder__year=year,date_reorder__month=month).values('quantity_reorder','item_code'))
    lastmonthquantity_reorder=pd.DataFrame.from_records(reorder.objects.filter(date_reorder__year=last_year,date_reorder__month=last_month).values('quantity_reorder','item_code'))
    order_id=pd.DataFrame.from_records(reorder.objects.select_related().values('order_id','item_code'))
    Allreceive_date=pd.DataFrame.from_records(reorder.objects.select_related().values('date_of_receive','order_id'))
    receive_date=Allreceive_date[Allreceive_date['date_of_receive'].isna()]
    reorder_date=pd.DataFrame.from_records(reorder.objects.select_related().values('date_reorder','order_id'))
    Allreceive_quantity=pd.DataFrame.from_records(reorder.objects.select_related().values('quantity_receive','order_id'))
    receive_quantity=Allreceive_quantity[Allreceive_quantity['quantity_receive'].isna()]
    reorder_quantity=pd.DataFrame.from_records(reorder.objects.select_related().values('quantity_reorder','order_id'))
    
    # item=item.reset_index()
    # item=item.set_index('item_code')
    profit=0
    lastprofit=0
    cost=0
    lastcost=0
    reorderlist=[]
    # add all null receive date value to re-order yet to receive
    for ind in receive_date.index:
        receivequantity=receive_quantity.loc[receive_quantity['order_id'] == receive_date['order_id'][ind], 'quantity_receive'].iloc[0]
        code=order_id.loc[order_id['order_id'] == receive_date['order_id'][ind], 'item_code'].iloc[0]
        reorderdate=reorder_date.loc[reorder_date['order_id'] == receive_date['order_id'][ind], 'date_reorder'].iloc[0]
        reorderquantity=reorder_quantity.loc[reorder_quantity['order_id']==receive_date['order_id'][ind],'quantity_reorder'].iloc[0]
        receivedate=receive_date['date_of_receive'][ind]
        name=item_name.loc[item_name['item_code'] == code, 'item_name'].iloc[0]
        orderid=order_id['order_id'][ind]
        lacks=0
        # if receivequantity.isnull() :
        #     receivequantity='Not Receive yet'
        #     receivedate='Not Receive yet'
        lacks=-reorderquantity
        newreorderrow={'order_id':orderid,'item_code':code,'item_name':name,'reorder_date':reorderdate,'reorder_quantity':reorderquantity,'receive_date':receivedate,'receive_quantity':receivequantity,'Lack':lacks}
        reorderlist.append(newreorderrow.copy())
        # add all receive but yet not fully receive compare to reorder amount
    for ind in order_id.index:
        receivequantity=Allreceive_quantity.loc[Allreceive_quantity['order_id'] == order_id['order_id'][ind], 'quantity_receive'].iloc[0]
        code=order_id['item_code'][ind]
        names=item_name.loc[item_name['item_code'] == code, 'item_name'].iloc[0]
        reorderdate=reorder_date.loc[reorder_date['order_id'] ==  order_id['order_id'][ind], 'date_reorder'].iloc[0]
        reorderquantity=reorder_quantity.loc[reorder_quantity['order_id']== order_id['order_id'][ind],'quantity_reorder'].iloc[0]
        receivedate=Allreceive_date.loc[Allreceive_date['order_id'] == order_id['order_id'][ind], 'date_of_receive'].iloc[0]

        lacks=0
        if reorderquantity>receivequantity:
            lacks=receivequantity-reorderquantity
            orderid1=order_id['order_id'][ind]
            newreorderrow={'order_id':orderid1,'item_code':code,'item_name':names,'reorder_date':reorderdate,'reorder_quantity':reorderquantity,'receive_date':receivedate,'receive_quantity':receivequantity,'Lack':lacks}
            reorderlist.append(newreorderrow.copy())

    for ind in quantity_reorder.index:
        stock_price=stockitem.loc[stockitem['item_code'] == quantity_reorder['item_code'][ind], 'stock_price'].iloc[0]
        reorderval=quantity_reorder['quantity_reorder'][ind]
        reordervalue=reorderval*stock_price
        cost=+reordervalue
    for ind in lastmonthquantity_reorder.index:
        stock_price=stockitem.loc[stockitem['item_code'] == lastmonthquantity_reorder['item_code'][ind], 'stock_price'].iloc[0]
        reorderval=lastmonthquantity_reorder['quantity_reorder'][ind]
        reordervalue=reorderval*stock_price
        lastcost=+reordervalue
        
    for ind in solditem.index:
        retail_price=item.loc[item['item_code'] == solditem['item_code'][ind], 'retail_price'].iloc[0]
        stock_price=stockitem.loc[stockitem['item_code'] == solditem['item_code'][ind], 'stock_price'].iloc[0]
        gain=retail_price-stock_price
        sold=solditem['item_quantity_sold'][ind]
        gain=gain*sold
        profit+=gain
        
    for ind in lastmonthsolditem.index:
        if lastmonthsolditem.empty:
            continue

        else:
            lastprice=item.loc[item['item_code'] == lastmonthsolditem['item_code'][ind], 'retail_price'].iloc[0]
            laststockprice=stockitem.loc[stockitem['item_code'] == lastmonthsolditem['item_code'][ind], 'stock_price'].iloc[0]
            lastsold=lastmonthsolditem['item_quantity_sold'][ind]
            lastmonthprofit=lastprice-laststockprice
            lastgain=lastmonthprofit*lastsold
            lastprofit+=lastgain
    profit=profit-cost
    lastmonthprofit=lastmonthprofit-lastcost

    # cost_percent=(cost-lastcost)/100
    # cost_indicator=None
    # if cost_percent<0:
    #     cost_indicator=False
    # else:
    #     cost_indicator=True

    percent=(profit-lastprofit)/100
    indicator=None
    if percent<0:
        indicator=False
    else:
        indicator=True

    
    # pred_test_rf=pd.DataFrame(columns = ['prediction','date_sold'])
 
    item_code=item['item_code']
    quantity_available=pd.DataFrame.from_records(item_status.objects.select_related().values('item_code','item_quantity_available'))
    import datetime as dt
    urgent=[]
    # g=[]
    for datas in item_code:
        sales=pd.DataFrame.from_records(sales_record.objects.filter(item_code=datas).values('record_id','item_quantity_before_sales','item_quantity_sold','item_quantity_after_sales','date_sold','item_code_id'))
        cols = ['record_id','item_code_id']
        sales.drop(cols, axis=1, inplace=True)

        sales['date_sold'] = pd.to_datetime(sales['date_sold'],format='%Y/%m/%d')
        max_year=pd.DataFrame.from_records(sales_record.objects.filter(item_code=datas).values('date_sold','record_id').latest('date_sold'),index=[0])
        max_year['date_sold']=pd.to_datetime(max_year['date_sold'],format='%Y/%m/%d')
        max_year['year']=max_year['date_sold'].dt.year
        max_year=max_year.reset_index()
        sales = sales.reset_index()
        sales = sales.set_index('date_sold')

        y = sales

        z=y[str(max_year['year'][0]):]

        train=pd.DataFrame({'date_sold':y.index,'item_quantity_sold':y['item_quantity_sold'],'item_quantity_before_sales':y['item_quantity_before_sales'],'item_quantity_after_sales':y['item_quantity_after_sales']})
        # validation=pd.DataFrame({'date_sold':w.index,'item_quantity_sold':w['item_quantity_sold'],'item_quantity_before_sales':w['item_quantity_before_sales'],'item_quantity_after_sales':w['item_quantity_after_sales']})
        test=pd.DataFrame({'date_sold':z.index,'item_quantity_sold':z['item_quantity_sold'],'item_quantity_before_sales':z['item_quantity_before_sales'],'item_quantity_after_sales':z['item_quantity_after_sales']})
        
        X_train = train.drop(columns=['item_quantity_sold'])
        y_train = train['item_quantity_sold'].values
        X_train['date_sold']=X_train['date_sold'].map(dt.datetime.toordinal)


        X_test = test.drop(columns=['item_quantity_sold'])
        y_test = test['item_quantity_sold'].values
        X_test['date_sold']=X_test['date_sold'].map(dt.datetime.toordinal)


        model_pipeline=RandomForestRegressor(n_estimators=10, oob_score=False, random_state=10)

        # print(y_train)
        mp=model_pipeline.fit(X_train,y_train)
   
        
        # pred_test_rf.append(mp.predict(X_test))
        # pred_test_rf.append('month')
        a=test['date_sold'].dt.month.tolist()
        position=[]
        count=0
        for x in a:
            if x == month:
                position.append(count)
            count=count+1
        
        # pred_test_rf.append(position)
        sumval=0
        average=0
        listofdata=mp.predict(X_test)
        
        for x in position:
            sumval=sumval+listofdata[x]
            
        if len(position)!=0:
            average=(sumval/len(position))
        # g.append(average)
        if average<50:
            average=(average+50)/2
        if average>50:
            average= (average+100)/2
        average=int(average)
            

        numberleft=quantity_available.loc[quantity_available['item_code']==datas, 'item_quantity_available'].iloc[0]
        namess=item_name.loc[item_name['item_code']==datas,'item_name'].iloc[0]
        if numberleft<average:
            lack=numberleft-average
            newrow={'item_code':datas,'item_name':namess,'item_quantity_available':numberleft,'Predicted_item':int(average),'Lack':lack}
            urgent.append(newrow.copy())


    item_list=['item_code','item_name','item_quantity_available','Predicted item Required','Lack']  
    reorder_list=['order_id','item_code','item_name','reorder_date','reorder_quantity','receive_date','receive_quantity','Lack']

    context={
        'form': form,
        'data':data,
        'profit':profit,
        'percent':percent,
        'indicator':indicator,
        'urgent':urgent,
        'list':item_list,
        'reorder_list':reorder_list,
        'reorderlist':reorderlist,
        # 'cost_percent':cost_percent,
        # 'cost_indicator':cost_indicator,
        'cost':cost
    }

    return render(request, 'inv/home.html',context)

@login_required
def report(request):
    datem=datetime.datetime.now()
    year=datem.year
    month=datem.month
    last_month = datem.month-1 if datem.month > 1 else 12
    last_year = year if datem.month > 1 else year-1
    import datetime as dt
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
    max_year=pd.DataFrame.from_records(sales_record.objects.filter(item_code=data).values('date_sold','record_id').latest('date_sold'),index=[0])
    max_year['date_sold']=pd.to_datetime(max_year['date_sold'],format='%Y/%m/%d')
    max_year['year']=max_year['date_sold'].dt.year
    max_year=max_year.reset_index()
    sales = sales.reset_index()
    sales = sales.set_index('date_sold')
    
    # print(sales.index)
    y = sales
    # for x in max_year['year']:
    #     year=max_year['year']
        # if max_year['year']!=year:
        #     return year
    # w=y['2015-1':'2015-12']
    # print(max_year)
    d=max_year['year'][0]
    w=y[str(d-1)+'-1':str(d-1)+'-12']
    z=y[str(max_year['year'][0]):]
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


    train=pd.DataFrame({'date_sold':y.index,'item_quantity_sold':y['item_quantity_sold'],'item_quantity_before_sales':y['item_quantity_before_sales'],'item_quantity_after_sales':y['item_quantity_after_sales']})
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
    X_valid['date_sold']=X_valid['date_sold'].map(datetime.datetime.toordinal)



    # print('\n\nBuilding Pipeline\n\n')


    # print('Fitting the pipeline with the training data from 2014')

    model_pipeline=RandomForestRegressor(n_estimators=10, oob_score=False, random_state=10)

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
    # pred=test['date_sold']+ pd.DateOffset(years=1)
    # observe=test['date_sold']


    # plt.plot(observe,y_test,label='2016 data')
    # plt.plot(pred,pred_test_rf,label='predicted')
    # plt.ylabel('Item Quantity Sold')
    # plt.xlabel('Date Sold')
    # plt.show()
    a=test['date_sold'].dt.month.tolist()
    position=[]
    count=0
    for x in a:
        if x == month:
            position.append(count)
        count=count+1
    sumval=0
    average=0
    for x in position:
            sumval=sumval+pred_test_rf[x]
            
    if len(position)!=0:
        average=(sumval/len(position))
    
    if average<50:
        suggested=(average+50)/2
    if average>50:
        suggested= (average+100)/2
    average=int(average)
    suggested=int(suggested)

    pred1=test['date_sold']+ pd.DateOffset(years=1)
    observe1=test['date_sold']
    pred2=validation['date_sold']+ pd.DateOffset(years=1)
    figure1 = go.Figure()
    scatter1 = go.Scatter(x=observe1 , y=y_test, mode='lines+markers',name="Historical Sales")
    scatter2= go.Scatter(x=pred2, y=pred_valid_rf,mode='lines+markers',name="Historical Predicted Sales")
    scatter3=go.Scatter(x=pred1, y=pred_test_rf,mode='lines+markers',name="Predicted Sales")
    figure1.add_trace(scatter1)
    figure1.add_trace(scatter2)
    figure1.add_trace(scatter3)
    figure1.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
    fig1= plot(figure1,output_type='div')
    months=['January', 'February', 'March', 'April','May','June','July','August','September','October','November','December']
    
    context ={
        'fig':fig,
        'fig1':fig1,
        'form': form,
        'data':data,
        'month':months[month-1],
        'average':average,
        'suggested':suggested
    }
    
    return render(request, 'inv/report.html',context)



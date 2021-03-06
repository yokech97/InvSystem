from django.shortcuts import render
import pandas as pd
import math
# from sklearn.compose import ColumnTransformer 
# from sklearn.impute import SimpleImputer
from django.shortcuts import render, redirect, get_object_or_404
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from django.contrib import messages
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
    lastmonthprofit=0
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
        cost=cost+reordervalue
    for ind in lastmonthquantity_reorder.index:
        stock_price=stockitem.loc[stockitem['item_code'] == lastmonthquantity_reorder['item_code'][ind], 'stock_price'].iloc[0]
        reorderval=lastmonthquantity_reorder['quantity_reorder'][ind]
        reordervalue=reorderval*stock_price
        lastcost=lastcost+reordervalue
        
    for ind in solditem.index:
        retail_price=item.loc[item['item_code'] == solditem['item_code'][ind], 'retail_price'].iloc[0]
        stock_price=stockitem.loc[stockitem['item_code'] == solditem['item_code'][ind], 'stock_price'].iloc[0]
        gain=retail_price-stock_price
        sold=solditem['item_quantity_sold'][ind]
        gain=gain*sold
        profit=profit+gain
        
    for ind in lastmonthsolditem.index:
        if lastmonthsolditem.empty==False:
            lastprice=item.loc[item['item_code'] == lastmonthsolditem['item_code'][ind], 'retail_price'].iloc[0]
            laststockprice=stockitem.loc[stockitem['item_code'] == lastmonthsolditem['item_code'][ind], 'stock_price'].iloc[0]
            lastsold=lastmonthsolditem['item_quantity_sold'][ind]
            lastmonthprofit=lastprice-laststockprice
            lastgain=lastmonthprofit*lastsold
            lastprofit=lastprofit+lastgain


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
    overstock=[]
    for datas in item_code:
        sales=pd.DataFrame.from_records(sales_record.objects.filter(item_code=datas).values('record_id','item_quantity_before_sales','item_quantity_sold','item_quantity_after_sales','date_sold','item_code_id'))
        if sales.empty==False:
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
            suggested=0
            listofdata=mp.predict(X_test)
            
            for x in position:
                sumval=sumval+listofdata[x]
                
            if len(position)!=0:
                average=(sumval/len(position))
            # g.append(average)
            numberleft=quantity_available.loc[quantity_available['item_code']==datas, 'item_quantity_available'].iloc[0]
            if average<50:
                if average==0:
                    suggested=0
                else:
                    if 5<average<10:
                        diff=numberleft-average
                        if diff<15:
                            suggested=(average+15)
                    if 10<average<30:
                        diff=numberleft-average
                        if diff<30:
                            suggested=(average+25)
                    if 30<average<50:
                        diff=numberleft-average
                        if diff<50:
                                suggested=(average+40)
                    
            if average>50:
                diff=numberleft-average
                if diff<70:
                    suggested=(average+60)
            average=int(average)
            suggested=int(suggested)
                


            namess=item_name.loc[item_name['item_code']==datas,'item_name'].iloc[0]
            if numberleft<suggested:
                lack=numberleft-suggested
                newrow={'item_code':datas,'item_name':namess,'item_quantity_available':numberleft,'Predicted_item':suggested,'Lack':lack}
                urgent.append(newrow.copy())
            if numberleft>suggested:
                over=numberleft-suggested
                overnewrow={'item_code':datas,'item_name':namess,'item_quantity_available':numberleft,'Predicted_item':suggested,'Over':over}
                overstock.append(overnewrow.copy())



    item_list=['item_code','item_name','item_quantity_available','Predicted item Required','Lack']  
    reorder_list=['order_id','item_code','item_name','reorder_date','reorder_quantity','receive_date','receive_quantity','Lack']
    overstock_list=['item_code','item_name','item_quantity_available','Predicted_item','Over']
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
        'overstock':overstock,
        'overstock_list':overstock_list,
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

    if request.method == "POST":
            if form.is_valid():
                data=form.cleaned_data.get('post')

    quantity_available=pd.DataFrame.from_records(item_status.objects.filter(item_code=data).values('item_quantity_available'))
    quantity_left=quantity_available['item_quantity_available'][0]
    sales=pd.DataFrame.from_records(sales_record.objects.filter(item_code=data).values('record_id','item_quantity_before_sales','item_quantity_sold','item_quantity_after_sales','date_sold','item_code_id'))

    # for x in max_year['year']:
    #     year=max_year['year']
        # if max_year['year']!=year:
        #     return year
    # w=y['2015-1':'2015-12']
    # print(max_year)
    if sales.empty==True:
    
        
         return redirect('/', messages.error(request, 'Sales Data is not found', 'alert-danger'))
    else:
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
        d=max_year['year'][0]
        w=y[str(d-1)+'-1':str(d-1)+'-12']
        z=y[str(max_year['year'][0]):]
        if w.empty==True:
            train=pd.DataFrame({'date_sold':y.index,'item_quantity_sold':y['item_quantity_sold'],'item_quantity_before_sales':y['item_quantity_before_sales'],'item_quantity_after_sales':y['item_quantity_after_sales']})
            test=pd.DataFrame({'date_sold':z.index,'item_quantity_sold':z['item_quantity_sold'],'item_quantity_before_sales':z['item_quantity_before_sales'],'item_quantity_after_sales':z['item_quantity_after_sales']})
            
            X_train = train.drop(columns=['item_quantity_sold'])
            y_train = train['item_quantity_sold'].values
            X_train['date_sold']=X_train['date_sold'].map(dt.datetime.toordinal)


            X_test = test.drop(columns=['item_quantity_sold'])
            y_test = test['item_quantity_sold'].values
            X_test['date_sold']=X_test['date_sold'].map(dt.datetime.toordinal)
            model_pipeline=RandomForestRegressor(n_estimators=10, oob_score=False, random_state=10)

            mp=model_pipeline.fit(X_train,y_train)
            pred_test_rf=mp.predict(X_test)
            a=test['date_sold'].dt.month.tolist()
            j=test['date_sold'].dt.year.tolist()
            position=[]
            count=0
            bulan=[1,2,3,4,5,6,7,8,9,10,11,12]
            
            new_pred=[]
        

            for b in bulan:
                counter=0
                positions=[]
                for x in a:
                    if x==b:
                        positions.append(counter)
                    counter=counter+1
                sumofval=0
                averageval=0
                for x in positions:
                    sumofval=sumofval+pred_test_rf[x]
                if len(positions)!=0:
                    averageval=(sumofval/len(positions))
                new_pred.append(averageval)       
        

            new_test=[]
            for x in bulan:
                counter=0
                positions=[]
                for b in a:
                    if x==b:
                        positions.append(counter)
                    counter=counter+1
                sumofval=0
                averageval=0
                for x in positions:
                    sumofval=sumofval+y_test[x]
                if len(positions)!=0:
                    averageval=(sumofval/len(positions))
                new_test.append(averageval)    



            for x in a:
                if x == month:
                    position.append(count)
                count=count+1
            sumval=0
            average=0
            suggested=0
            for x in position:
                    sumval=sumval+pred_test_rf[x]
                    
            if len(position)!=0:
                average=(sumval/len(position))
            
            if average<50:
                if average==0:
                    suggested=0
                else:
                    if 5<average<10:
                        diff=quantity_left-average
                        if diff<15:
                            suggested=(average+15)
                    if 10<average<30:
                        diff=quantity_left-average
                        if diff<30:
                            suggested=(average+25)
                    if 30<average<50:
                        diff=quantity_left-average
                        if diff<50:
                                suggested=(average+40)
                   
            if average>50:
                    diff=quantity_left-average
                    if diff<70:
                        suggested=(average+60)
            average=int(average)
            suggested=int(suggested)
            months=['January', 'February', 'March', 'April','May','June','July','August','September','October','November','December']
            new_ytest=[]
            newpredtest=[]
    
            for x in months:
                new_ytest.append(x+''+str(j[0]))
                newpredtest.append(x+''+str(j[0]+1))
           

            figure1 = go.Figure()
            scatter1 = go.Scatter(x=new_ytest , y=new_test, mode='lines+markers',name="Historical Sales")
            scatter2= go.Scatter(mode='lines+markers',name="Historical Predicted Sales")
            scatter3=go.Scatter(x=newpredtest, y=new_pred,mode='lines+markers',name="Predicted Sales")
            figure1.add_trace(scatter1)
            figure1.add_trace(scatter2)
            figure1.add_trace(scatter3)
            figure1.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
            fig1= plot(figure1,output_type='div')



        else:
            
            train=pd.DataFrame({'date_sold':y.index,'item_quantity_sold':y['item_quantity_sold'],'item_quantity_before_sales':y['item_quantity_before_sales'],'item_quantity_after_sales':y['item_quantity_after_sales']})
            test=pd.DataFrame({'date_sold':z.index,'item_quantity_sold':z['item_quantity_sold'],'item_quantity_before_sales':z['item_quantity_before_sales'],'item_quantity_after_sales':z['item_quantity_after_sales']})
            
            X_train = train.drop(columns=['item_quantity_sold'])
            y_train = train['item_quantity_sold'].values
            X_train['date_sold']=X_train['date_sold'].map(dt.datetime.toordinal)


            X_test = test.drop(columns=['item_quantity_sold'])
            y_test = test['item_quantity_sold'].values
            X_test['date_sold']=X_test['date_sold'].map(dt.datetime.toordinal)
            validation=pd.DataFrame({'date_sold':w.index,'item_quantity_sold':w['item_quantity_sold'],'item_quantity_before_sales':w['item_quantity_before_sales'],'item_quantity_after_sales':w['item_quantity_after_sales']})
            X_valid = validation.drop(columns=['item_quantity_sold'])
            y_valid = validation['item_quantity_sold'].values
            X_valid['date_sold']=X_valid['date_sold'].map(datetime.datetime.toordinal)
            model_pipeline=RandomForestRegressor(n_estimators=10, oob_score=False, random_state=10)

            mp=model_pipeline.fit(X_train,y_train)
            pred_test_rf=mp.predict(X_test)
            pred_valid_rf = mp.predict(X_valid)
            a=test['date_sold'].dt.month.tolist()
            j=test['date_sold'].dt.year.tolist()
            k=validation['date_sold'].dt.month.tolist()
            s=validation['date_sold'].dt.year.tolist()
            position=[]
            count=0
            bulan=[1,2,3,4,5,6,7,8,9,10,11,12]
            
            new_pred=[]
        

            for b in bulan:
                counter=0
                positions=[]
                for x in a:
                    if x==b:
                        positions.append(counter)
                    counter=counter+1
                sumofval=0
                averageval=0
                for x in positions:
                    sumofval=sumofval+pred_test_rf[x]
                if len(positions)!=0:
                    averageval=(sumofval/len(positions))
                new_pred.append(averageval)       
        
 
            new_test=[]
            for x in bulan:
                counter=0
                positions=[]
                for b in a:
                    if x==b:
                        positions.append(counter)
                    counter=counter+1
                sumofval=0
                averageval=0
                for x in positions:
                    sumofval=sumofval+y_test[x]
                if len(positions)!=0:
                    averageval=(sumofval/len(positions))
                new_test.append(averageval)     


            new_valid=[]
            
            for x in bulan:
                counter=0
                positions=[]
                for b in k:
                    if x==b:
                        positions.append(counter)
                    counter=counter+1
                sumofval=0
                averageval=0
                for x in positions:
                    sumofval=sumofval+pred_valid_rf[x]
                if len(positions)!=0:
                    averageval=(sumofval/len(positions))
                new_valid.append(averageval)     

                
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
            months=['January', 'February', 'March', 'April','May','June','July','August','September','October','November','December']
            new_ytest=[]
            newpredtest=[]
            new_validt=[]
            for x in months:
                new_ytest.append(x+''+str(j[0]))
                newpredtest.append(x+''+str(j[0]+1))
                new_validt.append(x+''+str(s[0]+1))

            figure1 = go.Figure()
            scatter1 = go.Scatter(x=new_ytest , y=new_test, mode='lines+markers',name="Historical Sales")
            scatter2= go.Scatter(x=new_validt, y=new_valid,mode='lines+markers',name="Historical Predicted Sales")
            scatter3=go.Scatter(x=newpredtest, y=new_pred,mode='lines+markers',name="Predicted Sales")
            figure1.add_trace(scatter1)
            figure1.add_trace(scatter2)
            figure1.add_trace(scatter3)
            figure1.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
            fig1= plot(figure1,output_type='div')



        figure = go.Figure()
        scatter = go.Scatter(x=y.index, y=y['item_quantity_sold'], mode='lines+markers')
        figure.add_trace(scatter)
        figure.update_layout(title='Total Item Quantity Sold over Time',xaxis_title='Time Stamp', yaxis_title='Item Quantity Sold')
        fig= plot(figure,output_type='div')




        




    
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



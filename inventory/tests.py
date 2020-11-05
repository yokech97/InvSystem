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
import pandas as pd
import matplotlib
import datetime as dt
# read the training data set

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
df = pd.read_csv("inventory_sales_record.csv")
sales = df.loc[df['item_code_id'] == 1749]
cols = ['record_id','item_code_id']
sales.drop(cols, axis=1, inplace=True)

sales['date_sold'] = pd.to_datetime(sales['date_sold'],format='%m/%d/%Y')


sales = sales.reset_index()
sales = sales.set_index('date_sold')
print(sales)
# print(sales.index)
y = sales

x=y['2014-1':'2014-12']
w=y['2015-1':'2015-12']
z=y['2016-1':'2016-12']
print(x)
h = sales['item_quantity_sold'].resample('MS').mean()


print(h.head())
plt.figure(figsize=(14, 8))
plt.plot(y.index, y['item_quantity_sold'], 'b-', label = '1046')
plt.xlabel('Date'); plt.ylabel('Item Quantity Sold'); plt.title('Sales of 1046')
plt.legend();
plt.show()
print(h)

train=pd.DataFrame({'date_sold':x.index,'item_quantity_sold':x['item_quantity_sold'],'item_quantity_before_sales':x['item_quantity_before_sales'],'item_quantity_after_sales':x['item_quantity_after_sales']})
test=pd.DataFrame({'date_sold':z.index,'item_quantity_sold':z['item_quantity_sold'],'item_quantity_before_sales':z['item_quantity_before_sales'],'item_quantity_after_sales':z['item_quantity_after_sales']})
validation=pd.DataFrame({'date_sold':w.index,'item_quantity_sold':w['item_quantity_sold'],'item_quantity_before_sales':w['item_quantity_before_sales'],'item_quantity_after_sales':w['item_quantity_after_sales']})
print(train)
X_train = train.drop(columns=['item_quantity_sold'])
y_train = train['item_quantity_sold'].values
X_train['date_sold']=X_train['date_sold'].map(dt.datetime.toordinal)


X_test = test.drop(columns=['item_quantity_sold'])
y_test = test['item_quantity_sold'].values
X_test['date_sold']=X_test['date_sold'].map(dt.datetime.toordinal)


X_valid = validation.drop(columns=['item_quantity_sold'])
y_valid = validation['item_quantity_sold'].values
X_valid['date_sold']=X_valid['date_sold'].map(dt.datetime.toordinal)



print('\n\nBuilding Pipeline\n\n')


print('Fitting the pipeline with the training data from 2014')

model_pipeline=RandomForestRegressor(n_estimators=5000, oob_score=False, random_state=100)

print(y_train)
mp=model_pipeline.fit(X_train,y_train)
pred_train_rf = mp.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))
print(pred_train_rf)
# predict target values on the training data

print('\n\nPredict target on the validation data in 2015')
# model_pipeline.fit(X_valid,y_valid)
pred_valid_rf = mp.predict(X_valid)
print(pred_valid_rf)
print(np.sqrt(mean_squared_error(y_valid,pred_valid_rf)))
print(r2_score(y_valid, pred_valid_rf))



print('\n\nPredict target on the test data in 2016')

pred_test_rf=mp.predict(X_test)
print(pred_test_rf)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(y_test, pred_test_rf))
print(y_test)
pred=validation['date_sold']+ pd.DateOffset(years=1)
observe=test['date_sold']


plt.plot(observe,y_test,label='observed')
plt.plot(pred,pred_valid_rf,label='predicted')
plt.ylabel('Item Quantity Sold')
plt.xlabel('Date Sold')


plt.legend()
plt.show()




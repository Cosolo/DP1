import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# LO 4

data=pd.read_csv('inclusions.csv')

data.head()

data.describe()

data.dtypes

data.shape

# Visualisation

plt.scatter(data.Na,data.Type)

plt.figure(figsize=(10,10))
sns.boxplot(x=data.Type, y=data.RI)

plt.figure(figsize=(10,10))
sns.boxplot(x=data.Type, y=data.Na)

data.columns

data.Type.unique()

data.Type.value_counts().plot.bar()

We can use label encoder to transform categorical data to numerical.

data.head()

X=data.drop(columns='Type')
y=data.Type

from sklearn.preprocessing import StandardScaler

Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model. Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data

X_std=StandardScaler().fit_transform(X)

X_std=pd.DataFrame(X_std)

X_std.columns=X.columns

X_std.head()

plt.hist(X.RI)

plt.hist(X_std.RI)

# we can recognize outliers!!

for col in X_std.columns:
    hist=plt.hist(X_std[col])
    plt.title(col)
    plt.show()

# we can now easily detect outliers!

from sklearn.decomposition import PCA

pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

* How many principal components do you need so that the explained variance score in total would be greater than 80%? 

np.cumsum(pca.explained_variance_ratio_)

I need 4 prinicipal components to have variance score bigger than 80%.

sklearn_pca = PCA(n_components=4)
Y_sklearn = sklearn_pca.fit_transform(X_std)

print(Y_sklearn)

# LO 5
4. From https://finance.yahoo.com download historical data for quote ? from Jan 01, 2013 – May 9, 2018. Load dataset and using column ‘Date’ set index to Pandas DataFrame that contains your data. (Remember first to convert ‘Date’ to DatetimeIndex. Is there an easier way to set time index?) [5 points]

5. Take column Adj Close and resample it with weekly frequency from Monday using the mean value of the stock price. [5 points]

import pandas_datareader as pdr
from datetime import datetime, date

def get_stock_data (ticker, start, end):
    return pdr.get_data_yahoo(ticker,start,end)

start_date=datetime(year=2013, month=1, day=1)
end_date=datetime(year=2018, month=5, day=9)

stock=get_stock_data('NFLX', start_date, end_date)

stock.head()

stock.dtypes

stock_adjClose=apple['Adj Close']

stock_adjClose=pd.DataFrame(apple['Adj Close'])

stock_adjClose.head(2)

# first plot it
stock_adjClose['Adj Close'].plot()

data_weekly=stock['Adj Close'].resample('W-MON').mean()

data_weekly=pd.DataFrame(data_weekly)

data_weekly.head()

plt.figure(figsize=(15,10))
plt.plot(data_weekly['Adj Close'])
plt.xlabel('Time period')
plt.ylabel('Value')
plt.title('Apple price weekly on Monday')

# Learning outcome 6 – 10 points
6. Using the same data from LO5 calculate the simple daily percentage change in adjusted closing price (Adj Close), add another column to your data frame that will contain the descriptive variable “UP” or “DOWN” that will reflect positive and negative returns respectfully. [5 points]

7. Using only Adj Close column resample all values to the end of the month and forward fill any missing values. Calculate the simple monthly percentage changes and compare that number to the proportion of “UP” movements you found in the previous question. [5 points]

stock.head()

stock['Daily changes']=stock['Adj Close']/stock['Adj Close'].shift(1)-1

stock.head()

import math

stock['Daily changes']=stock['Daily changes'].fillna(0)
def func(stock):
    if stock['Daily changes'] == 0:
        return 'SAME'
    elif stock['Daily changes'] > 0:
        return 'UP'
    else:
        return 'DOWN'

stock['Daily RETURNS']=stock.apply(func, axis=1)

stock.head()

stock['Daily RETURNS'].value_counts().plot.bar()

stock_monthly=stock['Adj Close'].asfreq('M').ffill()

stock_monthly=pd.DataFrame(stock_monthly)

stock_monthly.head()

Calculate the simple monthly percentage changes and compare that number to the proportion of “UP” movements you found in the previous question. 

stock_monthly['Monthly changes']=stock_monthly/stock_monthly.shift(1)-1

stock_monthly.head()

stock_monthly['Monthly changes']=stock_monthly['Monthly changes'].fillna(0)

def func(stock_monthly):
    if stock_monthly['Monthly changes'] == 0:
        return 'SAME'
    elif stock_monthly['Monthly changes'] > 0:
        return 'UP'
    else:
        return 'DOWN'

stock_monthly.head()

stock_monthly['Monthly RETURN']=stock_monthly.apply(func, axis=1)

stock_monthly.head()

stock_monthly['Monthly RETURN'].value_counts().plot.bar()

stock['Daily RETURNS'].value_counts()

stock_monthly['Monthly RETURN'].value_counts()

print('Percentage UP on daily : ', 690/(690+655+3)*100,'%')
print('Percentage UP on monthly : ', 31/(31+20+13)*100,'%')

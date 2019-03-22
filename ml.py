import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('EURONEXT/ADYEN')
df = df[['Open','High','Low','Last','Volume']]
df['Abierto'] = df['Open']
df['MAS_alto'] = df['High']
df['MAS_bajo'] = df['Low']
df['Ultimo'] = df['Last']
df['Volumen'] = df['Volume']
df['AL_BA_pct'] = (df['MAS_alto'] - df['Ultimo']) / df['Ultimo'] *100.0
df['PCT_cambio'] = (df['Ultimo'] - df['Abierto']) / df['Abierto'] *100.0

df = df[['Ultimo','AL_BA_pct','PCT_cambio','Volumen']]

print(df.head())

forecast_col = 'Ultimo'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

print("dias de adelantados de prediccion: ",forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())

x = np.array(df.drop(['label'],1))
x = x[:-forecast_out]
x_lately = x[-forecast_out:]
x = preprocessing.scale(x)

df.dropna(inplace=True)

y = np.array(df['label'])
y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print("precision de la prediccion: ",accuracy)

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Ultimo'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()










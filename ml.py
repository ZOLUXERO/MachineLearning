import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
df.dropna(inplace=True)
print(df.head())
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(x)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print("precision de la prediccion: ",accuracy)

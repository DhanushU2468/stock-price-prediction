import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.models import load_model

model = load_model('c:/Users/dhanush/OneDrive/Desktop/stock price prediction/stock price prediction.keras')

st.header('stock market predictor')
stock=st.text_input('enter the stock symbol','GOOG')
start='2012-01-01'
end='2024-12-03'
data=yf.download(stock,start,end)
st.subheader('stock data')
st.write(data)
data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
past_100days=data_train.tail(100)
data_test=pd.concat([past_100days,data_train],ignore_index=True)
data_test_scaler=scaler.fit_transform(data_test)
x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y=np.array(x),np.array(y)

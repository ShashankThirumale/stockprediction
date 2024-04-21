# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast and Analysis App')

stocks = ("AAPL", "MSFT", "AMZN", "GOOGL", "FB", "BRK.B", "TSLA", "JNJ", "JPM", "V",
    "NVDA", "PG", "MA", "HD", "UNH", "DIS", "BABA", "PYPL", "BAC", "KO",
    "VZ", "CRM", "CMCSA", "NFLX", "T", "TMO", "MRK", "PFE", "ABT", "WMT",
    "NKE", "XOM", "CSCO", "ADBE", "ACN", "NVS", "ORCL", "CVX", "INTU", "DHR",
    "QCOM", "COST", "IBM", "CRM", "PM", "LLY", "PEP", "UNP", "ADBE", "MMM",
    "SBUX", "BA", "TXN", "GS", "USB", "AMD", "CAT", "MCD", "TMUS", "CVS",
    "AMT", "LMT", "AXP", "SPGI", "COP", "NEE", "CSX", "SO", "LIN", "MDLZ",
    "KMB", "UPS", "DOW", "ISRG", "BIIB", "LRCX", "REGN", "GM", "EQIX", "HON",
    "ABBV", "AMGN", "MO", "MMC", "GILD", "ISRG", "CI", "DUK", "VRSK", "ICE",
    "FDX", "SHW", "HUM", "MET", "PGR", "D", "CHTR", "SRE", "PLD", "FIS")
selected_stock = st.selectbox('Select dataset for prediction', stocks)


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, plot_bgcolor='white')
	st.plotly_chart(fig)
	
plot_raw_data()


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
n_years = 1
period = n_years * 365
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
st.header(f'Long term prediction using Facebook Prophet')
fig1 = plot_plotly(m, forecast)
fig1.update_layout(
    plot_bgcolor='white'
)
st.plotly_chart(fig1)

#st.write("Forecast components")
#fig2 = m.plot_components(forecast)
#st.write(fig2)

import pandas as pd
import yfinance as yf
import datetime as date

START = "2022-01-01"
TODAY = date.date.today().strftime("%Y-%m-%d")
df = yf.download(selected_stock, START, TODAY)

import matplotlib.pyplot as plt
st.write('')
st.header("Analysis of Stocks")

st.subheader('Closing Price vs Time Chart')
fig3 = plt.figure(figsize = (12, 6)) 
plt.plot(df.Close)
st.pyplot(fig3)

st.subheader('Closing Price vs Time Chart with MA100')
ma100 = df.Close.rolling(100).mean()
fig3 = plt.figure(figsize = (12, 6)) 
plt.plot(df.Close)
plt.plot(ma100)
plt.legend(["Closing price","Moving Average 100 days"])
st.pyplot(fig3)

st.subheader('Closing Price vs Time Chart with MA100 and MA200')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize = (12, 6)) 
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
plt.legend(["Closing price","Moving Average 100 days", "Moving Average 200 days"])
st.pyplot(fig3)


import pandas as pd
import yfinance as yf
import datetime as date

df = START = "2015-01-01"
TODAY = date.date.today().strftime("%Y-%m-%d")
df = yf.download(selected_stock, START, TODAY)
df = df.reset_index()

df = df[['Date', 'Close']]
import datetime

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

datetime_object = str_to_datetime('1986-03-19')
df.index = df.pop('Date')
import matplotlib.pyplot as plt

import numpy as np

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)
	
  target_date = first_date
  
  
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)
	
    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  
  for i in range(0, n):
    #X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

# Start day second time around: '2021-03-25'
windowed_df = df_to_windowed_df(df, '2016-01-25', '2024-02-23',  n=3)

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)


q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
st.write('')
st.header("Short Term Stock Prediction with LSTM Model")
st.subheader("Train, Validation, and Test Breakdown")
fig4 = plt.figure(figsize = (12, 6)) 
plt.plot(dates_train, y_train, "#0000FF")
plt.plot(dates_val, y_val, "#FFA500")
plt.plot(dates_test, y_test, "#00FF00")

plt.legend(['Train', 'Validation', 'Test'])
st.pyplot(fig4)

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

st.subheader("Prediction vs Actual")

fig5 = plt.figure(figsize = (12, 6)) 
plt.plot(dates_train, train_predictions, "#0000FF")
plt.plot(dates_train, y_train, '#FFA500')
plt.plot(dates_val, val_predictions, "#0000FF")
plt.plot(dates_val, y_val, '#FFA500')
plt.plot(dates_test, test_predictions, "#0000FF")
plt.plot(dates_test, y_test, '#FFA500')
plt.legend(['Predictions', 
            'Actual'])
st.pyplot(fig5)
prediction = y_test[-1]
write = "Approximate stock price tommorow will be: $" + str(prediction)
st.subheader(write)

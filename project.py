import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

begin = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction Application")
stocks = ("AAPL","GOOG","MSFT","GME","ZOMATO.NS")
selected_stocks = st.selectbox("Select the dataset for prediction",stocks)
years  = st.slider("Years of Prediction:",1,4)
period = years * 365
@st.cache
def load_data(ticker):
    data = yf.download(ticker,begin,today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading Data...Done!!")

st.subheader("Raw Data")

st.write(data.tail())

def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
st.write(f'Forecast plot for {years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


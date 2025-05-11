import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import requests
import pytz
import re

from prophet import Prophet

# ----------------------
# CONFIGURATION
# ----------------------
API_KEY = "brof81kr451p8hbgilp1nebb87fhr0rs98sspc5sr4ej7acdqk06p1b61b5kdqj2"  # Replace with your actual Keepa API key

# ----------------------
# Keepa API Extraction
# ----------------------
def fetch_keepa_price_history(asin: str, api_key: str, domain: int = 3) -> pd.DataFrame:
    url = f"https://api.keepa.com/product?key={api_key}&domain={domain}&asin={asin}"
    response = requests.get(url)
    data = response.json()
    parsed_data = []

    if "products" in data and data["products"]:
        product = data["products"][0]
        if "csv" in product and len(product["csv"]) > 0:
            price_history = product["csv"][0]
            keepa_epoch = datetime(2011, 5, 13, 0, 0, 0)
            utc_tz = pytz.utc
            berlin_tz = pytz.timezone('Europe/Berlin')

            for i in range(0, len(price_history), 2):
                timestamp_minutes = price_history[i]
                price_cents = price_history[i + 1]
                dt = keepa_epoch + timedelta(minutes=timestamp_minutes)
                dt_corrected = dt - timedelta(days=132)
                dt_corrected_berlin = dt_corrected.replace(tzinfo=utc_tz).astimezone(berlin_tz)
                price = price_cents / 100.0 if price_cents != -1 else None
                if price is not None:
                    parsed_data.append({"date": dt_corrected_berlin.date(), "price": price})

    return pd.DataFrame(parsed_data)

# ----------------------
# Forecasting
# ----------------------
def forecast_with_prophet(price_df: pd.DataFrame, forecast_days: int = 30) -> pd.DataFrame:
    df_daily = price_df.copy()
    df_daily = df_daily.drop_duplicates(subset='date')  # 👈 remove duplicate days
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.set_index('date').resample('D').ffill().dropna().reset_index()
    df_daily.rename(columns={'date': 'ds', 'price': 'y'}, inplace=True)
    df_daily['floor'] = 0

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df_daily)

    future = model.make_future_dataframe(periods=forecast_days)
    future['floor'] = 0
    forecast = model.predict(future)

    forecast_period = forecast.tail(forecast_days)[['ds', 'yhat']]
    forecast_period.columns = ['Date', 'Predicted_Price (€)']
    forecast_period['Predicted_Price (€)'] = forecast_period['Predicted_Price (€)'].round(2)

    return forecast_period

# ----------------------
# Promotion Calendar
# ----------------------
promotion_events = [
    {"name": "Mid January", "start": (1, 11), "end": (1, 17), "avg_drop": -23.93},
    {"name": "Early February", "start": (2, 6), "end": (2, 14), "avg_drop": -26.20},
    {"name": "Easter Sale", "start": (3, 19), "end": (3, 28), "avg_drop": -30.47},
    {"name": "Mid April", "start": (4, 15), "end": (4, 18), "avg_drop": -34.36},
    {"name": "Mothers/Fathers Day", "start": (5, 6), "end": (5, 12), "avg_drop": -34.88},
    {"name": "End May", "start": (5, 27), "end": (5, 29), "avg_drop": -25.65},
    {"name": "Prime Day July", "start": (7, 7), "end": (7, 11), "avg_drop": -41.57},
    {"name": "Back to School", "start": (8, 28), "end": (8, 31), "avg_drop": -15.46},
    {"name": "Prime Day October", "start": (10, 2), "end": (10, 9), "avg_drop": -44.26},
    {"name": "Black Friday/Cyber Monday", "start": (11, 21), "end": (11, 27), "avg_drop": -46.27},
    {"name": "Pre Christmas", "start": (12, 13), "end": (12, 22), "avg_drop": -22.58},
]

def get_upcoming_promotions(today):
    next_30 = today + timedelta(days=30)
    year = today.year
    results = []
    for event in promotion_events:
        start = datetime(year, *event['start'])
        end = datetime(year, *event['end'])
        if today <= end and start <= next_30:
            results.append({"event": event['name'], "start": start, "end": end, "expected_drop": event['avg_drop']})
    return results

# ----------------------
# Classification
# ----------------------
def classify_product_type(price_df):
    price_df = price_df.copy()
    price_df = price_df.drop_duplicates(subset='date')
    price_df = price_df.sort_values('date')
    price_df['change'] = price_df['price'].diff().abs()

    # If not enough data, default to promo
    if len(price_df) < 30:
        return "promo"

    # % of days with big changes
    big_change_days = (price_df['change'] > 1).mean()
    
    # How many unique prices are there?
    unique_price_ratio = price_df['price'].nunique() / len(price_df)

    # Heuristic rules:
    if big_change_days > 0.25 or unique_price_ratio > 0.4:
        return "dynamic"
    else:
        return "non-dynamic"

# ----------------------
# ASIN Extraction
# ----------------------
def extract_asin_from_url(url):
    match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url)
    return match.group(1) if match else None

# ----------------------
# Streamlit App
# ----------------------
st.title("📈 Amazon Price Forecasting Tool")

product_url = st.text_input("Paste full Amazon Product URL:")

if st.button("Generate Forecast"):
    asin_input = extract_asin_from_url(product_url)
    if not asin_input:
        st.error("Could not extract ASIN from URL. Please check the link.")
    else:
        st.info(f"Fetching price history for ASIN `{asin_input}` from Keepa...")
        price_df = fetch_keepa_price_history(asin_input, API_KEY)
        st.write("📊 Price history preview:", price_df.head())

        if price_df.empty:
            st.error("No price data found for this product. Check the ASIN or tracking status in Keepa.")
        else:
            st.line_chart(price_df.set_index('date')['price'])
            product_type = classify_product_type(price_df)
            st.markdown(f"**Detected Product Type:** `{product_type}`")

            if product_type == "dynamic":
                st.success("Using time series forecasting model...")
                forecast_df = forecast_with_prophet(price_df)
                st.dataframe(forecast_df)
            else:
                st.success("Using promotion calendar to detect upcoming price events...")
                today = datetime.today()
                upcoming = get_upcoming_promotions(today)
                if upcoming:
                    for event in upcoming:
                        st.markdown(f"\n**📅 {event['event']}**\n- Date: {event['start'].strftime('%d %b')} to {event['end'].strftime('%d %b')}\n- Expected Price Drop: `{event['expected_drop']}%`")
                else:
                    st.info("No upcoming promotion events in the next 30 days.")

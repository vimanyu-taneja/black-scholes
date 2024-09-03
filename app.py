from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st
import yfinance as yf
from matplotlib import pyplot as plt

from model import BlackScholes


def greek_summary(r, S, K, T, sigma):
    greeks_list = ["delta", "gamma", "vega", "theta", "rho"]
    call_greeks = [
        BlackScholes(r, S, K, T, sigma).greeks("Call")[greek] for greek in greeks_list
    ]
    put_greeks = [
        BlackScholes(r, S, K, T, sigma).greeks("Put")[greek] for greek in greeks_list
    ]
    summary = {"Call Greeks": call_greeks, "Put Greeks": put_greeks}
    df = pd.DataFrame(summary, index=["Delta", "Gamma", "Vega", "Theta", "Rho"])
    return df


def volatility_sensitivity(r, spot, strike, T, type):
    min_v = 0.01
    max_v = 0.60
    if type == "Call":
        min_s = spot * (1.0)
        max_s = spot * (1.05)
    elif type == "Put":
        min_s = spot * (0.95)
        max_s = spot * (1.0)
    volatility_values = np.linspace(min_v, max_v, 10)
    spot_values = np.linspace(min_s, max_s, 10)
    spot_values = [int(i) for i in spot_values]
    volatility_values = [round(i, 3) for i in volatility_values]
    sensitivity_data = {}
    for value in spot_values:
        column_prices = [
            round(BlackScholes(r, value, strike, T, sigma).black_scholes(type), 1)
            for sigma in volatility_values
        ]
        sensitivity_data.update({value: column_prices})

    df = pd.DataFrame(data=sensitivity_data, index=volatility_values)
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(
        ax=ax,
        data=df,
        cmap="viridis_r",
        annot=True,
        fmt="0.1f",
        annot_kws={"fontsize": 11},
    )
    ax.set_xlabel("Spot", size=14)
    ax.set_ylabel("Volatility", size=14)
    ax.set_title(type, size=16)
    return heatmap


def greek_visualisation(r, spot, strike, T, sigma, type, greek):
    fig = go.Figure()
    if type == "Call":
        line_color = "#FA7070"
        min_s = spot * (0.92)
        max_s = spot * (1.09)
    elif type == "Put":
        line_color = "#799351"
        min_s = spot * (0.92)
        max_s = spot * (1.09)
    spot_values = np.linspace(min_s, max_s, 200)

    greek_values = [
        BlackScholes(r, spot, strike, T, sigma).greeks(type)[greek]
        for spot in spot_values
    ]
    current_greek_value = BlackScholes(r, spot, strike, T, sigma).greeks(type)[greek]

    fig.add_trace(
        go.Scatter(
            x=spot_values,
            y=greek_values,
            mode="lines",
            name=f"{greek.capitalize()}",
            line=dict(color=line_color, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[spot],
            y=[current_greek_value],
            mode="markers",
            name=f"Current {greek.capitalize()}",
            marker=dict(color="black", size=7),
        )
    )

    fig.update_layout(
        title=f"{greek.capitalize()} vs Spot Price ({type})",
        xaxis_title="Spot Price",
        yaxis_title=greek.capitalize(),
    )

    return fig


def fetch_nifty():
    try:
        nifty_latest = yf.download("^NSEI", interval="1m", period="1d")
        nifty_latest = round(nifty_latest.Close.iloc[-1], 1)
        return nifty_latest
    except:
        return 25000.0


def main():
    st.set_page_config(layout="wide")

    ## Side Bar
    st.sidebar.markdown(
        "<h1 style = 'text-align: left;'>Parameters</h1>",
        unsafe_allow_html=True,
    )
    spot = st.sidebar.number_input(
        "Current stock (or other underlying) price",
        value=fetch_nifty(),
        step=5.0,
        format="%0.1f",
    )
    strike_default = (100 - ((fetch_nifty()) % 100)) + fetch_nifty()
    strike = st.sidebar.number_input(
        "Strike price", value=strike_default, step=5.0, format="%0.1f"
    )
    r = st.sidebar.number_input(
        "Risk-free interest rate (%)",
        min_value=0.00,
        max_value=100.00,
        step=0.01,
        format="%0.2f",
        value=6.731,
    )
    expiry = st.sidebar.date_input(
        "Expiry date",
        min_value=date.today(),
        value=date.today() + timedelta(days=15),
    )
    sigma = st.sidebar.number_input(
        "Volatility (%)",
        min_value=0.00,
        max_value=100.00,
        step=1.0,
        format="%0.2f",
        value=40.00,
    )
    r = r / 100
    sigma = sigma / 100

    if expiry == date.today:
        current_datetime = datetime.now()
        hours = current_datetime.hour
        minutes = current_datetime.minute
        seconds = current_datetime.second
        total_minutes = (hours * 60) + minutes + (seconds / 60)
        total_hours = total_minutes / 60

        delta = 15.5 - total_hours
        T = delta / (24 * 365)
    else:
        delta = expiry - date.today()
        T = delta.days / 365

    ## Creating an instance of BlackScholes
    Black = BlackScholes(r, spot, strike, T, sigma)
    price_call = Black.black_scholes("Call")
    price_put = Black.black_scholes("Put")

    ## Option Premiums
    prices_container = st.container(border=True)
    col1, col2 = prices_container.columns(2)
    col1.metric(label="Call Option Price", value=price_call)
    col2.metric(label="Put Option Price", value=price_put)

    ## Greek Summaries

    def color_negatives(val):
        color = "#FA7070" if (val < 0) else "#799351"
        return f"color: {color}"

    summary_combined = greek_summary(r, spot, strike, T, sigma)
    greek_container = st.container(border=True)
    greek_container.subheader("Greek Summary", divider="gray")
    greek_container.dataframe(
        summary_combined.style.map(
            color_negatives, subset=["Call Greeks", "Put Greeks"]
        ),
        use_container_width=True,
    )

    ## Sensitivity Map
    sensy_container = st.container(border=True)
    sensy_container.subheader("Sensitivity Analysis", divider="gray")
    call_sen, put_sen = sensy_container.columns(2)
    call_sen.markdown(
        '<h5 style="text-align: center;">Call Price Heatmap</h6>',
        unsafe_allow_html=True,
    )
    call_sen.pyplot(
        volatility_sensitivity(r, spot, strike, T, "Call").get_figure(),
        use_container_width=True,
    )
    put_sen.markdown(
        '<h5 style="text-align: center;">Put Price Heatmap</h6>', unsafe_allow_html=True
    )
    put_sen.pyplot(
        volatility_sensitivity(r, spot, strike, T, "Put").get_figure(),
        use_container_width=True,
    )

    ## Visualisation of Greeks
    greeks = ["delta", "gamma", "theta", "vega", "rho"]

    greek_vis = st.container(border=True)
    greek_vis.subheader("Visualisation of Greeks", divider="gray")
    call_col, put_col = greek_vis.columns(2)
    for greek in greeks:
        fig_greeks_call = greek_visualisation(r, spot, strike, T, sigma, "Call", greek)
        fig_greeks_put = greek_visualisation(r, spot, strike, T, sigma, "Put", greek)
        call_col.plotly_chart(fig_greeks_call)
        put_col.plotly_chart(fig_greeks_put)


if __name__ == "__main__":
    main()

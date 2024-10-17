import streamlit as st
import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns
import math


#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)


# (Include the BlackScholes class definition here)

class CalcOption:
    def __init__(
            self,
            time_to_maturity: float,
            strike: float,
            current_price: float,
            volatility: float,
            interest_rate: float,
    ):

        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate


    def calculate_prices(self):
        def blackscholes(price, strike, exp, rfr, vol):
            # first intermediate step
            #
            exp = exp / 365
            d1 = ((np.log(price / strike)) + (exp * (rfr + ((vol ** 2) / 2)))) / (vol * (exp ** 0.5))

            # second step
            d2 = d1 - (vol * (exp) ** 0.5)
            # [call price, put price]
            return [(price * norm.cdf(d1) - strike * np.exp(-rfr * exp) * norm.cdf(d2)),
                    (strike * math.exp(-rfr * exp) * norm.cdf(-d2) - price * norm.cdf(-d1))]

        def montecarlo(price, strike, exp, rfr, vol):
            # first simulate prices
            # init price movements over time (1000 simulations can be changed)
            exp = (int(exp))
            years = exp / 365
            timeperstep = years / exp
            numsim = 1000
            simulate = np.zeros((exp, numsim))
            simulate[0] = price

            for time in range(1, exp):
                # random values
                rand = np.random.standard_normal(numsim)
                # fill in prices at each time
                simulate[time] = simulate[time - 1] * np.exp(
                    (rfr - 0.5 * vol ** 2) * timeperstep + (vol * np.sqrt(timeperstep) * rand))

            return [np.exp(-rfr * years) * (1 / numsim) * np.sum(np.maximum(simulate[-1] - strike, 0)),
                    np.exp(-rfr * years) * (1 / numsim) * np.sum(np.maximum(strike - simulate[-1], 0))]

        def binomialput(price, strike, exp, rfr, vol):
            steps = 1000
            exp = exp / 365
            dt = exp / steps
            upfactor = np.exp(vol * np.sqrt(dt))
            downfactor = 1 / upfactor
            upprob = (np.exp(rfr * dt) - downfactor) / (upfactor - downfactor)
            tree = {}

            for m in range(steps + 1):
                currnode = price * (upfactor ** (2 * m - steps))
                tree[(steps, m)] = max(strike - currnode, 0)

            for k in range(steps - 1, -1, -1):
                for m in range(k + 1):
                    val = np.exp(-rfr * dt) * (upprob * tree[(k + 1, m + 1)] +
                                               (1 - upprob) * tree[(k + 1, m)])
                    node = price * (upfactor ** (2 * m - k))
                    tree[(k, m)] = max(val, max((strike - node), 0))

            return tree[0, 0]

        def binomialcall(price, strike, exp, rfr, vol):
            steps = 1000
            exp = exp / 365
            dt = exp / steps
            upfactor = np.exp(vol * np.sqrt(dt))
            downfactor = 1 / upfactor
            upprob = (np.exp(rfr * dt) - downfactor) / (upfactor - downfactor)
            tree = np.zeros(steps + 1)
            tree[0] = price * downfactor ** steps
            for i in range(1, steps + 1):
                tree[i] = tree[i - 1] * (upfactor / downfactor)

            optiontree = np.zeros(steps + 1)
            for i in range(0, steps + 1):
                optiontree[i] = max(0, tree[i] - strike)

            for i in range(steps, 0, -1):
                for j in range(0, i):
                    optiontree[j] = ((np.exp(-rfr * dt)) *
                                     (upprob * optiontree[j + 1] + (1 - upprob) * optiontree[j]))

            return optiontree[0]

        exp = self.time_to_maturity
        strike = self.strike
        price = self.price
        vol = self.volatility
        rfr = self.interest_rate
        bscall = round(blackscholes(price, strike, exp, rfr, vol)[0], 2)
        bsput = round(blackscholes(price, strike, exp, rfr, vol)[1], 2)
        #mccall = round(montecarlo(price, strike, exp, rfr, vol)[0], 2)
        #mcput = round(montecarlo(price, strike, exp, rfr, vol)[1], 2)
        #bccall = round(binomialcall(price, strike, exp, rfr, vol), 2)
        #bcput = round(binomialput(price, strike, exp, rfr, vol), 2)
        call = np.mean([bscall])#,mccall,bccall])
        put = np.mean([bsput])#, mcput, bcput])
        self.call = call
        self.put = put
        return call, put


# Function to generate heatmaps
# ... your existing imports and BlackScholes class definition ...


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/mprudhvi/"
    st.markdown(
        f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Prudhvi Reddy, Muppala`</a>',
        unsafe_allow_html=True)
    ticker = yf.Ticker(st.text_input("Equity Symbol", value='AAPL'))

    current_price = ticker.info.get('currentPrice')
    strike = st.number_input("Strike Price", min_value=0.0, value=current_price)
    time_to_maturity = st.number_input("Time to Maturity (Days)",min_value=1, value=7)
    volatility = st.number_input("Volatility (σ)",min_value=0.0, max_value=1.0, value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", min_value=0.0, max_value=1.0, value=0.05)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    pp = st.number_input("Purchase Price", value = 1.00)
    spot_min = current_price * 0.9
    spot_max = current_price * 1.1
    exp_min = 1.0
    exp_max = time_to_maturity

    spot_range1 = np.linspace(spot_min, current_price, 6)
    spot_range2 = np.linspace(current_price, spot_max, 7)
    spot_range = np.concatenate((spot_range1[:-1], spot_range2))
    exp_range = np.linspace(exp_min, exp_max, (time_to_maturity if time_to_maturity <= 14 else time_to_maturity // 7))



def plot_heatmap(bs_model, spot_range, exp_range, strike, pp):
    call_prices = np.zeros((len(spot_range), len(exp_range)))
    put_prices = np.zeros((len(spot_range), len(exp_range)))

    for i, spot in enumerate(spot_range[::-1]):
        for j, exp in enumerate(exp_range[::-1]):
            bs_temp = CalcOption(
                time_to_maturity=exp,
                strike=strike,
                current_price=spot,
                volatility=bs_model.volatility,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = 100 * ((bs_temp.call - pp) / pp)
            put_prices[i, j] = 100 * ((bs_temp.put - pp) / pp)

    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(exp_range[::-1], 2), yticklabels=np.round(spot_range[::-1], 2), annot=True,
                fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Days to Maturity')
    ax_call.set_ylabel('Spot Price')

    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(exp_range[::-1], 2), yticklabels=np.round(spot_range[::-1], 2), annot=True,
                fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Days to Maturity')
    ax_put.set_ylabel('Spot Price')

    return fig_call, fig_put


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Equity Symbol": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Days)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = CalcOption(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()
print(call_price, put_price)


# Display Call and Put Values in colored tables
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info(
    "Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, exp_range, strike, pp)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, exp_range, strike, pp)
    st.pyplot(heatmap_fig_put)
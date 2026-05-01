# Portfolio Optimization Dashboard

A Streamlit app for exploring portfolio optimization with live market data, risk/return analytics, and Monte Carlo projections. Enter a list of tickers, choose a date range, and the app will find the weight allocation that maximizes the Sharpe ratio while also comparing the optimized portfolio against the S&P 500.

## Features

- Optimizes portfolio weights using historical adjusted close prices from Yahoo Finance.
- Calculates annualized return, volatility, and Sharpe ratio for each asset and the optimized portfolio.
- Shows allocation breakdowns by stock and by sector.
- Plots risk/return comparisons and cumulative performance versus the S&P 500 benchmark.
- Runs a parametric Monte Carlo simulation to estimate future portfolio value paths.
- Reports terminal value statistics, probability of loss, probability of doubling, and value at risk.

## Requirements

- Python 3.9 or newer
- Internet access for Yahoo Finance data
- The Python packages listed in [requirements.txt](requirements.txt)

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the dashboard with Streamlit:

```bash
streamlit run stock.py
```

Then open the local URL shown in the terminal, usually http://localhost:8501.

## How to Use

1. Enter stock tickers separated by commas, such as AAPL,MSFT,GOOGL.
2. Choose a start and end date with at least a few months of history.
3. Set the risk-free rate and Monte Carlo assumptions.
4. Click Calculate Optimal Portfolio & Run Simulation.

## Notes

- The optimizer uses historical returns to maximize the Sharpe ratio with long-only weights that sum to 1.
- The benchmark comparison uses S&P 500 data through ^GSPC.
- Monte Carlo paths are generated from a simple parametric model, so results are best treated as directional estimates rather than forecasts.




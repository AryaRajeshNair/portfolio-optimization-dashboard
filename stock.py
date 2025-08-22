import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("Portfolio Optimization Dashboard")
st.write("""
This dashboard calculates optimal portfolio weights to maximize the Sharpe ratio using selected stocks and date ranges.
""")

<<<<<<< HEAD
#lol
#lol
=======
##
>>>>>>> origin/main##
# SIDEBAR CONTROLS

st.sidebar.header("Portfolio Inputs")

# Portfolio selection inputs
default_tickers = "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA"
ticker_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", default_tickers)
start_date = st.sidebar.date_input("Start Date", value=datetime(2018, 1, 1), max_value=datetime.today())
end_date = st.sidebar.date_input("End Date", value=datetime.today(), max_value=datetime.today())
rf = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=1.7, step=0.1) / 100

# Monte Carlo simulation parameters
st.sidebar.header("Monte Carlo Parameters")
mc_initial_investment = st.sidebar.number_input("Initial Investment ($)", 
                                             min_value=1000, 
                                             value=10000, 
                                             step=1000)
mc_time_horizon = st.sidebar.number_input("Time Horizon (Years)", 
                                       min_value=1, 
                                       max_value=30, 
                                       value=5, 
                                       step=1)
mc_num_simulations = st.sidebar.number_input("Number of Simulations", 
                                           min_value=100, 
                                           max_value=10000, 
                                           value=1000, 
                                           step=100)

calculate = st.sidebar.button("Calculate Optimal Portfolio & Run Simulation")


# MAIN APP FUNCTIONS

pd.options.display.float_format = '{:.4f}'.format
np.set_printoptions(suppress=True)

@st.cache_data
def fetch_stock_data(tickers, start, end):
    """Fetch adjusted close prices for given tickers from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def fetch_sectors(tickers):
    """Fetch sector information for each ticker."""
    sector_dict = {}
    for ticker in tickers:
        try:
            sector = yf.Ticker(ticker).info.get('sector', 'Unknown')
            sector_dict[ticker] = sector
        except:
            sector_dict[ticker] = 'Unknown'
    return sector_dict

def port_ret(weights, returns):
    """Calculate annualized portfolio return."""
    return returns.dot(weights.T).mean() * 252

def port_vol(weights, returns):
    """Calculate annualized portfolio volatility."""
    return returns.dot(weights.T).std() * np.sqrt(252)

def min_func_sharpe(weights, returns, rf):
    """Calculate negative Sharpe ratio for minimization."""
    return (rf - port_ret(weights, returns)) / port_vol(weights, returns)

def ann_risk_return(returns_df):
    """Calculate annualized return, risk, and Sharpe ratio for assets."""
    summary = returns_df.agg(["mean", "std"]).T
    summary.columns = ["Return", "Risk"]
    summary.Return = summary.Return * 252
    summary.Risk = summary.Risk * np.sqrt(252)
    summary["Sharpe Ratio"] = (summary["Return"] - rf) / summary["Risk"]
    return summary

def optimize_portfolio(returns, rf):
    """Optimize portfolio weights to maximize Sharpe ratio."""
    noa = len(returns.columns)
    eweights = np.full(noa, 1/noa)
    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(noa))
    opts = sco.minimize(min_func_sharpe, eweights, args=(returns, rf), method="SLSQP", bounds=bnds, constraints=cons)
    if not opts['success']:
        st.warning("Optimization did not converge. Results may be unreliable.")
    return opts["x"]

def monte_carlo_simulation(daily_returns, years, initial_investment, sims, trading_days=252, seed=123):
    """Run parametric Monte Carlo simulation for portfolio projection."""
    np.random.seed(seed)
    mean = daily_returns.mean() * trading_days  # Annualized mean
    std = daily_returns.std() * np.sqrt(trading_days)  # Annualized std
    
    # Generate daily returns from normal distribution
    days = int(years * trading_days)
    random_returns = np.random.normal(mean/trading_days, std/np.sqrt(trading_days), size=(sims, days))
    
    # Calculate cumulative returns
    cum_returns = (1 + random_returns).cumprod(axis=1)
    paths = initial_investment * cum_returns
    
    # Add initial investment as starting point
    paths = np.column_stack([np.ones(sims) * initial_investment, paths])
    
    return paths, mean, std


if calculate:
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    if not tickers or len(tickers) < 2:
        st.error("Please enter at least two valid stock tickers.")
    else:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        elif (end_date - start_date).days < 90:
            st.error("A date range of at least 3 months is required.")
        else:
            with st.spinner('Fetching data and running calculations...'):
                stock_data = fetch_stock_data(tickers + ['^GSPC'], start_date, end_date)
                if stock_data is not None and not stock_data.empty:
                    daily_returns = stock_data.pct_change().dropna()
                    if len(daily_returns) < 63:
                        st.error("Insufficient data. Please select a longer date range.")
                    else:
                        port_returns = daily_returns[tickers]
                        bench_returns = daily_returns['^GSPC']

                        # Optimize portfolio
                        optimal_weights = optimize_portfolio(port_returns, rf)
                        opt_ret = port_ret(optimal_weights, port_returns)
                        opt_vol = port_vol(optimal_weights, port_returns)
                        opt_sharpe = -min_func_sharpe(optimal_weights, port_returns, rf)

                        # ==============================================
                        # PORTFOLIO METRICS SECTION
                        # ==============================================
                        st.header("Optimal Portfolio Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Annual Return", f"{opt_ret * 100:.2f}%")
                        col2.metric("Annual Volatility", f"{opt_vol * 100:.2f}%")
                        col3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")

                        summary = ann_risk_return(port_returns)
                        weights_df = pd.DataFrame({
                            "Stock": tickers,
                            "Weight (%)": optimal_weights * 100
                        })
                        combined_df = weights_df.set_index("Stock").join(summary)
                        combined_df = combined_df[["Weight (%)", "Return", "Risk", "Sharpe Ratio"]]
                        combined_df["Return"] = combined_df["Return"] * 100
                        combined_df["Risk"] = combined_df["Risk"] * 100
                        combined_df = combined_df.rename(columns={
                            "Return": "Return (%)", 
                            "Risk": "Risk (%)"
                        })
                        
                        st.table(combined_df)

                        # ==============================================
                        # VISUALIZATIONS SECTION
                        # ==============================================
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_pie = px.pie(weights_df, values="Weight (%)", names="Stock", 
                                           title="Portfolio Stock Allocation")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col2:
                            sectors = fetch_sectors(tickers)
                            sector_weights = {}
                            for ticker, weight in zip(tickers, optimal_weights):
                                sector = sectors[ticker]
                                sector_weights[sector] = sector_weights.get(sector, 0) + weight * 100
                            sector_df = pd.DataFrame(list(sector_weights.items()), 
                                                   columns=["Sector", "Weight (%)"])
                            fig_sector_pie = px.pie(sector_df, values="Weight (%)", names="Sector", 
                                                  title="Portfolio Sector Allocation")
                            st.plotly_chart(fig_sector_pie, use_container_width=True)

                        # Risk/Return scatter plot
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=summary["Risk"], y=summary["Return"], mode="markers+text",
                            marker=dict(size=12, color="black", symbol="diamond"), name="Stocks",
                            text=summary.index, textposition="top center"
                        ))
                        fig_scatter.add_trace(go.Scatter(
                            x=[opt_vol], y=[opt_ret], mode="markers",
                            marker=dict(size=15, color="red", symbol="star"), name="Optimal Portfolio"
                        ))
                        fig_scatter.update_layout(
                            xaxis_title="Annual Risk (Std)", yaxis_title="Annual Return",
                            title="Risk/Return Profile", width=1000, height=600, showlegend=True
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                        # Portfolio vs. Benchmark plot
                        port_daily_returns = port_returns.dot(optimal_weights)
                        port_cum_returns = (1 + port_daily_returns).cumprod() - 1
                        bench_cum_returns = (1 + bench_returns).cumprod() - 1
                        fig_bench = go.Figure()
                        fig_bench.add_trace(go.Scatter(
                            x=port_cum_returns.index, y=port_cum_returns * 100, mode="lines",
                            name="Optimal Portfolio", line=dict(color="blue")
                        ))
                        fig_bench.add_trace(go.Scatter(
                            x=bench_cum_returns.index, y=bench_cum_returns * 100, mode="lines",
                            name="S&P 500 Benchmark", line=dict(color="green")
                        ))
                        fig_bench.update_layout(
                            xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                            title="Portfolio vs. S&P 500 Benchmark", width=1000, height=600, showlegend=True
                        )
                        st.plotly_chart(fig_bench, use_container_width=True)

                        # ==============================================
                        # MONTE CARLO SIMULATION SECTION
                        # ==============================================
                        st.header("Monte Carlo Simulation Results")
                        
                        with st.spinner('Running Monte Carlo Simulation...'):
                            paths, mean, std = monte_carlo_simulation(
                                port_daily_returns,
                                years=mc_time_horizon,
                                initial_investment=mc_initial_investment,
                                sims=mc_num_simulations
                            )
                            
                            terminal_values = paths[:, -1]
                            
                            # Plot all simulation paths
                            st.subheader(f"Projected Portfolio Paths ({mc_time_horizon} Years)")
                            fig_paths = go.Figure()
                            for i in range(min(200, mc_num_simulations)):  # Limit to 200 paths for visibility
                                fig_paths.add_trace(go.Scatter(
                                    x=np.arange(len(paths[i])),
                                    y=paths[i],
                                    mode='lines',
                                    line=dict(width=1, color='blue'),
                                    opacity=0.1,
                                    showlegend=False
                                ))
                            
                            # Add percentiles
                            percentiles = np.percentile(paths, [5, 50, 95], axis=0)
                            fig_paths.add_trace(go.Scatter(
                                x=np.arange(len(percentiles[1])),
                                y=percentiles[1],
                                mode='lines',
                                line=dict(width=2, color='black'),
                                name='Median'
                            ))
                            fig_paths.add_trace(go.Scatter(
                                x=np.arange(len(percentiles[0])),
                                y=percentiles[0],
                                mode='lines',
                                line=dict(width=1.5, color='red', dash='dash'),
                                name='5th Percentile'
                            ))
                            fig_paths.add_trace(go.Scatter(
                                x=np.arange(len(percentiles[2])),
                                y=percentiles[2],
                                mode='lines',
                                line=dict(width=1.5, color='green', dash='dash'),
                                name='95th Percentile'
                            ))
                            
                            fig_paths.update_layout(
                                xaxis_title="Trading Days",
                                yaxis_title="Portfolio Value ($)",
                                title=f"{mc_num_simulations} Simulations | Initial: ${mc_initial_investment:,}",
                                height=600
                            )
                            st.plotly_chart(fig_paths, use_container_width=True)
                            
                            # Terminal values analysis
                            st.subheader("Terminal Value Statistics")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Average Value", f"${terminal_values.mean():,.2f}")
                            col2.metric("Best Case (95th %ile)", f"${np.percentile(terminal_values, 95):,.2f}")
                            col3.metric("Worst Case (5th %ile)", f"${np.percentile(terminal_values, 5):,.2f}")
                            
                            # Histogram of terminal values
                            st.subheader("Terminal Value Distribution")
                            fig_hist = px.histogram(
                                x=terminal_values,
                                nbins=50,
                                title=f"Distribution After {mc_time_horizon} Year(s)",
                                labels={'x': 'Portfolio Value ($)'}
                            )
                            fig_hist.add_vline(
                                x=mc_initial_investment,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Initial Investment"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Risk metrics
                            st.subheader("Risk Analysis")
                            prob_loss = np.mean(terminal_values < mc_initial_investment) * 100
                            prob_double = np.mean(terminal_values > 2 * mc_initial_investment) * 100
                            var_95 = mc_initial_investment - np.percentile(terminal_values, 5)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Probability of Loss", f"{prob_loss:.1f}%")
                            col2.metric("Probability of Doubling", f"{prob_double:.1f}%")
                            col3.metric("95% Value at Risk", f"${var_95:,.2f}")
                            
                            # Annualized return statistics
                            annualized_returns = (terminal_values / mc_initial_investment) ** (1/mc_time_horizon) - 1
                            st.write(f"**Annualized Return Statistics:** Mean = {annualized_returns.mean()*100:.2f}%, "
                                    f"Std Dev = {annualized_returns.std()*100:.2f}%")
                else:
                    st.error("Failed to fetch data for the selected tickers or date range.")
else:
    st.info("Enter tickers, select parameters, and click 'Calculate Optimal Portfolio & Run Simulation' to see results.")

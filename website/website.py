import streamlit as st


# Title for the web app
st.title("Investment Portfolio Analyzer")

# Sidebar for inputs
st.sidebar.header("Investment Parameters")

# Risk tolerance slider (1-10)
risk_tolerance = st.sidebar.slider(
    "Risk Tolerance",
    min_value=1,
    max_value=10,
    value=5,
    help="1 = Very Conservative, 10 = Very Aggressive"
)

# Years invested slider
years_invested = st.sidebar.slider(
    "Investment Horizon (Years)",
    min_value=1,
    max_value=40,
    value=10
)

# Investment amount input
investment_amount = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

# Main area display
st.header("Portfolio Recommendation")
st.write(f"Based on a risk tolerance of {risk_tolerance}/10, an investment horizon of {years_invested} years, "
         f"and an investment amount of ${investment_amount:,}.")

# Placeholder for portfolio allocation
st.subheader("Recommended Asset Allocation")
# Here you would add logic to calculate allocation based on inputs
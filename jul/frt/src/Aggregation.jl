# phuonglh@gmail.com

using Statistics

"""
    aggregate(xs, width,=1 agg=sum)

    Aggregation of a series using an aggration method (summation, average, maximum, minimum).
    Default method is summation.
"""
function aggregate(xs, width=1, agg=sum)
    ys = collect(Iterators.partition(xs, width))
    map(y -> agg(y), ys)
end


function createPredictor()
end

# Forecasting models: 
# - Linear space model: ARIMA or ETS
# - Bayesian structural time series model: Prophet
# - Non-parametric time series: NPTS (Amazon)
# - DL methods: DeepAR+: train a single model jointly over the entire collection of the time series in the dataset 
#    (grouping of demand for different products, server loads, requests for web pages)

# Target: allows missing values
# Features: price, weekeday, weekend, holidays, promotion. Not allow missing values.
# CSV: (item SKU, timestamp, target values)
# CSV: (item SKU, timestamp, features)
# Categorical related data: static information: color of items, binary indicator (e.g., smart or not smart)
# Local model: one model per SKU

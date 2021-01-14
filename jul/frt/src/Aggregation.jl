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


# ORDR

# - docentry: so don hang
# - doctype: loai don hang (01: Ban tai quay; 02: Ban tra gop;...)
# - docstatus: trang thai (A: Đã duyệt, C: Hủy, D: Hoàn tất thu cọc, F: Hoàn tất, N: Chờ xử lý, O: Mở, S: Lưu, T: Đã trả hàng, W: Đang trả hàng)
# - ubus_lin: kenh ban hang
# - u_shpcod: cua hang
# - u_tmonpr: tong tien chua thue
# - u_exdate: ngay doi hang
# - u_desc: ghi chu
# - u_tmontx: tong tien thue
# - u_tmonbi: tong tien phai thu
# - type_returnso: kieu don hang
# - docdate: ngay ban

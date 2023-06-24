# Stock

    The problem is predicting the target price of a stock given its n-day history. 
    Each day has 5 features: open, high, low, close, and volume. The feature vector is normalized to have mean 0 and standard 
    deviation 1. The target value if the close price of the (n+1)-th day, which is normalized by dividing by 1000. 

    We use a sequential model consisting of a LSTM layer and a dense layer for prediction. The loss function is MSE.   

# Portfolio

    
from binance.client import Client
import pandas as pd
from models.model import variables
import numpy as np
import matplotlib.pyplot as plt

# Initialize Binance client
api_key = 'LbHBILQPXruQ1vnAXpvIiod3nmXYTlQZYTlnpLUQPkQFZ66mU0j5kwNI1mrgb6QL'
api_secret = 'v5lns9HLw5PNhNI4SqaBNcQkIVjjgzaTYadXNpMWy0wJtABoGZPPCOS2htx9uegG'
client = Client(api_key, api_secret)

# Fetch historical price data for Bitcoin (BTCUSDT)
klines = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2021")

# Convert to DataFrame and save to CSV for future use
data = pd.DataFrame(klines)
data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
data['close'] = data['close'].astype(float)
data.to_csv('data/historical_data.csv', index=False)



def predict_and_backtest(model, scaler,X):
    # Generate predictions on the training set (for backtesting)
    predicted_prices = model.predict(X)
    
    # Inverse transform to get actual price values back from normalized values
    predicted_prices = scaler.inverse_transform(predicted_prices)

  # Get actual prices for comparison (assuming X was created from historical data)
    actual_prices = scaler.inverse_transform(X[:, -1].reshape(-1, 1))  # Last price used for prediction

    # Calculate accuracy metrics
    accuracy = np.mean(np.abs((predicted_prices - actual_prices) / actual_prices) * 100)
    
    # Calculate profit/loss if a simple trading strategy was applied:
    signals = np.where(predicted_prices > actual_prices, 1, -1)  # Buy if predicted price is higher than actual
    print(signals)
    returns = np.diff(actual_prices.flatten())  # Calculate daily returns
    strategy_returns = signals[:-1] * returns  # Apply signals to returns
    # Here you can implement backtesting logic (e.g., comparing predictions with actual prices)
    total_return = np.sum(strategy_returns)  # Total return from the strategy

    print(f'Accuracy: {100 - accuracy:.2f}%')
    print(f'Total Strategy Return: {total_return:.2f}')

    # Visualization of predictions vs actual prices
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
# Example usage:
model,scaler,X = variables()
predict_and_backtest(model, scaler,X)


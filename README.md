crypto Price Prediction Project

Overview

This project aims to predict cryptocurrency prices using historical price data fetched from the Binance API. The predictions are made using an LSTM (Long Short-Term Memory) neural network, and the results include various performance metrics, such as accuracy, percentage profit/loss, and visualizations comparing actual prices with predictions.

Requirements

The following Python libraries are required to build and run this project:

numpy

pandas

matplotlib

scikit-learn

tensorflow

python-binance

Install the dependencies using the following command:

pip install numpy pandas matplotlib scikit-learn tensorflow python-binance

Steps to Build the Project

1. Integrate Binance API to Fetch Historical Price Data

In main.py, connect to the Binance API using the python-binance library.

Fetch historical cryptocurrency price data (e.g., Bitcoin, Ethereum) for a specified timeframe.

Save the fetched data locally for preprocessing and model training.

2. Data Preprocessing and Normalization

In models/model.py, preprocess the fetched data using the pandas library:

Handle missing values.

Normalize the data using MinMaxScaler from scikit-learn to ensure compatibility with the LSTM model.

3. Implement an LSTM Model

In models/model.py, build an LSTM model using the tensorflow library:

Define the architecture of the LSTM network.

Compile the model with an appropriate loss function (e.g., mean_squared_error) and optimizer (e.g., adam).

Train the model using historical price data.

4. Add Prediction and Backtesting Logic

In main.py:

Use the trained LSTM model to make predictions on test data.

Implement backtesting logic to simulate trading based on the predictions.

Calculate metrics such as accuracy, percentage profit/loss, and other relevant statistics.

5. Visualize Predictions vs Actual Prices

Use the matplotlib library to plot:

Actual cryptocurrency prices.

Predicted prices from the LSTM model.

Overlay these plots for easy comparison.

Output Metrics

The project outputs the following metrics:

Accuracy:

The percentage of predictions that are within a certain threshold of the actual prices.

Percentage Profit/Loss:

The profit or loss percentage if trades were executed based on predictions.

Profit/Loss:

Total profit or loss if trades were executed based on predictions.

Graph of Actual Prices vs Predicted Prices:

A visual comparison of actual and predicted prices using Matplotlib.

Usage Instructions

Clone the Repository:

git clone <repository_url>
cd crypto_price_prediction

Install Requirements:

pip install -r requirements.txt

Run the Project:

Fetch historical price data:

python main.py --fetch-data

Train the model and make predictions:

python main.py --train-model

Visualize results and metrics:

python main.py --visualize

File Structure

crypto_price_prediction/
├── main.py              # Main script for integration, predictions, and backtesting
├── models/
│   └── model.py         # LSTM model implementation and data preprocessing
├── data/
│   └── historical.csv   # Saved historical price data (auto-generated)
├── results/
│   └── output.png       # Visualization of actual vs predicted prices
├── requirements.txt     # List of required Python libraries
└── README.md            # Project documentation

Acknowledgments

This project uses the following libraries and tools:

Python Binance API for fetching cryptocurrency data.

TensorFlow for building the LSTM model.

Matplotlib for data visualization.
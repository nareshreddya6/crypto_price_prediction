# Crypto Price Prediction Project

This project aims to predict cryptocurrency prices using historical price data fetched from the Binance API. The predictions are made using an LSTM (Long Short-Term Memory) neural network, and the results include various performance metrics, such as accuracy, percentage profit/loss, and visualizations comparing actual prices with predictions.

## Requirements

The following Python libraries are required to build and run this project:

* numpy
* pandas
* matplotlib
* scikit-learn
* tensorflow
* python-binance

Install the dependencies using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow python-binance


## Steps to Build the Project

### 1. Integrate Binance API to Fetch Historical Price Data

1. In `main.py`, connect to the Binance API using the `python-binance` library.
2. Fetch historical cryptocurrency price data (e.g., Bitcoin, Ethereum) for a specified timeframe.
3. Save the fetched data locally as a CSV file (e.g., `historical.csv`) for preprocessing and model training.

### 2. Data Preprocessing and Normalization

1. In `models/model.py`, use the pandas library to preprocess the fetched data:
    * Handle missing values (e.g., imputation techniques).
    * Normalize the data using `MinMaxScaler` from `scikit-learn` to ensure compatibility with the LSTM model. Normalization scales the data to a specific range (often 0 to 1).

### 3. Implement an LSTM Model

1. In `models/model.py`, build an LSTM model using the `tensorflow` library:
    * Define the architecture of the LSTM network, specifying the number of layers, units per layer, and activation functions.
    * Compile the model with an appropriate loss function (e.g., `mean_squared_error` for predicting continuous values like prices) and optimizer (e.g., `adam` for efficient gradient descent).
    * Train the model using the preprocessed historical price data.

### 4. Add Prediction and Backtesting Logic

1. In `main.py`:
    * Use the trained LSTM model to make predictions on unseen test data.
    * Implement backtesting logic to simulate trading based on the predictions. This might involve comparing predicted prices with a defined threshold to determine buy or sell signals.
    * Calculate metrics such as accuracy (percentage of predictions close to actual prices), percentage profit/loss based on simulated trades, and total profit/loss.

### 5. Visualize Predictions vs Actual Prices

1. Use the `matplotlib` library to create plots that compare:
    * Actual cryptocurrency prices.
    * Predicted prices from the LSTM model.
2. Overlay these plots on the same graph for easy visual comparison of how well the predictions match the actual price movements.


Usage Instructions
Clone the Repository:

Bash

git clone <repository_url>cd crypto_price_prediction
Install Requirements:

Bash

pip install -r requirements.txt
Run the Project:
Fetch historical price data:

Bash

python main.py --fetch-data
Train the model and make predictions:

Bash

python main.py --train-model
Visualize results and metrics:

Bash

python main.py --visualize
File Structure
crypto_price_prediction/
├── main.py                # Main script for integration, predictions, and backtesting
├── models/
│   └── model.py            # LSTM model implementation and data preprocessing
├── data/
│   └── historical.csv     # Saved historical price data (auto-generated)
├── results/
│   └── output.png          # Visualization of actual vs predicted prices
├── requirements.txt       # List of required Python libraries
└── README.md              # Project documentation

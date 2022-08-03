# Dax forecasting with LSTM in TensorFlow


This project creates a forecast for a given univariate time series with TensorFlow.

To achieve this the input data needs to be:

1. loaded:
The loaded dataset will be a multiple of the window and any excess
data will be cut off in the beginning.

2. preprocessed:
During preprocessing the data will be normalized via MinMax and a testing dataset is created.

3. windowed:
During windowing of the data the time series a supervised learning problem is created and the data is split into training and validation data.
Example:
time-series = [1,2,3,4,5,6,7,8,9]
input length = 5
output length = 3

A window would be [1,2,3,4,5]->[6,7,8]

4. input into the model:
Since the first layer of the network is an LSTM-layer the input data needs to be 3-dimensions. Architecture is held simple to avoid overfitting and consists of one LSTM and two Dense-Layers.

5. visualized:
A scatterplot with Plotly. Shows the ground truth for the testing dataset as well as the prediction.

Time series data is from:
https://de.finance.yahoo.com/quote/%5EGDAXI?p=%5EGDAXI&.tsrc=fin-srch
https://de.finance.yahoo.com/quote/VOW.DE/history?p=VOW.DE

How to run?
python main.py inputlenght outputlenght learning_rate filename.csv

Example 1: python main.py 25 5 0.001 Dax.csv

Example 2: python main.py is also possible and takes default values

Selecting possible inputs:
inputlenght >= outputlength! It would not make too much sense to try and predict 10 days into the future based on the last 2 days

A large input leads to longer training times but potentially better results. A small output length leads to better results since predicting one day into the future is easier than 10. A higher learning rate leads to faster training times due to the early stopping but decreases the quality of results. Currently, only VW.csv and Dax.csv are in the inputs folder. Any other stock data from yahoo should work. The CSV file needs "Date","High" and "Low" column.


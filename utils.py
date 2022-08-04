"""Helper functions for timeseries predictions."""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import plotly.graph_objects as go


def load_data(input_len, output_len, filename):
    """Read the dataset from a csv file.

    Cut off values to make the resulting array-lengths a
    multiple of the sum of input and output_len

    Args:
        input_len (int): Mumber of timesteps looked at to make a prediction
        output_len (int): Number of timesteps to predict into the future

    Returns:
        data (np.array): Mean of the columns "High" and "Low".
        dates (np.array): Date column of the csv.
    """
    df = pd.read_csv(os.path.join("input", filename), parse_dates=["Date"])
    df["Middle"] = df[['High', 'Low']].mean(axis=1)
    df.dropna(subset=['Middle'], inplace=True)
    data = np.array(df['Middle'])
    dates = np.array(df['Date'])

    number_of_windows = int(len(data) / (input_len + output_len))
    number_of_lines = (input_len + output_len) * number_of_windows
    return data[-number_of_lines:], dates[-number_of_lines:]


def preprocess(series, input_len, output_len):
    """Split the series into train and test and standardize via MinMax.

    Returns minimum and maximum value used for standardization.

    Args:
        input_len (int): Mumber of timesteps looked at to make a prediction
        output_len (int): Number of timesteps to predict into the future

    Returns:
        train (np.array): train series
        test (np.array): test series
        train_min (float): minimal value from train series
        train_max (float): maxmal value from train series
    """
    train_series = series[0:-output_len]
    test = series[-input_len - output_len:]
    train_min = train_series.min()
    train_max = train_series.max()

    # Standardization via MinMax-scaler-formula
    train = (train_series - train_min) / (train_max - train_min)
    test = (test - train_min) / (train_max - train_min)

    return train, np.array(test), train_min, train_max


def window(dataset, input_len, output_len):
    """Create windowed data for time series prediction.

    Split into train and validation data.

    Args:
        dataset (np.array): Values to be split into windows
        input_len (int): Mumber of timesteps looked at to make a prediction
        output_len (int): Number of timesteps to predict into the future

    Returns:
        feature (np.array): features for training, shape(len_train,input_len)
        labels (np.array): labels for training, shape(len_train,output_len)
        feature_val (np.array): features for validation, shape(len_val,input_len)
        labels_val (np.array): labels for validation, shape(len_val,output_len)
    """
    feature = []
    labels = []
    feature_val = []
    labels_val = []

    end_index = len(dataset) - output_len

    for i in range(input_len, end_index):
        if np.random.random() < 0.7:
            feature.append(dataset[i - input_len:i])
            labels.append(dataset[i:i + output_len])
        else:
            feature_val.append(dataset[i - input_len:i])
            labels_val.append(dataset[i:i + output_len])
            # After appending a value to validation-list leave out the next outputlen-
            # values to avoid overfitting
            i = i + output_len

    return np.array(feature), np.array(labels), np.array(
        feature_val), np.array(labels_val)


def plot(pred, test, date, output_len):
    """Plot the results of the forecast and save to png"""
    pred = pred.reshape(output_len)
    dates_pred = date[-output_len:]

    fig = go.Figure()
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Value")
    fig.add_trace(go.Scatter(x=date, y=test,
                             mode='lines',
                             name='ground truth'))
    fig.add_trace(go.Scatter(x=dates_pred, y=test[-output_len:],
                             mode='lines',
                             name='ground truth underlying pred'))
    fig.add_trace(go.Scatter(x=dates_pred, y=pred,
                             mode='lines',
                             name='prediction'))
    fig.write_image(os.path.join("output",
                                 "forecast_for_" + str(output_len) + "_days.png"))
    fig.show()

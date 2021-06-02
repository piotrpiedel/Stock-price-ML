import keras.models

FORMAT_DATE = "%Y-%m-%d"
DATA_SOURCE = "Microsoft 5 years.csv" # Gold, Apple, Gamestop
CLOSE_COLUMN = "Close"
DATE_COLUMN = "Date"
PREDICTIONS = "Predictions"
MODEL_OUTPUT_FILE = "lstm_model.h5"
PLOT_LABEL = 'Close Price history'
UNITS = 70
DATA_RANGE = 100
SEQUENCE_LENGTH = 60

import pandas as pandas
import math
pandas.options.mode.chained_assignment = None
import numpy as numpy
import matplotlib.pyplot as pyplot
from matplotlib.pylab import rcParams as plotParameters
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

plotParameters['figure.figsize'] = 20, 10

# Load data set using pandas library data frame (two dimensional data structure)
dataFrame = pandas.read_csv(DATA_SOURCE)
dataFrame.head()


def prepareDataset():
    dataFrame[DATE_COLUMN] = pandas.to_datetime(dataFrame.Date, format=FORMAT_DATE)
    dataFrame.index = dataFrame['Date']  # x axis for plot


def displayDataset():
    pyplot.figure(figsize=(20, 10))  # set size of plot
    pyplot.plot(dataFrame[CLOSE_COLUMN], label=PLOT_LABEL)


def createRawDataFrame():
    dataset = pandas.DataFrame(index=range(0, len(dataFrame)), columns=[DATE_COLUMN, CLOSE_COLUMN])
    for i in range(0, len(data)):
        dataset[DATE_COLUMN][i] = data[DATE_COLUMN][i]
        dataset[CLOSE_COLUMN][i] = data[CLOSE_COLUMN][i]
    return dataset


def normalizeInput():
    newDataset.index = newDataset.Date
    newDataset.drop(DATE_COLUMN, axis=1, inplace=True)
    dataset = newDataset.values
    data_train = dataset[0:DATA_RANGE, :]
    valid_data = dataset[DATA_RANGE:, :]
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = min_max_scaler.fit_transform(dataset)
    return data_train, scaled_data, valid_data, min_max_scaler


def getTimeSeriesAndTargetForTraining(xTrain, yTrain, trainLength):
    for i in range(SEQUENCE_LENGTH, trainLength):
        xTrain.append(scaledData[i - SEQUENCE_LENGTH:i, 0])
        yTrain.append(scaledData[i, 0])
    return xTrain, yTrain


def prepareTimeSeriesForPrediction(input_data):
    x_test = []
    for i in range(SEQUENCE_LENGTH, input_data.shape[0]):
        x_test.append(input_data[i - SEQUENCE_LENGTH:i, 0])
    x_test = numpy.array(x_test)
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


def predictClosingPrice(model):
    predicted = model.predict(xTest)
    predicted = minMaxScaler.inverse_transform(predicted)
    return predicted


def configureDataForPredictionChart(dataset, predicted):
    train = dataset[:DATA_RANGE]
    valid = dataset[DATA_RANGE:]
    valid[PREDICTIONS] = predicted
    return train, valid


def displayPredictionChart(train, valid):
    pyplot.plot(train[CLOSE_COLUMN])
    pyplot.plot(valid[[CLOSE_COLUMN, PREDICTIONS]])


prepareDataset()
displayDataset()

# Sort data set by date and filter by “Date” & “Close” columns
data = dataFrame.sort_index(ascending=True)
newDataset = createRawDataFrame()

# Normalize data set input using MinMaxScaler to fit between zero and one
# MinMaxScaler transformation:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
trainData, scaledData, validData, minMaxScaler = normalizeInput()
xTrainData, yTrainData = [], []
getTimeSeriesAndTargetForTraining(xTrainData, yTrainData, len(trainData))
xTrainData, yTrainData = numpy.array(xTrainData), numpy.array(yTrainData)
xTrainData = numpy.reshape(xTrainData, (xTrainData.shape[0], xTrainData.shape[1], 1))

# Build LSTM model using trained model on primary data
lstmModel = keras.models.load_model(MODEL_OUTPUT_FILE)
inputData = newDataset[len(newDataset) - len(validData) - SEQUENCE_LENGTH:].values
inputData = inputData.reshape(-1, 1)
inputData = minMaxScaler.transform(inputData)

# Take a sample of a dataset to make stock price predictions using the LSTM model:
xTest = prepareTimeSeriesForPrediction(inputData)

predictedClosingPrice = predictClosingPrice(lstmModel)

# Building and displaying plot using predicted stock costs comparing to actual stock costs:
trainData, validData = configureDataForPredictionChart(newDataset, predictedClosingPrice)
displayPredictionChart(trainData, validData)

testScore = math.sqrt(mean_squared_error(validData['Close'].array, validData['Predictions'].array))
print(testScore)

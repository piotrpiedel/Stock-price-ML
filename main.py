FORMAT_DATE = "%Y-%m-%d"
DATA_SOURCE = "NSE-Tata-Global-Beverages-Limited.csv"
CLOSE_COLUMN = "Close"
DATE_COLUMN = "Date"
PREDICTIONS = "Predictions"
MODEL_OUTPUT_FILE = "saved_model.h5"
PLOT_LABEL = 'Close Price history'
UNITS = 70
DATA_RANGE = 987
SEQUENCE_LENGTH = 60

import pandas as pandas

pandas.options.mode.chained_assignment = None  # default='warn'
import numpy as numpy
import matplotlib.pyplot as pyplot
from matplotlib.pylab import rcParams as plotParameters
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

plotParameters['figure.figsize'] = 20, 10

# Read the dataset
dataFrame = pandas.read_csv(DATA_SOURCE)
dataFrame.head()


# Analyze the closing prices from dataframe
def prepareDataset():
    dataFrame[DATE_COLUMN] = pandas.to_datetime(dataFrame.Date, format=FORMAT_DATE)
    dataFrame.index = dataFrame['Date']  # x axis for plot


def displayDataset():
    pyplot.figure(figsize=(20, 10))  # size of plot
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


def prepareTimeSeriesForModel(xTrain, yTrain, trainLength):
    for i in range(SEQUENCE_LENGTH, trainLength):
        xTrain.append(scaledData[i - SEQUENCE_LENGTH:i, 0])
        yTrain.append(scaledData[i, 0])
    return xTrain, yTrain


def buildLstmModel():
    model = Sequential()
    model.add(LSTM(units=UNITS, return_sequences=True, input_shape=(xTrainData.shape[1], 1)))
    model.add(LSTM(units=UNITS))
    model.add(Dense(1))
    return model


def prepareDataForModelValidation(input_data):
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
# Sort the dataset on date time and filter “Date” and “Close” columns:
data = dataFrame.sort_index(ascending=True)
newDataset = createRawDataFrame()
# 5. Normalize the new filtered dataset:
trainData, scaledData, validData, minMaxScaler = normalizeInput()
xTrainData, yTrainData = [], []
prepareTimeSeriesForModel(xTrainData, yTrainData, len(trainData))
xTrainData, yTrainData = numpy.array(xTrainData), numpy.array(yTrainData)
xTrainData = numpy.reshape(xTrainData, (xTrainData.shape[0], xTrainData.shape[1], 1))

# 6. Build and train the LSTM model:
lstmModel = buildLstmModel()
inputData = newDataset[len(newDataset) - len(validData) - SEQUENCE_LENGTH:].values
inputData = inputData.reshape(-1, 1)
inputData = minMaxScaler.transform(inputData)
lstmModel.compile(loss='mean_squared_error', optimizer='adam')
lstmModel.fit(xTrainData, yTrainData, epochs=1, batch_size=1, verbose=2)

# 7. Take a sample of a dataset to make stock price predictions using the LSTM model:
xTest = []
prepareDataForModelValidation(inputData)

predictedClosingPrice = predictClosingPrice(lstmModel)

# 8. Save the LSTM model:
lstmModel.save(MODEL_OUTPUT_FILE)

# 9. Visualize the predicted stock costs with actual stock costs:
trainData, validData = configureDataForPredictionChart(newDataset, predictedClosingPrice)
displayPredictionChart(trainData, validData)

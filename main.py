import pandas as pandas

FORMAT_DATE = "%Y-%m-%d"
DATA_SOURCE = "NSE-Tata-Global-Beverages-Limited.csv"
CLOSE_COLUMN = "Close"
DATE_COLUMN = "Date"
PREDICTIONS = "Predictions"
UNITS = 70
MODEL_OUTPUT_FILE = "saved_model.h5"

pandas.options.mode.chained_assignment = None  # default='warn'
import numpy as numpy
import matplotlib.pyplot as pyplot
# %matplotlib inline
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Read the dataset
dataFrame = pandas.read_csv(DATA_SOURCE)
dataFrame.head()

# Analyze the closing prices from dataframe
dataFrame[DATE_COLUMN] = pandas.to_datetime(dataFrame.Date, format=FORMAT_DATE)
dataFrame.index = dataFrame['Date']  # x axis for plot
pyplot.figure(figsize=(20, 10))  # size of plot
pyplot.plot(dataFrame[CLOSE_COLUMN], label='Close Price history')

# Sort the dataset on date time and filter “Date” and “Close” columns:
data = dataFrame.sort_index(ascending=True)
newDataset = pandas.DataFrame(index=range(0, len(dataFrame)), columns=[DATE_COLUMN, CLOSE_COLUMN])
for i in range(0, len(data)):
    newDataset[DATE_COLUMN][i] = data[DATE_COLUMN][i]
    newDataset[CLOSE_COLUMN][i] = data[CLOSE_COLUMN][i]

# 5. Normalize the new filtered dataset:
newDataset.index = newDataset.Date
newDataset.drop(DATE_COLUMN, axis=1, inplace=True)  # remove  column date normalization
finalDataset = newDataset.values
trainData = finalDataset[0:987, :]
validData = finalDataset[987:, :]
minMaxScaler = MinMaxScaler(feature_range=(0, 1))
scaledData = minMaxScaler.fit_transform(finalDataset)
xTrainData, yTrainData = [], []
for i in range(60, len(trainData)):
    xTrainData.append(scaledData[i - 60:i, 0])
    yTrainData.append(scaledData[i, 0])

xTrainData, yTrainData = numpy.array(xTrainData), numpy.array(yTrainData)
xTrainData = numpy.reshape(xTrainData, (xTrainData.shape[0], xTrainData.shape[1], 1))

# 6. Build and train the LSTM model:
lstmModel = Sequential()
lstmModel.add(LSTM(units=UNITS, return_sequences=True, input_shape=(xTrainData.shape[1], 1)))
lstmModel.add(LSTM(units=UNITS))
lstmModel.add(Dense(1))

inputData = newDataset[len(newDataset) - len(validData) - 60:].values
inputData = inputData.reshape(-1, 1)
inputData = minMaxScaler.transform(inputData)
lstmModel.compile(loss='mean_squared_error', optimizer='adam')
lstmModel.fit(xTrainData, yTrainData, epochs=1, batch_size=1, verbose=2)

# 7. Take a sample of a dataset to make stock price predictions using the LSTM model:
xTest = []
for i in range(60, inputData.shape[0]):
    xTest.append(inputData[i - 60:i, 0])
xTest = numpy.array(xTest)
xTest = numpy.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
predictedClosingPrice = lstmModel.predict(xTest)
predictedClosingPrice = minMaxScaler.inverse_transform(predictedClosingPrice)

# 8. Save the LSTM model:
lstmModel.save(MODEL_OUTPUT_FILE)

# 9. Visualize the predicted stock costs with actual stock costs:
trainData = newDataset[:987]
validData = newDataset[987:]
validData[PREDICTIONS] = predictedClosingPrice
pyplot.plot(trainData[CLOSE_COLUMN])
pyplot.plot(validData[[CLOSE_COLUMN, PREDICTIONS]])

#from keras.backend import conv2d
#from google.colab import drive
# soundfile is used for reading and writing .wav files
import soundfile as sf
import multiprocessing as mp
# scipy.signal is used for calculating the time delay in samples
from numpy import float16
from scipy import signal
# sounddevice is used for recording audio
# matplotlib.pyplot is used for visualizing audio
#import matplotlib.pyplot as plt
# audio arrays are numpy arrays, and numpy has some useful functions for arrays
import numpy as np
# i use random numbers to determine delays, noise, and volume changes
import random as rand
# used for testing processing times
import time
# import wandb
import os
import pandas as pd
import math

# from wandb.keras import WandbCallback

import csv

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.utils import np_utils

DISTANCE = 51400
maxSamplesBetweenMics = 9


def GetSingleChannel(file, channelNo):
    return np.array(file[:, channelNo])


def ReadWAVFile(file):
    return sf.read(file)[0]


# read an array of file names
def ReadDatasets(files):
    # Get the total number of training and test clips from the CSV rows
    totalTrainClips = 0

    filePath = os.path.dirname(__file__)
    totalTestClips = 0
    trainFilesRead = 0
    testFilesRead = 0
    for fileName in files:
        csvTrain = filePath + fileName + " Dataset/" + fileName + "TrainingValues.csv"
        csvTest = filePath + fileName + " Dataset/" + fileName + "TestValues.csv"
        trainSheet = pd.read_csv(csvTrain)
        testSheet = pd.read_csv(csvTest)
        totalTrainClips += len(trainSheet)
        totalTestClips += len(testSheet)
    testClips = np.empty((totalTestClips, 258, 346))
    testAngles = np.empty((totalTestClips, 1))
    trainingClips = np.empty((totalTrainClips, 258, 346))
    trainingAngles = np.empty((totalTrainClips, 1))
    # read in each dataset
    for fileName in files:
        print("Starting to read dataset: ", fileName)
        csvTrain = filePath + fileName + " Dataset/" + fileName + "TrainingValues.csv"
        csvTest = filePath + fileName + " Dataset/" + fileName + "TestValues.csv"

        trainingData = pd.read_csv(csvTrain)
        testData = pd.read_csv(csvTest)
        numTestClips = len(testData)
        numTrainingClips = len(trainingData)

        clipLength = 44100 + maxSamplesBetweenMics
        print("Starting to read test clips from dataset: ", fileName)

        for number in range(1, numTestClips + 1):
            if (number % 100 == 0):
                print("Processed ", number, " train clips")
            name = filePath + fileName + " Dataset/" + fileName + "TestClip" + str(
                number) + ".wav"
            stereoArray = ReadWAVFile(name)
            ch2array, ch2segtimes, ch2fft = signal.stft(stereoArray[:, 1])
            ch1array, ch1segtimes, ch1fft = signal.stft(stereoArray[:, 0])
            # print("ch1fft: " + str(ch1fft.shape))
            # print("ch2fft: " + str(ch2fft.shape))
            stereoArray = np.vstack((ch1fft, ch2fft))
            # print("stereoArray: " + str(stereoArray.shape))

            testClips[testFilesRead - 1] = stereoArray
            testAngles[testFilesRead - 1] = testData["Angle to Sound(deg)"][number - 1]
            testFilesRead += 1

        print("Starting to read train clips from dataset: ", fileName)

        for number in range(1, numTrainingClips + 1):
            if (number % 100 == 0):
                print("Processed ", number, " train clips")
            name = filePath + fileName + " Dataset/" + fileName + "TrainingClip" + str(
                number) + ".wav"
            stereoArray = ReadWAVFile(name)
            throw, away, ch2fft = signal.stft(stereoArray[:, 1])
            throw, away, ch1fft = signal.stft(stereoArray[:, 0])
            stereoArray = np.vstack((ch1fft, ch2fft))

            trainingClips[trainFilesRead - 1] = stereoArray

            trainingAngles[trainFilesRead - 1] = trainingData["Angle to Sound(deg)"][number - 1]
            trainFilesRead += 1

    return (trainingClips, trainingAngles), (testClips, testAngles)


if __name__ == '__main__':
    #drive.mount('/content/gdrive', force_remount=True)
    # run = wandb.init()
    # config = run.config
    Datasets = ['Bird Sounds', 'CalmCity', 'Fireworks', 'Market', 'SportsRadio', 'Video Game']
    (X_train, y_train), (X_test, y_test) = ReadDatasets(Datasets)

    cols = X_train.shape[1]
    rows = X_train.shape[2]
    labels = y_train

    print("X_train: " + str(X_train.shape))
    print("X_test: " + str(X_test.shape))

    # #X_train = X_train.reshape(X_train.shape[0], cols, 1)
    # #X_test = X_test.reshape(X_test.shape[0], cols, 1)

    # #one hot encoding, transforms the list of values into a binary matrix of 1s and 0s
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Get the number of different "buckets"
    num_classes = y_train.shape[1]

    model = Sequential()
    # #investigate passing as two channels and not flattening
    # model.add(Conv1D(3, (16), input_shape=(258, 1724), activation='relu'))
    # model.add(MaxPooling1D(pool_size=(16)))
    # model.add(Dropout(0.4))
    # model.add(Conv1D(3, (8), input_shape=(258, 1724), activation='relu'))
    # model.add(MaxPooling1D(pool_size=(8)))
    # model.add(Dropout(0.4))
    model.add(Conv1D(3, (2), input_shape=(258, 346), activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.4))
    # # model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    # # model.add(Dense(num_classes, activation='relu'))
    model.add(Flatten(input_shape=(cols, rows)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # sparsecategoricalcrossentropy(from logits=true)   - one hot encoding
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    # , callbacks=[WandbCallback()]
    model.summary()
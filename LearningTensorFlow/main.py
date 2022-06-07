# #soundfile is used for reading and writing .wav files
# import soundfile as sf
# import multiprocessing as mp
# #scipy.signal is used for calculating the time delay in samples
# from numpy import float16
# from scipy import signal
# #sounddevice is used for recording audio
# #import sounddevice as sd
# #matplotlib.pyplot is used for visualizing audio
# import matplotlib.pyplot as plt
# #audio arrays are numpy arrays, and numpy has some useful functions for arrays
# import numpy as np
# #i use random numbers to determine delays, noise, and volume changes
# import random as rand
# #used for testing processing times
# import time
# import wandb
# import os
#
# import pandas as pd
# import math
#
# # import wandb
# # from wandb.keras import WandbCallback
#
# import csv
#
# # import tensorflow as tf
# # from keras.models import Sequential
# # from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
# # from keras.utils import np_utils
#
# #debug flags
# GENERATE_FILES = not True
# GENERATE_GRAPHS = not True
# DISTANCE = 51400
# maxSamplesBetweenMics = 9
# positiveSamples = maxSamplesBetweenMics // 2 #floor division on max samples. there will be +- positiveSamples from the 90 degree mark
#
# def RecordAudioClip(frames, which_device, channel):
#     print("Recording Audio\n")
#     data = sd.rec(frames, device=which_device, channels=channel, samplerate=44100)
#     sd.wait()
#     print("Finished Recording Audio\n")
#     return data
#
# def GetSingleChannel(file, channelNo):
#     return np.array(file[:, channelNo])
#
# def GenerateDataset(file, clipSize = 44100):
#
#     print("Starting Creation Of Dataset For ", file)
#
#     dirName = file + ' Dataset'
#     os.mkdir(dirName, mode=700)
#     #csvTest = file + 'TestValues.csv'
#     #csvTrain = file + 'TrainingValues.csv'
#     csvTest = os.path.dirname(__file__) + '/' + dirName + '/' + file + 'TestValues.csv'
#     csvTrain = os.path.dirname(__file__) + '/' + dirName + '/' + file + 'TrainingValues.csv'
#
#     filename = os.path.dirname(__file__) + '/Audio/' + file + '.wav'
#     header = ['Clip #', 'Clip Delay(frames)', 'Angle to Sound(deg)', 'Total Frames']
#
#     with open(csvTest, 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#     with open(csvTrain, 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
#     sampleRate = 44100
#     clipLength = clipSize
#
#     AudioFile = ReadWAVFile(filename)
#     AudioCh1 = GetSingleChannel(AudioFile, 0)
#     AudioCh2 = GetSingleChannel(AudioFile, 1)
#
#
#     NumSamples = AudioCh2.size
#     numRecordings = math.ceil(NumSamples / clipLength)
#
#
#     # duplicate result into two arrays
#     # write the data
#     testCount = 1
#     trainingCount = 1
#     for recording in range(1, numRecordings):
#
#         regClip = np.copy(AudioCh1[:clipLength])
#         delClip = np.copy(AudioCh2[:clipLength])
#         AudioCh1 = np.copy(AudioCh1[clipLength:])
#         AudioCh2 = np.copy(AudioCh2[clipLength:])
#         newSize = regClip.size + maxSamplesBetweenMics
#
#         #np.resize(regClip, newSize)
#         #np.resize(delClip, newSize)
#         regClip.resize(newSize)
#         delClip.resize(newSize)
#         # calculate an offset and new size to shift one array
#         offset = int(rand.uniform(-4, 4))
#
#         # resize both arrays and roll the delayed one by the offset
#         delClip = np.roll(delClip, offset)
#
#         stereoArray = np.vstack((regClip, delClip)).T
#
#         if GENERATE_GRAPHS and recording == 1:
#             plt.title("Regular Channel Should Look Like This")
#             plt.plot(stereoArray[:, 0], color="red")
#             #
#             plt.show()
#
#             plt.title("Delayed Channel Should Look Like This")
#             plt.plot(stereoArray[:, 1], color="red")
#             #
#             plt.show()
#
#         rad = math.acos((offset / sampleRate) / (positiveSamples/sampleRate))
#         angle = (rad * 180) / math.pi
#
#         #Validation so I can stop the program and diagnose if there are issues
#         if (offset == 4 and angle != 0):
#             print("Bad Angle Calculations... Expected 0 but got ", angle)
#
#         if(offset == -4 and angle != 180):
#             print("Bad Angle Calculations... Expected 180 but got ", angle)
#
#         if (recording % 5 != 0):
#             fName = os.path.dirname(__file__) + '/' + dirName + '/' + file + "TrainingClip" + str(trainingCount) + ".wav"
#             sf.write(fName, stereoArray, sampleRate)
#
#             data = [str(trainingCount), str(offset) , str(int(angle)) ,  AudioCh2.size]
#
#             trainingCount += 1
#
#             with open(csvTrain, 'a', encoding='UTF8', newline='') as f:
#                 writer = csv.writer(f)
#
#                 # write the data
#                 writer.writerow(data)
#         else:
#             fName = os.path.dirname(__file__) + '/' + dirName + '/' + file + "TestClip" + str(testCount) + ".wav"
#             #fName = file + "TestClip" + str(testCount) + ".wav"
#             sf.write(fName, stereoArray, sampleRate)
#
#             data = [str(testCount), str(offset), str(int(angle)), AudioCh2.size]
#             testCount += 1
#
#             with open(csvTest, 'a', encoding='UTF8', newline='') as f:
#                 writer = csv.writer(f)
#
#                 # write the data
#                 writer.writerow(data)
#     print("Done Creating Dataset For ", file)
#         ####################################################################
#         #This part of the function is for testing that I can read files in correctly
#         ##when reading in an array, you get back a 2 object array where
#         #index 0 is a 2 column array of the channels, and index 1 is the sample rate
#         #But we normally just want the channels
#         # readArray = sf.read('DelayTesting.wav')[0]
#         # print(readArray)
#
# def ReadWAVFile(file):
#     return sf.read(file)[0]
#
# def WriteWAVFile(filename, Data):
#     sf.write(filename, data=Data, samplerate=44100)
# def ReadTrainingSet():
#     numTestClips = 31
#     numTrainingClips = 127
#     clipLength = 44100 * 5 + maxSamplesBetweenMics
#
#     testClips = np.empty((numTestClips, clipLength, 2))
#     testAngles = np.empty((numTestClips, 1))
#     trainingClips = np.empty((numTrainingClips, clipLength, 2))
#     trainingAngles = np.empty((numTrainingClips, 1))
#
#     trainingData = pd.read_csv("BirdTrainingValues.csv")
#     testData = pd.read_csv("BirdTestValues.csv")
#
#     for number in range(1, numTestClips + 1):
#         name = "BirdTestClip" + str(number) + ".wav"
#         testClips[number - 1] = ReadWAVFile(name)
#         testAngles[number - 1] = testData["Angle to Sound(deg)"][number - 1]
#
#     for number in range(1, numTrainingClips + 1):
#         name = "BirdTrainingClip" + str(number) + ".wav"
#         trainingClips[number - 1] = ReadWAVFile(name)
#         trainingAngles[number - 1] = trainingData["Angle to Sound(deg)"][number - 1]
#
#     return (trainingClips, trainingAngles), (testClips, testAngles)
#
# if __name__ == '__main__':
#
#     files = ['Bird Sounds','CalmCity', 'Fireworks', 'Market', 'SportsRadio', 'Video Game']
#
#     # uncomment these two lines if you are generating wav files
#     # fName = "BirdTrainingClip" + str(trainingCount) + ".wav"
#     # sf.write(fName, stereoArray, sampleRate)
#     #
#     # csvName = os.path.dirname(__file__) + '/Bird Sounds Dataset/Bird SoundsTrainingClip1.wav'
#     # arr = ReadWAVFile(csvName)
#     # print(np.shape(arr))
#     for file in files:
#         GenerateDataset(file)
#     #GenerateTrainingSet()
#
#     #wandb.init()
#     # VG = ReadWAVFile('Video Game.wav')
#     # Bird = ReadWAVFile('Bird Sounds.wav')
#     #
#     # print(VG.shape)
#     # print(Bird.shape)
#     #
#     #
#     # plt.title("Bird channel 1")
#     # plt.plot(VG[:, 0])
#     # plt.show()
#     #
#     # plt.title("Bird channel 2")
#     # plt.plot(VG[:, 1])
#     # plt.show()
#     #
#     # WriteWAVFile("Bird channel 1.wav", Bird[:, 0])
#     # WriteWAVFile("Bird channel 2.wav", Bird[:, 1])
#     #
#     # fftCorrelation = signal.correlate(Bird[:, 0] - np.mean(Bird[:, 0]), Bird[:, 1] - np.mean(Bird[:, 1]), method='fft', mode="full")
#     # lags = signal.correlation_lags(len(Bird[:, 0]), len(Bird[:, 1]), mode="full")
#     # lag = lags[np.argmax(fftCorrelation)]
#     #
#     # print("Delay = " + str(lag))
#     #
#     # array1 = np.array(Bird[:, 0])
#     #
#     # array1.resize(array1.size + 2)
#     #
#     # array1 = np.roll(array1, 2)
#     #
#     # array2 = np.array(Bird[:, 1])
#     #
#     # array2.resize(array2.size + 2)
#     #
#     # fftCorrelation2 = signal.correlate(array1 - np.mean(array1), array2 - np.mean(array2), method='fft', mode="full")
#     # lags2 = signal.correlation_lags(len(array1), len(array2), mode="full")
#     # lag2 = lags2[np.argmax(fftCorrelation2)]
#     #
#     # print("Delay after roll = " + str(lag2))
#     #
#     # stereoArray = np.vstack((array1, array2)).T
#     #
#     # WriteWAVFile("BirdSoundsForTesting.wav", stereoArray)
#
#     # #couldnt get wandb to work but it seems like a really cool tool for metrics
#     # run = wandb.init()
#     # config = run.config
#     #
#     #
#     # setSize = 50
#     #
#     # if GENERATE_FILES:
#     #     seconds = 5
#     #     sampleRate = 44100
#     #     device = 1
#     #     channels = 2
#     #     GenerateTrainingSet(seconds, sampleRate, device, channels, setSize)
#     # run = wandb.init()
#     # config = run.config
#     #
#     # (X_train, y_train), (X_test, y_test) = ReadTrainingSet()
#     # cols = X_train.shape[1]
#     # rows = X_train.shape[2]
#     # labels = y_train
#     #
#     # X_train = X_train.reshape(X_train.shape[0], cols, rows, 1)
#     # X_test = X_test.reshape(X_test.shape[0], cols, rows, 1)
#     #
#     # # one hot encoding, transforms the list of values into a binary matrix of 1s and 0sq
#     # y_train = np_utils.to_categorical(y_train)
#     # y_test = np_utils.to_categorical(y_test)
#     #
#     # # Get the number of different "buckets"
#     # num_classes = y_train.shape[1]
#     #
#     # model = Sequential()
#     # # investigate passing as two channels and not flattening
#     # model.add(Conv2D(3, (2, 2), input_shape=(cols, rows, 1), activation='relu'))
#     # model.add(MaxPooling2D(pool_size=(1, 1)))
#     # model.add(Dropout(0.4))
#     # model.add(Dense(10, activation='relu'))
#     # model.add(Dropout(0.4))
#     # # model.add(Dense(num_classes, activation='relu'))
#     # # model.add(Dense(num_classes, activation='relu'))
#     # model.add(Dense(num_classes, activation='softmax'))
#     # model.add(Flatten(input_shape=(cols, rows)))
#     # model.add(Dense(num_classes, activation='softmax'))
#     #
#     # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # # sparsecategoricalcrossentropy(from logits=true)   - one hot encoding
#     # model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), callbacks=[WandbCallback()])
#     #
#     # model.summary()
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

    filePath = os.path.dirname(__file__) + '/'
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
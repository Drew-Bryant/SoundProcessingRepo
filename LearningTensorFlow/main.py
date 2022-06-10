#    This file is used as a scratch pad for generating audio datasets and training machine learning models
#    using tensorflow.
#
#   Author: Drew Bryant
#   Email: drew.bryant@oit.edu
#
#
#   Functions (more detail available in the function headers):
#   GenerateDataset(file, clipSize = 44100)
#       Parameters:
#           file - the name of a .wav file to generate a dataset from, include '.wav' in the argument
#           clipSize - length of each clip in the dataset. Defaults to 44100
#                      The number of total clips in the dataset is the size of the .wav file in samples divided by clipSize
#       Returns:
#           None, generates a "<file> Dataset" folder in the current directory
#
#   ReadDatasets(files)
#       Parameters:
#           files - array with the names of all .wav files that have dataset folders generated in the current directory
#                   DO NOT include '.wav' in the names passed in this array
#       Returns:
#           None, generates a "<file> Dataset" folder in the current directory
#
#   The remaining functions are one-line helper functions for code readability and conciseness and will not be listed here
#   See their function headers for more information

#imports and why they are there-----------------------------------------------
#soundfile is used for reading and writing .wav files
import soundfile as sf
import multiprocessing as mp
#scipy.signal is used for calculating the time delay in samples
import scipy
from numpy import float16
from scipy import signal
#sounddevice is used for recording audio
#import sounddevice as sd
#matplotlib.pyplot is used for visualizing audio
import matplotlib.pyplot as plt
#audio arrays are numpy arrays, and numpy has some useful functions for arrays
import numpy as np
#i use random numbers to determine delays, noise, and volume changes
import random as rand
#used for testing processing times
import time
#import wandb
import os
import pandas as pd
import math



import csv

#tensoflow for machine learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils


#Values that determine data about mic separation
sampleRate = 44100
speedOfSound = 343   # units: meters/second
micSeparation = 0.07  #units: meters
maxSamplesBetweenMics = int((micSeparation / speedOfSound) * sampleRate)  #total number of samples that can fit between the mics (
positiveSamples = maxSamplesBetweenMics // 2 #floor division on max samples. there will be +- positiveSamples from the 90 degree mark



#Function: GenerateDatasets(file, clipSize = 44100)
#
#Generate a dataset of .wav files from a single, 2 channel .wav file in the same directory as this file
# Generated files will be placed in a sub-folder called "<file> Dataset"
# training set will be built using actual stereo data, not duplicated mono files
# delays generated are random, based on the global values above this function
# IMPORTANT: if you generate a dataset with this function, you MUST guarantee that all possible delay offsets
#            happened at least once by checking the .csv outputs, otherwise the dataset will have problems
#            This should not be a problem if you have an input file that will generate ~500 training files
# 4 training clips are generated for every 1 test clip
#Generates "<file>TestValues.csv" and "<file>TrainingValues.csv" files
#These .csv files contain clip #, clip delay in frames, Angle to sound in degrees and total clip length
# Parameters:
#   file - string representing a 2 channel .wav file in the same directory as main.py
#   clipSize - length of each clip in the dataset in samples
#              The number of total clips in the dataset is the size of the .wav file in samples divided by clipSize
#
# Return: None
#
# Precondition:
#   a .wav file in the same directory as this file
#
# Postcondition:
#   A folder will be generated in the same directory as this file called "<file> Dataset"
#   The folder will contain a ratio of 4 training clips to 1 test clip
#   The folder will contain two .csv files with correct values for training and test clips
def GenerateDataset(file, clipSize = 44100):

    print("Starting Creation Of Dataset For ", file)

    dirName = file + ' Dataset'
    os.mkdir(dirName, mode=700)
    #csvTest = file + 'TestValues.csv'
    #csvTrain = file + 'TrainingValues.csv'
    csvTest = os.path.dirname(__file__) + '/' + dirName + '/' + file + 'TestValues.csv'
    csvTrain = os.path.dirname(__file__) + '/' + dirName + '/' + file + 'TrainingValues.csv'

    filename = os.path.dirname(__file__) + '/Audio/' + file + '.wav'
    header = ['Clip #', 'Clip Delay(frames)', 'Angle to Sound(deg)', 'Total Frames']

    with open(csvTest, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    with open(csvTrain, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    clipLength = clipSize

    AudioFile = ReadWAVFile(filename)
    AudioCh1 = GetSingleChannel(AudioFile, 0)
    AudioCh2 = GetSingleChannel(AudioFile, 1)


    NumSamples = AudioCh2.size
    numRecordings = math.ceil(NumSamples / clipLength)


    # duplicate result into two arrays
    # write the data
    testCount = 1
    trainingCount = 1
    for recording in range(1, numRecordings):

        regClip = np.copy(AudioCh1[:clipLength])
        delClip = np.copy(AudioCh2[:clipLength])
        AudioCh1 = np.copy(AudioCh1[clipLength:])
        AudioCh2 = np.copy(AudioCh2[clipLength:])
        newSize = regClip.size + maxSamplesBetweenMics

        #np.resize(regClip, newSize)
        #np.resize(delClip, newSize)
        regClip.resize(newSize)
        delClip.resize(newSize)
        # calculate an offset and new size to shift one array
        offset = int(rand.uniform(-4, 4))

        # resize both arrays and roll the delayed one by the offset
        delClip = np.roll(delClip, offset)

        stereoArray = np.vstack((regClip, delClip)).T

        if GENERATE_GRAPHS and recording == 1:
            plt.title("Regular Channel Should Look Like This")
            plt.plot(stereoArray[:, 0], color="red")
            #
            plt.show()

            plt.title("Delayed Channel Should Look Like This")
            plt.plot(stereoArray[:, 1], color="red")
            #
            plt.show()

        rad = math.acos((offset / sampleRate) / (positiveSamples/sampleRate))
        angle = (rad * 180) / math.pi

        #Validation so I can stop the program and diagnose if there are issues
        if (offset == 4 and angle != 0):
            print("Bad Angle Calculations... Expected 0 but got ", angle)

        if(offset == -4 and angle != 180):
            print("Bad Angle Calculations... Expected 180 but got ", angle)

        if (recording % 5 != 0):
            fName = os.path.dirname(__file__) + '/' + dirName + '/' + file + "TrainingClip" + str(trainingCount) + ".wav"
            sf.write(fName, stereoArray, sampleRate)

            data = [str(trainingCount), str(offset) , str(int(angle)) ,  AudioCh2.size]

            trainingCount += 1

            with open(csvTrain, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)

                # write the data
                writer.writerow(data)
        else:
            fName = os.path.dirname(__file__) + '/' + dirName + '/' + file + "TestClip" + str(testCount) + ".wav"
            sf.write(fName, stereoArray, sampleRate)

            data = [str(testCount), str(offset), str(int(angle)), AudioCh2.size]
            testCount += 1

            with open(csvTest, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)

                # write the data
                writer.writerow(data)
    print("Done Creating Dataset For ", file)


# Function: ReadDatasets(files)
#
# Read 1 or more datasets generated with the GenerateDataset() function at the top of this file
# Parameters:
#   files - list of file names that have dataset folders in the current directory
#           File names should be the name of the .wav file that was passed to GenerateDataset()
#           but filenames passed to this function should NOT include '.wav'
# Return:
#       returns 4 arrays in two tuples.
#       The first tuple is training clip FFTs and correct angles.
#       The second tuple is test clip FFTs and correct angles.
#       dimensions are based on size of dataset and length of clips
#
#   Precondition:
#       A .wav file in the current directory has been processed with the GenerateDataset() function at the top of this file
#       and there is a corresponding "<file> Dataset" folder in the current directory
#   Postcondition:
#       4 arrays are returned containing testing and training data for use with tensorflow models
def ReadDatasets(files):
    # Get the total number of training and test clips from the CSV rows
    totalTrainClips = 0

    filePath = os.path.dirname(__file__) + '/'
    totalTestClips = 0
    trainFilesRead = 0
    testFilesRead = 0

    #Get the total number of training and test clips from the .csv files
    for fileName in files:
        csvTrain = filePath + fileName + " Dataset/" + fileName + "TrainingValues.csv"
        csvTest = filePath + fileName + " Dataset/" + fileName + "TestValues.csv"
        trainSheet = pd.read_csv(csvTrain)
        testSheet = pd.read_csv(csvTest)
        totalTrainClips += len(trainSheet)
        totalTestClips += len(testSheet)

    #read the first test clip, convert it to an FFT to get the shape of an FFT
    #for this clip size
    name = filePath + fileName + " Dataset/" + fileName + "TestClip1.wav"

    stereoArray = ReadWAVFile(name)
    FFTdim1 = len((signal.stft(stereoArray[:, 1]))[0])
    FFTdim2 = len((signal.stft(stereoArray[:, 1]))[1])
    testClips = np.empty((totalTestClips, FFTdim1 * 2, FFTdim2))
    testAngles = np.empty((totalTestClips, 1))
    trainingClips = np.empty((totalTrainClips, FFTdim1 * 2, FFTdim2))
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
        print("Finished Processing dataset: ", fileName, " --- ", testFilesRead, "test clips and ", trainFilesRead, " training clips were read")

    return (trainingClips, trainingAngles), (testClips, testAngles)

#Write a WAV file
#Parameters:
#   filename - file to write, must include '.wav'
#   Data - array-like, must be compatible with soundfile.write
def WriteWAVFile(filename, Data):
    sf.write(filename, data=Data, samplerate=sampleRate)


#Return one channel of a multi channel array generated by ReadWAVFile()
#This function assumes each column of the array is a channel, and each row is a sample
#Parameters:
#   file - array currently holding N channels worth of audio data
#   channelNo - channel to be returned, indexed from 0
def GetSingleChannel(file, channelNo):
    return np.array(file[:, channelNo])

#Return the audio data read by soundfile.read()
#Parameters:
#   file - name of the file to be read, must be in the same directory as main.py, must include '.wav'
def ReadWAVFile(file):
    return sf.read(file)[0]

#main program
#Contains example code for how to generate and read in datasets using the methods defined higher up in this file
#Also contains all of the machine learning models I came up with, in order of how good they were.
def main():
    Datasets = ['Bird Sounds', 'CalmCity', 'Fireworks', 'Market', 'SportsRadio', 'Video Game']

    for file in Datasets:
        GenerateDataset(file + '.wav', 44100)

    (X_train, y_train), (X_test, y_test) = ReadDatasets(Datasets)

    #get the shape of each piece of test data
    cols = X_train.shape[1]
    rows = X_train.shape[2]

    # #one hot encoding, transforms the list of correct angles into a binary matrix of 1s and 0s
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Get the number of different "buckets"
    num_classes = y_train.shape[1]

    #All models I tried were sequential
    model = Sequential()

    #Model 1(9% - 12% accuracy):
    #         Layers:
    #           A flatten layer to get the right shape for input into the dense layer
    #           A single dense layer with softmax activation.
    #
    #         Performance:  No better than random guessing, with accuracy between 9% and 12%
    #
    #         Layer code:
    #         model.add(Flatten(input_shape=(cols, rows)))
    #         model.add(Dense(num_classes, activation='softmax'))

    #Model 2( 12% - 19% accuracy):
    #    Layers:
    #       Two dense layers with linear activation. A dropout layer after each one which apparently helps the model to
    #           find multiple ways of solving the problem to help avoid overtraining
    #       One flatten layer to get the right shape for the final dense layer
    #       A final dense layer with softmax activation
    #
    #       Performance: up to 19% accuracy, an improvement over Model 1
    #    model.add(Dense(1000, activation='relu'))
    #    model.add(Dropout(0.4))
    #    model.add(Dense(100, activation='relu'))
    #    model.add(Dropout(0.4))
    #    model.add(Flatten(input_shape=(cols, rows)))
    #    model.add(Dense(num_classes, activation='softmax'))

    #Model 3(24%-29%):
    #    Layers:
    #       One Conv1D layer with the lowest numbers that would work.
    #       MaxPooling layer
    #       Dropout layer
    #       The rest of the layers are the same as model 2
    #
    #       Performance: 24-29% accuracy, an improvement over Model 1
    #    Layer Code:
    #    model.add(Conv1D(3, (2), input_shape=(cols, rows), activation='relu'))
    #    model.add(MaxPooling1D(pool_size=(2)))
    #    model.add(Dropout(0.4))
    #    model.add(Dense(1000, activation='relu'))
    #    model.add(Dropout(0.4))
    #    model.add(Dense(100, activation='relu'))
    #    model.add(Dropout(0.4))
    #    model.add(Flatten(input_shape=(cols, rows)))
    #    model.add(Dense(num_classes, activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    # model.summary()

    return

if __name__ == '__main__':
    main()
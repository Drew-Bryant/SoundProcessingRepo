#soundfile is used for reading and writing .wav files
import soundfile as sf
import multiprocessing as mp
#scipy.signal is used for calculating the time delay in samples
from numpy import float16
from scipy import signal
#sounddevice is used for recording audio
import sounddevice as sd
#matplotlib.pyplot is used for visualizing audio
import matplotlib.pyplot as plt
#audio arrays are numpy arrays, and numpy has some useful functions for arrays
import numpy as np
#i use random numbers to determine delays, noise, and volume changes
import random as rand
#used for testing processing times
import time
import wandb

import pandas as pd
import math

import wandb
from wandb.keras import WandbCallback

import csv

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

#debug flags
GENERATE_FILES = not True
GENERATE_GRAPHS = not True
DISTANCE = 51400

def RecordAudioClip(frames, which_device, channel):
    print("Recording Audio\n")
    data = sd.rec(frames, device=which_device, channels=channel, samplerate=44100)
    sd.wait()
    print("Finished Recording Audio\n")
    return data


def GenerateTrainingSet(seconds, sampleRate, device, channels, setSize):
    header = ['Clip #', 'Clip Delay(frames)', 'Angle to Sound(deg)', 'Total Frames']

    with open('trainingValues.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
    monoRecording = RecordAudioClip(int(seconds * sampleRate), device, channels)

    # duplicate result into two arrays
        # write the data
    for recording in range(1, setSize + 1):

        delayedChannel = np.array(monoRecording)
        regularChannel = np.array(monoRecording)

        # calculate an offset and new size to shift one array
        offset = int(rand.uniform(-128, 128))
        newSize = delayedChannel.size + 128

        print("The frame offset is: " + str(offset))

        # resize both arrays and roll the delayed one by the offset
        delayedChannel.resize(newSize)
        delayedChannel = np.roll(delayedChannel, offset)

        regularChannel.resize(newSize)

        stereoArray = np.vstack((regularChannel, delayedChannel)).T

        if GENERATE_GRAPHS and recording == 1:
            plt.title("Regular Channel Should Look Like This")
            plt.plot(stereoArray[:, 0], color="red")
            #
            plt.show()

            plt.title("Delayed Channel Should Look Like This")
            plt.plot(stereoArray[:, 1], color="red")
            #
            plt.show()
        fName = "TrainingClip" + str(recording) + ".wav"
        sf.write(fName, stereoArray, sampleRate)

        rad = math.acos(343 * (offset / sampleRate))
        angle = (rad * 180) / math.pi

        data = [str(recording), str(offset) , str(int(angle)) ,  delayedChannel.size]

        with open('trainingValues.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the data
            writer.writerow(data)

        ####################################################################
        #This part of the function is for testing that I can read files in correctly
        ##when reading in an array, you get back a 2 object array where
        #index 0 is a 2 column array of the channels, and index 1 is the sample rate
        #But we normally just want the channels
        # readArray = sf.read('DelayTesting.wav')[0]
        # print(readArray)

def ReadWAVFile(file):
    return sf.read(file)[0]
def ReadTrainingSet():
    recordings = np.empty((50, 441128, 2))
    angles = np.empty((50, 1))

    csv_data = pd.read_csv("trainingValues.csv")
    for number in range(1,51):
        name = "TrainingClip" + str(number) + ".wav"
        recordings[number - 1] = ReadWAVFile(name)



        angles[number - 1] = csv_data["Angle to Sound(deg)"][number - 1]

    return ( recordings, angles), (recordings, angles)

if __name__ == '__main__':

    #couldnt get wandb to work but it seems like a really cool tool for metrics
  ##  run = wandb.init()
  ##  config = run.config

    if GENERATE_FILES:
        seconds = 5
        sampleRate = 44100
        device = 1
        channels = 2
        setSize = 50
        GenerateTrainingSet(seconds, sampleRate, device, channels, setSize)

    (X_train, y_train), (X_test, y_test) = ReadTrainingSet()
    cols = X_train.shape[1]
    rows = X_train.shape[2]

    #one hot encoding
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    #Get the number of different "buckets"
    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Flatten(input_shape=(cols, rows)))
    model.add(Dense(num_classes))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

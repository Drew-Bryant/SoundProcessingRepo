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



import math

import csv

#debug flags
GENERATE_FILES = True
GENERATE_GRAPHS = True
DISTANCE = 51400

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
        readArray = sf.read('DelayTesting.wav')[0]
        print(readArray)
        # stereoArray = np.concatenate((regularChannel, delayedChannel), axis=0)
#generate a recording that is duplicated into two different recordings. Assumes a one channel initial recording
#where the second channel is delayed randomly by 1/3 to 1/10th of the length of the clip
#Args:
#seconds - length of the clip
#sampleRate - samples per second
#device - device number from a list that can be found by typing "python -m sounddevice" into a console with python installed
#channels - number of channels for the recording
#file - name of the file. must be passed with '.wav' in the string
#no return value, instead it writes a file to the local directory with the given filename
def GenerateFiles(seconds, sampleRate, device, channels, file):
    monoRecording = RecordAudioClip(int(seconds * sampleRate), device, channels)

    # duplicate result into two arrays
    delayedChannel = np.array(monoRecording)
    regularChannel = np.array(monoRecording)

    # calculate an offset and new size to shift one array
    offset = int(rand.uniform(1, 128))
    newSize = delayedChannel.size + offset

    print("The frame offset is: " + str(offset))

    # resize both arrays and roll the delayed one by the offset
    delayedChannel.resize(newSize)
    delayedChannel = np.roll(delayedChannel, offset)
    regularChannel.resize(newSize)

    stereoArray = np.vstack((regularChannel, delayedChannel)).T


    if GENERATE_GRAPHS:
        plt.title("Regular Channel Should Look Like This")
        plt.plot(stereoArray[:, 0], color="red")
        #
        plt.show()

        plt.title("Delayed Channel Should Look Like This")
        plt.plot(stereoArray[:, 1], color="red")
        #
        plt.show()
    sf.write(file, stereoArray, sampleRate)

    ####################################################################
    #This part of the function is for testing that I can read files in correctly
    ##when reading in an array, you get back a 2 object array where
    #index 0 is a 2 column array of the channels, and index 1 is the sample rate
    #But we normally just want the channels
    readArray = sf.read('DelayTesting.wav')[0]
    print(readArray)
    # stereoArray = np.concatenate((regularChannel, delayedChannel), axis=0)

    if GENERATE_GRAPHS:
        plt.title("Regular Channel")
        plt.plot(readArray[:, 0], color="red")
        #
        plt.show()

        plt.title("Delayed Channel")
        plt.plot(readArray[:, 1], color="red")
        #
        plt.show()

#record an audio clip into a numpy array that has one column per channel and one row per sample
#Args:
#frames - total number of samples to be recorded. should be passed an argument of the form: int(seconds * samples_per_second)
#which_device - device number from a list that can be found by typing "python -m sounddevice" into a console with python installed
#channel - number of channels for the recording
#Return: a numpy array that is samples x channels in size
def RecordAudioClip(frames, which_device, channel):
    print("Recording Audio\n")
    data = sd.rec(frames, device=which_device, channels=channel, samplerate=44100)
    sd.wait()
    print("Finished Recording Audio\n")
    return data

#helper function to return just the audio channels from a file read
#Args:
#file - name of the .wav file to read, '.wav' must be included in the argument string
#Return: a numpy array that is samples x channels in size
def ReadWAVFile(file):
    return sf.read(file)[0]

#helper function to determine which channel is louder for volume normalizing
#Args:
#channel1 - magic volume number gotten from averaging all the samples in a clip
#channel2 - same as channel1
#Returns: True if channel1 is louder, false otherwise
def LouderChannel(channel1, channel2):
    if(channel1 > channel2):
        return True
    else:
        return False

#generate a graph from a numpy array
#generates N graphs where N is the number of columns in the array
#Args:
#array - array to be graphed
#channels - number of columns in the array. Each column will be graphed separately
#No return value
def GenerateGraph(array, channels):
    for x in range(channels):
        plt.title('Channel ' + str(x + 1))
        plt.plot(array[:, x], color="red")
        plt.show()

def CrossCorrelateNoFFT(ch1, ch2):
    diagonals = ch1.size + ch2.size - 1
    correlationArray = np.zeros(diagonals)



    #Timing variables to track progress
    startTime = 0
    allTime = 0

    #reverse the second array for the calculation
    ch2Rev = np.flip(ch2)
    #Do the cross correlation, with transient matrix calculations
    #and each spot in the 'matrix' contains the product of row * col
    #from the correlation matrix, sum each diagonal to get the array of correlations
    #only do one array's worth of calculations
    for diagonal in range(ch1.size):
        #only do every 10th diagonal so the program finishes before the heat death of the universe
        if(diagonal % 10 == 0):
            if(diagonal % 1000 == 0):
                endTime = time.time()
                curTime = endTime - startTime
                allTime += curTime
                print("Current Diagonal: " + str(diagonal) + ", time since last batch: " + str(curTime) + ", Total Time: " + str(allTime) + "\n")
                startTime = time.time()
            currentSum = 0
            row = diagonal
            col = 0
            while (row >= 0):
                currentSum += ch1[row] * ch2Rev[col]
                row -= 1
                col += 1
            correlationArray[diagonal] = currentSum
    # for diagonal in range(ch1.size):
    #     if(diagonal % 1000 == 0):
    #         tyme = time.time()
    #         print("Current Diagonal: " + str(diagonal) + " and time: " + str(tyme) + "\n")
    #     currentSum = 0
    #     row = diagonal
    #     col = 0
    #     while (row >= 0):
    #         currentSum += ch1[row] + ch2[col]
    #         row -= 1
    #         col += 1
    #     correlationArray[ch1.size + diagonal] = currentSum
    return correlationArray


# #this function does not work
def DetectDelay(ch1, ch2):
    min_square = 200
    best_fit = 0
    for offset in range(int(-DISTANCE / 2), int((DISTANCE / 2) + 1), 1):
        sum = 0
        for a_sample in ch1:
            value = a_sample + offset
            diff = ch1[a_sample + offset] - ch2[a_sample]
            sum += diff * diff
        if sum < min_square:
            min_square = sum
            best_fit = offset
    return best_fit

#Trims the dead space off of a two channel recording
#the end of the later recording is lost in order to keep both channels the same length for other processing
#Args:
#recording - 2-column numpy array containing .wav compatible audio samples
#Returns: two numpy arrays. Get return values by providing variables in a comma separated list
# variables are returned in the following order: channel1Trimmed, channel2Trimmed
def Trim2ChRecording(recording):
    highMark = 0
    lowMark = 0

    # traverse channel 1 to determine a threshold for trimming
    # this could be more accurate if both channels were traversed and then averaged
    # but I have assumed both channels will be close enough in volume that it won't matter
    for sample in soundArray[:, 0]:

        if lowMark == 0:
            lowMark = abs(sample)

        if abs(sample) > highMark:
            highMark = abs(sample)

        if abs(sample < lowMark):
            lowMark = abs(sample)

    threshold = highMark * 0.03
    print("High Mark: " + str(highMark) + "\n")
    print("Threshold: " + str(threshold) + "\n")

    if GENERATE_GRAPHS:
        plt.title("Threshold")
        plt.plot(soundArray[:, 0], color='blue')
        plt.axhline(y=threshold, color='r', linestyle='-')
        plt.show()

    firstSlice = 0
    lastSlice = 0
    earliest = 0
    last = 0
    #
    # iterate over each channel and figure out where the useful chunk is
    # and make sure to preserve the time delay between channels
    for channel in soundArray.T:
        counter = 0
        # find start of interesting part
        while abs(channel[counter]) <= threshold:
            firstSlice = counter
            counter += 1

        if earliest == 0 or firstSlice < earliest:
            earliest = firstSlice

        # find end of interesting part by traversing from the end of the clip
        counter = channel.size - 1
        #pre-set lastSlice in case the later recording doesnt have a quiet end
        lastSlice = counter
        while abs(channel[counter]) <= threshold:
            lastSlice = counter
            counter -= 1

        if lastSlice > last:
            last = lastSlice

    print("Start of useful clip: " + str(earliest) + "\n")
    print("End of useful clip: " + str(last) + "\n")

    if GENERATE_GRAPHS:
        plt.title("Trim points regular channel")
        plt.plot(soundArray[:, 0], color='blue')
        plt.axvline(x=earliest, color='r', linestyle='-')
        plt.axvline(x=last, color='r', linestyle='-')
        plt.show()

        plt.title("Trim points delayed channel")
        plt.plot(soundArray[:, 1], color='blue')
        plt.axvline(x=earliest, color='r', linestyle='-')
        plt.axvline(x=last, color='r', linestyle='-')
        plt.show()

    firstRange = np.arange(1, earliest, 1)
    lastRange = np.arange(last, soundArray[:, 0].size, 1)
    totalRange = np.append(firstRange, lastRange)

    ch1 = np.delete(soundArray[:, 0], totalRange)
    ch2 = np.delete(soundArray[:, 1], totalRange)

    return ch1, ch2


if __name__ == '__main__':
    fileName = 'DelayTesting.wav'
    if GENERATE_FILES:
        #set variables and call recorder
        seconds = 5
        sampleRate = 44100
        device = 1
        channels = 2
        setSize = 50
        GenerateTrainingSet(seconds, sampleRate, device, channels, setSize)


def stuff():

    fileName = 'DelayTesting.wav'
    if GENERATE_FILES:
        #set variables and call recorder
        seconds = 5
        sampleRate = 44100
        device = 1
        channels = 1
        GenerateFiles(seconds, sampleRate, device, channels, fileName)

    soundArray = np.array(ReadWAVFile(fileName))


    #np.shape(soundArray)[0]) gets number of rows
    #np.shape(soundArray)[1]) gets number of cols, which is number of channels
    #GenerateGraph(soundArray, np.shape(soundArray)[1])

    channel1AfterTrim, channel2AfterTrim = Trim2ChRecording(soundArray)

    #adding noise
    #try uniform or poisson
    noise = np.random.normal(0, .005, channel2AfterTrim.shape)

    if GENERATE_GRAPHS:
        plt.title("Regular channel after trim")
        plt.plot(channel1AfterTrim, color='blue')
        plt.show()

        plt.title("Delayed channel after trim")
        plt.plot(channel2AfterTrim, color='blue')
        plt.show()

    channel2AfterTrim = channel2AfterTrim + noise

    if GENERATE_GRAPHS:
        plt.title("Delayed channel after noise added")
        plt.plot(channel2AfterTrim, color='blue')
        plt.show()

#Attempting to normalize volumes. Theres probably useful stuff here but I couldn't make sense of what I was doing
        plt.title("Regular Channel")
        plt.plot(channel1AfterTrim, color='blue')
        plt.show()

        plt.title("Delayed Channel before volume adjust")
        plt.plot(channel2AfterTrim, color='blue')
        plt.show()

    volumeAdjustment = 5

    print("Random Volume increase:" + str(volumeAdjustment) + "\n")

    channel2VolumeBeforeIncrease = channel2AfterTrim[channel2AfterTrim > 0].mean()

    channel2AfterVolume = channel2AfterTrim * (1 + volumeAdjustment)

    if GENERATE_GRAPHS:
        plt.title("Delayed Channel after volume adjust")
        plt.plot(channel2AfterVolume, color='blue')
        plt.show()

    channel2Volume = channel2AfterVolume[channel2AfterTrim > 0].mean()
    channel1Volume = channel1AfterTrim[channel1AfterTrim > 0].mean()

    print("Volume before increase for ch2:" + str(channel2VolumeBeforeIncrease))
    print("Volumes: " + str(channel1Volume) + ", " + str(channel2Volume))

    channel1Louder = LouderChannel(channel1Volume, channel2Volume)

    channel2Louder = not channel1Louder

    volumeDifference = abs(channel2Volume - channel1Volume)

    factor = 0
    if(channel1Louder):
        factor = channel2Volume / channel1Volume
        channel1AfterTrim *= factor
    else:
        #scale channel 2 down
        factor = channel1Volume / channel2Volume
        channel2AfterVolume *= factor
    print("Factor scaled down by: " + str(factor) + "\n")

    differenceArray = np.subtract(channel2AfterTrim, channel2AfterVolume)

    channel2VolumeAfterDecrease = channel2AfterVolume[channel2AfterVolume > 0].mean()

    print("Channel 2 Volume after decrease: " + str(channel2VolumeAfterDecrease))
    print("Difference between channel 1 and 2 volume:" + str(abs(channel1Volume - channel2VolumeAfterDecrease)))

    print("Error: " + str(abs(channel2VolumeAfterDecrease - channel2VolumeBeforeIncrease)))

    if GENERATE_GRAPHS:
        plt.title("Array after increasing and decreasing")
        plt.plot(channel2AfterVolume)
        plt.show()


        plt.title("Difference after inflating and deflating the array")
        plt.plot(differenceArray)
        plt.show()

    #DetectDelay function does not work
    #print("Delay after noise according to DetectDelay(): " + str(DetectDelay(channel1AfterTrim, channel2AfterTrim)))


    # start = time.time()

    #this figures out the correct frame delay even with uniform noise on one channel
    # creates a cross correlation array using fft methods
    fftCorrelation = signal.correlate(channel1AfterTrim - np.mean(channel1AfterTrim), channel2AfterTrim - np.mean(channel2AfterTrim), method='fft', mode="full")
    lags = signal.correlation_lags(len(channel1AfterTrim), len(channel2AfterTrim), mode="full")
    lag = lags[np.argmax(fftCorrelation)]

    manualCorrelation = CrossCorrelateNoFFT(channel1AfterTrim, channel2AfterTrim)
    convolution = signal.convolve(channel1AfterTrim, channel2AfterTrim, mode='same')
    print(fftCorrelation)
    print(convolution)

    print("Data type of channel arrays: " + str(channel1AfterTrim.dtype))

    plt.title("FFT correlation")
    plt.plot(fftCorrelation)
    plt.show()

    plt.title("Manual correlation")
    plt.plot(manualCorrelation)
    plt.show()



    # end = time.time()
    # print("Time with FFT: " + format(end - start))

    print("Delay according to magic: " + str(lag))




# Create a data set from one long sound file
import numpy as np
import wave
import soundfile as sf
import os
#soundfile is used for reading and writing .wav files
import csv
import math
import matplotlib.pyplot as plt
# i use random numbers to determine delays, noise, and volume changes
import random as rand
# scipy.signal is used for calculating the time delay in samples
import scipy.signal as signal
# sounddevice is used for recording audio
# matplotlib.pyplot is used for visualizing audio
# audio arrays are numpy arrays, and numpy has some useful functions for arrays
import numpy as np
import pandas as pd
import soundfile as sf
import os
g_dir_name = "sound_files"
g_sample_rate = 44100
g_index = 0


# Convert a numeric offset to a text directory name
def dir_name(offset):
    if offset < 0:
        return 'N' + str(-offset)
    elif offset == 0:
        return 'Z'
    else:
        return 'P' + str(offset)


# Convert a text directory name to an integer offset
def offset_from_name(name):
    if name[0] == 'Z':
        return 0
    elif name[0] == 'P':
        return int(name[1:])
    elif name[0] == 'N':
        return -int(name[1:])
    else:
        raise Exception("Invalid directory name")

def correlate(left, right):
    avg_l = np.average(left);
    avg_r = np.average(right);

    diff_l = left - avg_l
    diff_r = right - avg_r;

    sum1 = np.sum(diff_l * diff_r)

    square_l = diff_l * diff_l
    square_r = diff_r * diff_r

    sum2 = np.sum(square_l)
    sum3 = np.sum(square_r)

    ccv = sum1 / (np.sqrt(sum2) * np.sqrt(sum3))

    return ccv
#
#
def compute_correlations(left, right, samples_sep):
    max_ccv = -100000
    max_ccv_index = -1
    min_sq = 10000000
    min_sq_index = -1

    for shift in range(-samples_sep, samples_sep + 1):
        left_shift = np.roll(left, shift)
        ccv = correlate(left_shift, right)
        if abs(ccv) > max_ccv:
            max_ccv = abs(ccv)
            max_ccv_index = shift

        least_sq = correlate_least_squares(left_shift, right)
        if least_sq < min_sq:
            min_sq = least_sq
            min_sq_index = shift

        # print("Shift: ", shift, " correlation: ", ccv, " least squares: ", least_sq)

    # print("range: ", -samples_sep, samples_sep)
    # print("Max ccv:", max_ccv, max_ccv_index)
    # print("Min sq:", min_sq, min_sq_index)

    return [max_ccv, max_ccv_index, min_sq, min_sq_index]
# ##################################################
# # compute a correlation where small numbers (0) are good.
def correlate_least_squares(left, right):
    diff = left - right
    square = diff * diff
    return np.sum(square)
# Returns True if the sample isn't too quiet
# determination is based on the percentage of samples above a threshold
def valid_sample(slice, threshold=0.15, num_above=0.03):
    count = np.sum(abs(slice) > threshold)
    if count > len(slice) * num_above:
        return True

    return False


# create a unique filename given an offset
def create_filename(offset):
    global g_index
    g_index += 1
    return g_dir_name + '/' + dir_name(offset) + '/' + str(g_index) + '.wav'


# create a time shifted file from a mono source
def write_file(slice, clip_len, offset, start):
    sample = np.empty((clip_len, 2))
    sample[:, 0] = slice[offset + start: offset + start + clip_len]
    sample[:, 1] = slice[start: start + clip_len]

    sf.write(create_filename(offset), sample, g_sample_rate, 'PCM_16')


##################################################
# if __name__ == '__main__':
#     filename = input("source sound file: ")
#     csvName = filename[:len(filename) - 4]
#     csvName += "Thresholds.csv"
#     header = ['Clip #', 'Actual Delay(frames)', 'Calculated (scipy.signal)', 'Difference(scipy signal)', 'Calculated(least squares)', 'Difference(least squares)', 'Calculated(max ccv)', 'Difference (max ccv)', 'Clip Length(frames)']
#
#     with open(csvName, 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#     data, g_sample_rate = sf.read(filename)
#
#     print('Length', len(data), 'sample rate:', g_sample_rate)
#
#     g_dir_name = input("Output directory: ")
#     clip_len_t = input("Length per sample: ")
#     min_offset_t = input("Min offset: ")
#     max_offset_t = input("Max offset: ")
#
#     # files_per_offset = int(files_per_offset_t)
#     clip_len = int(clip_len_t)
#     min_offset = int(min_offset_t)
#     max_offset = int(max_offset_t)
#
#     oversize = 2 * max(abs(min_offset), abs(max_offset))
#
#     # create directory structure
#     os.mkdir(g_dir_name, mode=700)
#     for offset in range(min_offset, max_offset + 1):
#         subdirname = dir_name(offset)
#         print("Creating", subdirname)
#         os.mkdir(g_dir_name + "/" + subdirname, 700)
#
#     num_files = len(data) // clip_len
#
#     offset = min_offset
#     count = 0
#     for index in range(num_files):
#         slice = data[index * clip_len: (index + 1) * clip_len + oversize]
#         #if valid_sample(slice):
#         if True:
#             #write_file(slice, clip_len, offset, abs(min_offset))
#             offset += 1
#             count += 1
#             if offset > max_offset:
#                 offset = min_offset
#             start = abs(min_offset)
#
#             stereoArray = np.empty((clip_len, 2))
#             stereoArray = np.vstack((slice[offset + start: offset + start + clip_len], slice[start: start + clip_len])).T
#
#
#             #fftCorrelation = signal.correlate(stereoArray[:, 0] - np.mean(stereoArray[:, 0]), stereoArray[:, 1] - np.mean(stereoArray[:, 1]), method='fft', mode="full")
#             #lags = signal.correlation_lags(len(stereoArray[:, 0]), len(stereoArray[:, 1]), mode="full")
#             #lag = lags[np.argmax(fftCorrelation)]
#             lag = -1000000
#             max_ccv, max_ccv_index, min_sq, min_sq_index = compute_correlations(stereoArray[:, 0], stereoArray[:, 1], 9)
#
#             data = [str(index), str(offset) , str(lag),str(abs(offset - lag)), str(min_sq_index), str(abs(offset - min_sq_index)), str(max_ccv_index), str(abs(offset - max_ccv_index)), len(slice)]
#
#
#             with open(csvName, 'a', encoding='UTF8', newline='') as f:
#                 writer = csv.writer(f)
#
#                 # write the data
#                 writer.writerow(data)
#
#     print('wrote', count, 'files')
#-----------------------------------------------------------------------------------------------------------------------
# #soundfile is used for reading and writing .wav files
# import csv
# import math
# import matplotlib.pyplot as plt
# # i use random numbers to determine delays, noise, and volume changes
# import random as rand
# # scipy.signal is used for calculating the time delay in samples
# import scipy.signal as signal
# # sounddevice is used for recording audio
# # matplotlib.pyplot is used for visualizing audio
# # audio arrays are numpy arrays, and numpy has some useful functions for arrays
# import numpy as np
# import pandas as pd
# import soundfile as sf
# import os
#
# # used for testing processing times
#
maxSamplesBetweenMics = 9

#Trims the dead space off of a two channel recording
#the end of the later recording is lost in order to keep both channels the same length for other processing
#Args:
#recording - 2-column numpy array containing .wav compatible audio samples
#Returns: two numpy arrays. Get return values by providing variables in a comma separated list
# variables are returned in the following order: channel1Trimmed, channel2Trimmed
def Trim2ChRecording(soundArray):
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

    firstRange = np.arange(1, earliest, 1)
    lastRange = np.arange(last, soundArray[:, 0].size, 1)
    totalRange = np.append(firstRange, lastRange)

    ch1 = np.delete(soundArray[:, 0], totalRange)
    ch2 = np.delete(soundArray[:, 1], totalRange)

    return ch1, ch2

def GetSingleChannel(file, channelNo):
    return np.array(file[:, channelNo])

def ThresholdTesting(file):
    header = ['Clip #', 'Actual Delay(frames)', 'Calculated (scipy.signal)', 'Difference(scipy signal)', 'Calculated(least squares)', 'Difference(least squares)', 'Calculated(max ccv)', 'Difference (max ccv)', 'Clip Length(frames)']
    csvName = file + 'Thresholds.csv'

    with open(csvName, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    filename = '/Audio/' + file + '.wav'
    sampleRate = 44100
    BirdAudio = ReadWAVFile(os.path.dirname(__file__) + filename)
    BirdCh1 = GetSingleChannel(BirdAudio, 0)
    BirdCh2 = GetSingleChannel(BirdAudio, 1)


    BirdClipSamples = BirdCh2.size
    BirdRecordings = math.ceil(BirdClipSamples / (44100 * 5))

    x = 0
    # duplicate result into two arrays
    # write the data
    testCount = 1
    trainingCount = 1
    offset = 0
    lag = 0
    while len(BirdCh1) > 0:
        clipLength = 44100
        regClip = np.copy(BirdCh1[:(clipLength)])
        delClip = np.copy(regClip)

        #uncomment these two lines if you want all the clips to be different
        BirdCh1 = np.copy(BirdCh1[clipLength:])
        #BirdCh2 = np.copy(BirdCh2[clipLength:])

        newSize = regClip.size + maxSamplesBetweenMics

        regClip.resize(newSize)
        delClip.resize(newSize)
        # calculate an offset and new size to shift one array
        offset = int(rand.uniform(-4, 4))
        if (x % 100) == 0:
            print("in outer loop ", x, "Offset for this clip:", offset)
        x += 1
        # resize both arrays and roll the delayed one by the offset
        delClip = np.roll(delClip, offset)
        stereoArray = np.vstack((regClip, delClip)).T

        #uncomment these two lines if you are generating wav files
        #fName = "BirdTrainingClip" + str(trainingCount) + ".wav"
        #sf.write(fName, stereoArray, sampleRate)


        #np.fft.rfft(ch1, n=n)
        #n = ch1/shape + ch2.shape

        #correlate gives the actual correlation. np.fft.rfft() gives the fft
        # fftCorrelation = signal.correlate(stereoArray[:, 1] - np.mean(stereoArray[:, 1]), stereoArray[:, 0] - np.mean(stereoArray[:, 0]), method='fft', mode="full")
        # lags = signal.correlation_lags(len(stereoArray[:, 1]), len(stereoArray[:, 0]), mode="full")
        # lag = lags[np.argmax(fftCorrelation)]
        regClip = GetSingleChannel(stereoArray, 0)
        delClip = GetSingleChannel(stereoArray, 1)
        clipLength = -1
        clipLength = len(regClip)

        while clipLength > 0:
            if (x % 100) == 0:
                print("in inner loop ", x)
            x += 1
            regClip = np.copy(regClip[:(clipLength)])
            delClip = np.copy(delClip[:(clipLength)])
            stereoArray = np.vstack((regClip, delClip)).T

            fftCorrelation = signal.correlate(stereoArray[:, 0] - np.mean(stereoArray[:, 0]), stereoArray[:, 1] - np.mean(stereoArray[:, 1]), method='fft', mode="full")
            lags = signal.correlation_lags(len(stereoArray[:, 0]), len(stereoArray[:, 1]), mode="full")
            lag = lags[np.argmax(fftCorrelation)]

            max_ccv, max_ccv_index, min_sq, min_sq_index = compute_correlations(stereoArray[:, 0], stereoArray[:, 1], 9)

            data = [str(trainingCount), str(offset) , str(lag),str(abs(offset - lag)), str(min_sq_index), str(abs(offset - min_sq_index)), str(max_ccv_index), str(abs(offset - max_ccv_index)), regClip.size]

            if clipLength >= 2000:
                clipLength -= 1000
            else:
                clipLength -= 100

            with open(csvName, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)

                # write the data
                writer.writerow(data)

        trainingCount += 1


##################################################
# compute a correlation where large numbers (1) are good.



##################################################

def ReadWAVFile(file):
    return sf.read(file)[0]

def WriteWAVFile(filename, Data):
    sf.write(filename, data=Data, samplerate=44100)
def ReadTrainingSet():
    numTestClips = 31
    numTrainingClips = 127
    clipLength = 44100 * 5 + maxSamplesBetweenMics

    testClips = np.empty((numTestClips, clipLength, 2))
    testAngles = np.empty((numTestClips, 1))
    trainingClips = np.empty((numTrainingClips, clipLength, 2))
    trainingAngles = np.empty((numTrainingClips, 1))

    trainingData = pd.read_csv("BirdTrainingValues.csv")
    testData = pd.read_csv("BirdTestValues.csv")

    for number in range(1, numTestClips + 1):
        name = "BirdTestClip" + str(number) + ".wav"
        testClips[number - 1] = ReadWAVFile(name)
        testAngles[number - 1] = testData["Angle to Sound(deg)"][number - 1]

    for number in range(1, numTrainingClips + 1):
        name = "BirdTrainingClip" + str(number) + ".wav"
        trainingClips[number - 1] = ReadWAVFile(name)
        trainingAngles[number - 1] = trainingData["Angle to Sound(deg)"][number - 1]

    return (trainingClips, trainingAngles), (testClips, testAngles)

if __name__ == '__main__':
    # stereoArray = ReadWAVFile('Market.wav')
    # print(stereoArray.shape)

    files = ['Market', 'SportsRadio', 'Video Game']
    # uncomment these two lines if you are generating wav files
    # fName = "BirdTrainingClip" + str(trainingCount) + ".wav"
    # sf.write(fName, stereoArray, sampleRate)
    #print(os.path.dirname(__file__) + '/Audio/CalmCity.wav')

    for file in files:
        wavFile = ReadWAVFile(os.path.dirname(__file__) + '/Audio/' + file + '.wav')
        plt.title(file)
        plt.plot(wavFile)
        plt.show()
        #ThresholdTesting(file)

# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

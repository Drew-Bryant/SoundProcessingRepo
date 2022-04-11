
#from scipy.io.wavfile import write
#read and write to files with soundfile
import soundfile as sf
import multiprocessing as mp
#recorded with sounddevice
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np


def rec_audio(frames, which_device, storage_array, channel):
    print("in device" + str(which_device) + "\n")
    data = sd.rec(frames, device=which_device, channels=channel, samplerate=44100)
    sd.wait()
    storage_array.put(data)

if __name__ == '__main__':
    #mp.set_start_method('fork')
    recordings = mp.Queue()
    seconds = 5
    sampleRate = 44100
    handheldProcess = mp.Process(target=rec_audio, args=(int(seconds * sampleRate), 1, recordings, 2))
    webcamProcess = mp.Process(target=rec_audio, args=(int(seconds * sampleRate), 1, recordings, 1))

    handheldProcess.start()
    # webcamProcess.start()
    print("Recording audio...")
    handheldProcess.join(15)
    handheldRecArray = recordings.get()

    plt.title("Channel 1 No noise")
    plt.plot(handheldRecArray[:, 0], color="red")
    #
    plt.show()

    plt.title("Channel 2 No noise")
    plt.plot(handheldRecArray[:, 1], color="red")
    #
    plt.show()
    # sf.write('TwoChannelStereo.wav', handheldRecArray, sampleRate)  # Save as WAV file
    # webcamProcess.join(1)
    # webcamRecArray = recordings.get()
    # sf.write('OneChannelMono.wav', webcamRecArray, sampleRate)  # Save as WAV file
    # print("Finished Recording.")
    #
    # #sf.read reads the same number of channels as the array
    # MonoSourceStereoOutput = sf.read('OneChannelMono.wav', always_2d=True)
    # StereoSourceStereoOutput = sf.read('TwoChannelStereo.wav')
    #
    # MonoSourceStereoOutput = np.array(MonoSourceStereoOutput)
    # StereoSourceStereoOutput = np.array(StereoSourceStereoOutput)
    # print(MonoSourceStereoOutput)
    # print(StereoSourceStereoOutput)
    #
    # plt.title("Channel 1 Stereo")
    # plt.plot(StereoSourceStereoOutput[:0], color="red")
    #
    # plt.show()
    #
    # plt.title("Channel 2 Stereo")
    # plt.plot(StereoSourceStereoOutput[:1], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 1 Mono")
    # plt.plot(MonoSourceStereoOutput[:0], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 2 Mono")
    # plt.plot(MonoSourceStereoOutput[:1], color="red")
    # #
    # plt.show()


    # np.set_printoptions(precision=3)
    # print("Original Size: " + str(handheldRecArray.size))
    # print(str(handheldRecArray.ndim))
    # print (handheldRecArray)
    #
    # newArray = np.array(handheldRecArray)
    # x = seconds
    # y = handheldRecArray
    #
    # noise = np.random.normal(0, .001, handheldRecArray.shape)
    # new_signal = handheldRecArray + noise
    # noise2 = np.random.normal(0, .0002, handheldRecArray.shape)
    # new_signal2 = handheldRecArray + noise2
    # noise3 = np.random.normal(0, .0003, handheldRecArray.shape)
    # new_signal3 = handheldRecArray + noise3
    #sf.write('OneChannelMono.wav', webcamProcess, sampleRate)  # Save as WAV file

    #how to print
    #when doing multi channel stuff


    # plt.title("Channel 1 With Noise")
    # plt.plot(new_signal[:, 0], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 2 With noise")
    # plt.plot(new_signal[:, 1], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 1 With Noise")
    # plt.plot(new_signal2[:, 0], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 2 With noise")
    # plt.plot(new_signal2[:, 1], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 1 With Noise")
    # plt.plot(new_signal3[:, 0], color="red")
    # #
    # plt.show()
    #
    # plt.title("Channel 2 With noise")
    # plt.plot(new_signal3[:, 1], color="red")
    # #
    # plt.show()

    # shiftFactor = int(newArray.size/5)
    # newArray.resize(newArray.size + shiftFactor)
    # newArray = np.roll(newArray, shiftFactor)
    #
    # print("Adjusted Size: " + str(newArray.size))
    #
    # plt.title("Shifted Recording")
    # plt.plot(newArray, color="red")
    # #
    # plt.show()



# handheldMicRec = sd.rec(int(seconds * sampleRate), device=1)
# webcamMicRec = sd.rec(int(seconds * sampleRate), device=2)
# print("Recording audio...")
# sd.wait()  # Wait until recording is finished
# print("Finished Recording.")





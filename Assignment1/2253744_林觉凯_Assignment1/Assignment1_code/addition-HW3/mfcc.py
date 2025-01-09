import librosa
import numpy as np
import matplotlib.pyplot as plt
import struct

# 对信号进行预加重处理
def preEmphasis(signal, alpha=0.97):
    emphasizedSignal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasizedSignal

# 信号需要被分成短时间帧
def frameDivide(data, frameLen, frameMov):
    sigLen = len(data)
    frameNum = int(np.ceil((sigLen - frameLen) / frameMov))
    zeroNum = (frameNum * frameMov + frameLen) - sigLen
    zeros = np.zeros(zeroNum)
    filledSignal = np.concatenate((data, zeros))
    indices = np.tile(np.arange(0, frameLen), (frameNum, 1)) + \
              np.tile(np.arange(0, frameNum * frameMov, frameMov), (frameLen, 1)).T
    indices = np.array(indices, dtype=np.int32)
    divided = filledSignal[indices]
    return divided

# 使用汉明窗进行加窗操作
def hammingWindow(audio, frameLen, alpha=0.46164):
    saveImage(audio[300], "BeforeWindowing", 'samples', 'Amplitude')
    n = np.arange(frameLen)
    window = 1 - alpha - alpha * np.cos(2 * np.pi * n / (frameLen - 1))
    saveImage(window, "HammingWindow", 'samples', 'Amplitude')
    windowed_audio = audio * window
    saveImage(windowed_audio[300], "AfterWindowing", 'samples', 'Amplitude')
    return windowed_audio

# 进行快速傅里叶变换
def stft(audioFrame, nFft):
    magnitudeFrame = np.absolute(np.fft.rfft(audioFrame, nFft))
    powerFrame = (1.0 / nFft * (magnitudeFrame ** 2))
    saveImage(powerFrame[300], "Power", 'freq(Hz)', 'Amplitude')
    return powerFrame

# 梅尔滤波器处理
def melFilter(sampleRate, nFft):
    lowFreqMel = 0
    highFreqMel = 2595 * np.log10(1 + (sampleRate / 2) / 700)
    nfilt = 40
    melPoints = np.linspace(lowFreqMel, highFreqMel, nfilt + 2)
    hzPoints = 700 * (10 ** (melPoints / 2595) - 1)
    fbank = np.zeros((nfilt, int(nFft / 2 + 1)))
    bin = (hzPoints / (sampleRate / 2)) * (nFft / 2)
    for m in range(1, nfilt + 1):
        fMMinus = int(bin[m - 1])
        fM = int(bin[m])
        fMPlus = int(bin[m + 1])
        for k in range(fMMinus, fM):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(fM, fMPlus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    plotSpectrogram(fbank.T, "Mel Filter Bank", "Filter Number", "Frequency Bin")
    return fbank

# 进行对数转换
def applyLogFilterBankEnergy(filterBanksEnergy):
    filterBanksEnergy = np.where(filterBanksEnergy <= 0, np.finfo(float).eps, filterBanksEnergy)
    logMelEnergy = 20 * np.log10(filterBanksEnergy)
    return logMelEnergy

# 离散余弦变换
def dct(logMel, nMfcc=26, nCeps=12):
    transpose = logMel.T
    lenData = len(transpose)
    dctAudio = []
    for j in range(nMfcc):
        temp = 0
        for m in range(lenData):
            temp += (transpose[m]) * np.cos(j * (m - 0.5) * np.pi / lenData)
        dctAudio.append(temp)
    ret = np.array(dctAudio[1:nCeps + 1])
    return ret

# 动态特征提取
def delta(data, k=1):
    deltaFeat = []
    transpose = data.T
    q = len(transpose)
    for t in range(q):
        if t < k:
            deltaFeat.append(transpose[t + 1] - transpose[t])
        elif t >= q - k:
            deltaFeat.append(transpose[t] - transpose[t - 1])
        else:
            denominator = 2 * sum([i ** 2 for i in range(1, k + 1)])
            numerator = sum([i * (transpose[t + i] - transpose[t - i]) for i in range(1, k + 1)])
            deltaFeat.append(numerator / denominator)
    return np.array(deltaFeat)

# 特征变换，归一化或标准化
def normalization(data):
    dataMean = np.mean(data, axis=0, keepdims=True)
    dataVari = np.var(data, axis=0, keepdims=True)
    return (data - dataMean) / dataVari

# 使用Librosa库生成的MFCC结果
def compareWithLibrosa(audio, sr, mfccHandwritten):
    librosaMfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=frameMov // 1, n_fft=512)
    minFrames = min(mfccHandwritten.shape[1], librosaMfcc.shape[1])
    mfccHandwritten = mfccHandwritten[:, :minFrames]
    librosaMfcc = librosaMfcc[:, :minFrames]
    librosaMfcc = librosa.util.normalize(librosaMfcc, axis=1)
    plotSpectrogram(librosaMfcc, 'Librosa MFCC Spectrum', 'frameNum', 'MFCC coefficients')

def plotAudio(x, y, title, xLabel, yLabel):
    plt.plot(x, y, linewidth=1)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"image/{title}.png")
    plt.close()

def saveImage(data, title, xLabel, yLabel):
    plt.plot(data, linewidth=1)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"image/{title}.png")
    plt.close()

def plotSpectrogram(spec, title, xLabel, yLabel):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.tight_layout()
    plt.savefig(f"image/{title}.png")
    plt.close()

def saveMfccToFile(mfccData, filename):
    with open(filename, 'wb') as f:
        for frame in mfccData.T:
            f.write(struct.pack('f' * len(frame), *frame))

if __name__ == "__main__":
    audioPath = "ljkTest2.wav"
    startTime = 1.0
    endTime = 3.0
    sr = 16000

    audio, _ = librosa.load(audioPath, sr=sr)[:15*sr]
    print("Reading complete!")
    plotAudio(np.arange(len(audio))/sr, audio, 'Origin Waveform', 'Time (s)', 'Amplitude')
    
    audio = preEmphasis(audio)
    print("PreEmphasis complete!")
    plotAudio(np.arange(len(audio))/sr, audio, 'PreEmphasis Audio Waveform', 'Time (s)', 'Amplitude')
    
    frameLen = int(sr * 0.025)
    frameMov = int(sr * 0.010)
    audioFrame = frameDivide(audio, frameLen, frameMov)
    print("FrameDivide complete!")
    plotSpectrogram(audioFrame.T, 'Divided Frames Spectrum', 'Frame Number', 'Amplitude')
    
    audioFrame = hammingWindow(audioFrame, frameLen)
    print("Windowing complete!")
   
    powerFrame = stft(audioFrame, nFft=512)
    print("Stft complete!")
   
    fbank = melFilter(sampleRate=sr, nFft=512)
    print("Fbank generate complete!")
    
    filterBanksEnergy = np.dot(powerFrame, fbank.T)
    logMelPowerFrame = applyLogFilterBankEnergy(filterBanksEnergy)
    print("Apply log mel complete!")
    plotSpectrogram(logMelPowerFrame.T, 'logMelPowerFrame Spectrum', 'frameNum', 'Log Mel Filter Banks')

    mfcc = dct(logMelPowerFrame)
    print("Discrete Fourier transform complete!")
    
    energyFrameSquareSum = np.sum((powerFrame ** 2).T, axis=0, keepdims=True)
    energyFrameSquareSum[energyFrameSquareSum <= 0] = 1e-30
    
    energyFrame = 10 * np.log10(energyFrameSquareSum)
    mfcc = np.append(mfcc, energyFrame, axis=0)
    plotSpectrogram(mfcc, 'mfcc Spectrum', 'frameNum', 'MFCC coefficients')
    print("Mfcc feature generate complete!")
    
    deltaData = delta(mfcc)
    deltaSquareData = delta(deltaData.T)
    mfccWithDelta1 = np.append(mfcc, deltaData.T, axis=0)
    mfccWithDelta12 = np.append(mfccWithDelta1, deltaSquareData.T, axis=0)
    plotSpectrogram(deltaData.T, 'Delta Spectrum', 'frameNum', 'Delta coefficients')
    plotSpectrogram(deltaSquareData.T, 'Delta-Delta Spectrum', 'frameNum', 'Delta-Delta coefficients')
    print("Dynamic feature generation complete!")

    norData = normalization(mfccWithDelta12)
    plotSpectrogram(norData, 'normalized mfccWithDelta12 Spectrum', 'frameNum', 'MFCC coefficients')

    saveMfccToFile(norData, 'ljkTest2.mfc')
    print("MFCC features saved to ljkTest2.mfc")

    compareWithLibrosa(audio, sr, mfccWithDelta12)
    print("Comparison with Librosa complete!")
import librosa
import numpy as np
import matplotlib.pyplot as plt

 # 对信号进行预加重处理
def preEmphasis(signal, alpha=0.97):
    # alpha: 预加重系数为0.97
    emphasizedSignal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasizedSignal

# 信号需要被分成短时间帧
def frameDivide(data, frameLen, frameMov):
    sigLen = len(data)
    # 计算帧数，向上取整，保证信号能被完整分帧
    frameNum = int(np.ceil((sigLen - frameLen) / frameMov))

    zeroNum = (frameNum * frameMov + frameLen) - sigLen
    zeros = np.zeros(zeroNum)

    # 将信号填充零后的完整信号
    filledSignal = np.concatenate((data, zeros))
    indices = np.tile(np.arange(0, frameLen), (frameNum, 1)) + \
              np.tile(np.arange(0, frameNum * frameMov, frameMov), (frameLen, 1)).T

    # 根据索引提取分帧后的信号
    indices = np.array(indices, dtype=np.int32)
    divided = filledSignal[indices]

    return divided

# 使用汉明窗进行加窗操作
def hammingWindow(audio,frameLen,alpha=0.46164):
    saveImage(audio[300],"BeforeWindowing",'samples','Amplitude')
    
    # 创建一个帧长为 frame_len 的汉明窗函数
    n = np.arange(frameLen)
    #  Hamming窗的公式
    window = 1-alpha - alpha * np.cos(2 * np.pi * n / (frameLen - 1))
    saveImage(window,"HammingWindow",'samples','Amplitude')
    # 将每一帧的音频信号与汉明窗函数逐点相乘，得到加窗后的信号
    windowed_audio = audio * window
    saveImage(windowed_audio[300],"AfterWindowing",'samples','Amplitude')
    return windowed_audio

# 进行快速傅里叶变换
def stft(audioFrame, nFft):
    # 使用 rfft 得到每个频率的结果
    magnitudeFrame = np.absolute(np.fft.rfft(audioFrame, nFft))
    powerFrame = (1.0 / nFft * (magnitudeFrame ** 2))
    saveImage(powerFrame[300], "Power", 'freq(Hz)', 'Amplitude')
    return powerFrame

# 梅尔滤波器处理
def melFilter(sampleRate, nFft):
    # 将频率范围映射到梅尔刻度上
    lowFreqMel = 0
    highFreqMel = 2595 * np.log10(1 + (sampleRate / 2) / 700)
    
    nfilt = 40
    melPoints = np.linspace(lowFreqMel, highFreqMel, nfilt + 2)
    hzPoints = 700 * (10 ** (melPoints / 2595) - 1)
    # 创建滤波器组，每个滤波器有 nFft/2 + 1 个频率点
    fbank = np.zeros((nfilt, int(nFft / 2 + 1)))
    bin = (hzPoints / (sampleRate / 2)) * (nFft / 2)
     # 生成每个梅尔滤波器的三角滤波器
    for m in range(1, nfilt + 1):
        fMMinus = int(bin[m - 1])
        fM = int(bin[m])
        fMPlus = int(bin[m + 1])
        for k in range(fMMinus, fM):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(fM, fMPlus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    plotSpectrogram(fbank.T, "Mel Filter Bank", "Filter Number", "Frequency Bin")
    # 返回梅尔滤波器组
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
    
    # 对每个MFCC系数进行计算
    dctAudio = []
    for j in range(nMfcc):
        temp = 0
        # 计算DCT系数
        for m in range(lenData):
            temp += (transpose[m]) * np.cos(j * (m - 0.5) * np.pi / lenData)
        dctAudio.append(temp)
    # 只保留前nCeps个系数
    ret = np.array(dctAudio[1:nCeps + 1])
    return ret

# 动态特征提取
def delta(data, k = 1):
    # deltaFeat 用于存储每一帧的导数特征
    deltaFeat = []
    transpose = data.T
    q = len(transpose)
    for t in range(q):
        # 对于前k帧，使用后一帧与当前帧的差值计算
        if t < k:
            deltaFeat.append(transpose[t + 1] - transpose[t])
        # 对于最后k帧，使用当前帧与前一帧的差值计算
        elif t >= q - k:
            deltaFeat.append(transpose[t] - transpose[t - 1])
        # 对于中间帧，使用对称的加权差分公式计算
        else:
            denominator = 2 * sum([i ** 2 for i in range(1, k + 1)])
            numerator = sum([i * (transpose[t + i] - transpose[t - i]) for i in range(1, k + 1)])
            deltaFeat.append(numerator / denominator)
    return np.array(deltaFeat)

# 特征变换，归一化或标准化
def normalization(data):
    dataMean = np.mean(data, axis = 0, keepdims = True)
    dataVari = np.var(data, axis = 0, keepdims = True)
     # 归一化操作，将每个特征值减去均值并除以方差
    return (data - dataMean) / dataVari

# 使用Librosa库生成的MFCC结果
def compareWithLibrosa(audio, sr, mfccHandwritten):
    # 设置 hop_length 更小，增加每帧的精度
    librosaMfcc = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = 13, hop_length=frameMov // 1, n_fft = 512)
    minFrames = min(mfccHandwritten.shape[1], librosaMfcc.shape[1])
    mfccHandwritten = mfccHandwritten[:, :minFrames]
    librosaMfcc = librosaMfcc[:, :minFrames]
    
    # 归一化 Librosa 生成的 MFCC 系数
    librosaMfcc = librosa.util.normalize(librosaMfcc, axis=1)
    # 显示 Librosa 的 MFCC 计算结果
    plotSpectrogram(librosaMfcc, 'Librosa MFCC Spectrum', 'frameNum', 'MFCC coefficients')

def plotAudio(x, y, title, xLabel, yLabel):
    plt.plot(x, y, linewidth = 1)  
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"image/{title}.png")
    plt.close()

def saveImage(data, title, xLabel, yLabel):
    plt.plot(data, linewidth = 1) 
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

if __name__ == "__main__":
    # 导入项目音频
    audioPath = "Assignment1_voice.wav"
    startTime = 1.0
    endTime = 3.0
    sr = 16000

    # 原始音频绘图
    audio, _ = librosa.load(audioPath, sr=sr)[:15*sr]
    print("Reading complete!")
    plotAudio(np.arange(len(audio))/sr, audio, 'Origin Waveform', 'Time (s)', 'Amplitude')
    
    # 预加重操作
    audio = preEmphasis(audio)
    print("PreEmphasis complete!")
    plotAudio(np.arange(len(audio))/sr, audio, 'PreEmphasis Audio Waveform', 'Time (s)', 'Amplitude')
    
    # 分帧操作
    frameLen = int(sr * 0.025)
    frameMov = int(sr * 0.010)
    audioFrame = frameDivide(audio, frameLen, frameMov)
    print("FrameDivide complete!")
    plotSpectrogram(audioFrame.T, 'Divided Frames Spectrum', 'Frame Number', 'Amplitude')
    
    # 加窗操作
    audioFrame = hammingWindow(audioFrame, frameLen)
    print("Windowing complete!")
   
    # STFT操作
    powerFrame = stft(audioFrame, nFft=512)
    print("Stft complete!")
   
    # 梅尔滤波器操作
    fbank = melFilter(sampleRate=sr, nFft=512)
    print("Fbank generate complete!")
    
    # 加对数操作
    filterBanksEnergy = np.dot(powerFrame, fbank.T)
    logMelPowerFrame = applyLogFilterBankEnergy(filterBanksEnergy)
    print("Apply log mel complete!")
    plotSpectrogram(logMelPowerFrame.T, 'logMelPowerFrame Spectrum', 'frameNum', 'Log Mel Filter Banks')

    # DCT
    mfcc = dct(logMelPowerFrame)
    print("Discrete Fourier transform complete!")
    
    energyFrameSquareSum = np.sum((powerFrame ** 2).T, axis=0, keepdims=True)
    energyFrameSquareSum[energyFrameSquareSum <= 0] = 1e-30
    
    energyFrame = 10 * np.log10(energyFrameSquareSum)
    mfcc = np.append(mfcc, energyFrame, axis=0)
    plotSpectrogram(mfcc, 'mfcc Spectrum', 'frameNum', 'MFCC coefficients')
    print("Mfcc feature generate complete!")
    
    # 计算一阶导数 (Delta)
    deltaData = delta(mfcc)
    # 计算二阶导数 (Delta-Delta)
    deltaSquareData = delta(deltaData.T)
    mfccWithDelta1 = np.append(mfcc, deltaData.T, axis=0)
    mfccWithDelta12 = np.append(mfccWithDelta1, deltaSquareData.T, axis=0)
    # 单独绘制一阶导数图
    plotSpectrogram(deltaData.T, 'Delta Spectrum', 'frameNum', 'Delta coefficients')
    # 单独绘制二阶导数图
    plotSpectrogram(deltaSquareData.T, 'Delta-Delta Spectrum', 'frameNum', 'Delta-Delta coefficients')
    print("Dynamic feature generation complete!")

    # 归一化操作
    norData = normalization(mfccWithDelta12)
    plotSpectrogram(norData, 'normalized mfccWithDelta12 Spectrum', 'frameNum', 'MFCC coefficients')

    # 对比 python 库中的 MFCC 函数
    compareWithLibrosa(audio, sr, mfccWithDelta12)
    print("Comparison with Librosa complete!")
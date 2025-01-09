# Automatic_Speech_Recognition_TongjiSSE_2024Fall

## 1. Speech-Signal-Processing

Extract acoustic features (MFCC) for a segment of speech. Comment your codes properly.

 Processing steps include:

- Pre-emphasis
- Windowing
- STFT
- Mel-filter bank
- Log()
- DCT
- Dynamic feature extraction
- Feature transformation

Compare your results with the output of MFCC function provided by Python package. Give possible reasons for the difference in your report.

Upload your codes and report.

## 2. Viterbi-Algorithm

Teacher-mood-model

 One week, your teacher gave the following homework assignments:

| **Monday** | **Tuesday** | **Wednesday** | **Thursday** | **Friday** |
| :--------: | :---------: | :-----------: | :----------: | :--------: |
|     A      |      C      |       B       |      A       |     C      |

 Questions:

What did his mood curve look like most likely that week?

Give the full process of computation in your report.

You can refer to the Result Table in P. 60 in ch3 HMM.pptx. p.s. Some numbers in the table are incorrect.

## 3. HMM-GMM

1. Refer to the experiment manual and submit the experiment report. At the same time, take the features extracted by myself in the first job as input, observe and identify the results. If there is a big difference between the features extracted by myself and those returned by the platform interface, analyze the possible reasons.

2. Using Maximum Likelihood Estimation method, estmate the parameters of mean $\boldsymbol{\mu}$ in a multivariate Gaussian model given a set of sampled data $\bold X=\{\bold {x_1},\bold {x_2},...,\bold {x_n}\}$.

   The pdf of the multivariate Gaussian model is $p(\bold x|\boldsymbol{\mu},\boldsymbol{\Sigma})=\frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} e^{-\frac{1}{2}(\bold x - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\bold x - \boldsymbol{\mu})}$

## 4. Deepspeech2

Upload your experimental report which describes the outputs of the important steps of the experiment as well as the evaluation results.

Answer the question:

How to speed the program when it costs 50 hours to train only an epoch?

|  Assignment   |                 Content                 | Scores |
| :-----------: | :-------------------------------------: | :----: |
|  Assignment1  |        Speech-Signal-Processing         | 10/10  |
|  Assignment2  |            Viterbi-Algorithm            | 10/10  |
|  Assignment3  |                 HMM-GMM                 | 10/10  |
|  Assignment4  |               Deepspeech2               | 10/10  |
| Final_Project | VoiceLink-Intelligent-Meeting-Assistant |   U    |


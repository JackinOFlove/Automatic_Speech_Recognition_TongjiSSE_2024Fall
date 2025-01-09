# Assignment4: Deepspeech2

[TOC]

## 1. Development Environment Construction

### 1.1. Server Purchase

On the Huawei Cloud ECS homepage, search "ECS" and click "Management Console" to enter the ECS management page.

<img src=".\assets\1.png" alt="1" width= 500px />

Create an elastic cloud server Select "North China-Beijing Four" in the console area, select "Elastic Cloud Server" in the left menu bar, and "Buy Elastic Cloud Server" in the upper right corner.

<img src=".\assets\2.png" alt="2" width= 500px />

In the Basic Configuration, select the following configuration:

+ Billing mode: billing on demand.

+ Region: North China-Beijing 4th.

+ Available area: Random allocation.

+ CPU architecture: x86 calculation.

+ Specification: General calculation enhanced type | c7.2xlarge.2 | 8vCPUs | 16 GiB

+ Mirror: Public mirror, Ubuntu, Ubuntu 18.04 server 64bit

+ System disk: Universal SSD, 100GB.

<img src=".\assets\3.png" alt="3" width= 500px />

Click "Next: Network Configuration" in the next right corner of the window In the Network Configuration, select the following configuration:

+  Network: You can go to the console to create a new virtual private cloud. 
+ Expand the network card: no. 
+ Security group: You can create a new security group. 
+ Flexible public network IP: 
+ Buy it now. Line: full dynamic BGP. 
+ Public network bandwidth: charge by traffic. 
+ Broadband size: custom, 200 Mbit/s. 
+ Release behavior: Check the Release with the instance.

<img src=".\assets\4.png" alt="4" width= 500px />

Click "Next: the advanced configuration" in the lower right corner of the window In the Advanced Configuration section, select the following configuration: 

Cloud server name: It can be customized. 

Login certificate: password. 

User name: root. 

Password: custom (subsequent login use, remember). C

loud backup: it is not purchased temporarily. 

Cloud Server Group: None. Advanced option: None.

<img src=".\assets\5.png" alt="5" width= 500px />

Click "Next: Confirm configuration" in the lower right corner of the window, In Confirm Configuration, select the following configuration: Agreement: Check that I have read and agreed to the Mirror Disclaimer.

Click "Buy Now" in the lower right corner of the window. After the Task Submission succeeded, select Return to Server List to return to the management console of the elastic cloud server and see that the ECS created is running. 

**Note** the flexible public network IP address displayed in the IP Address, which will be used later.

<img src=".\assets\6.png" alt="6" width= 500px />

### 1.2. MobaXterm Connect ECS

Download the MobaXterm Go to MobaXterm's official website home page: https://mobaxterm.mobatek.net/ Select the Home Edition and download the MobaXterm Home Edition v21.x (Portable edition). Unpack the MobaXterm_Portable_v21.x.zip file after the download is complete.

<img src=".\assets\7.png" alt="7" width= 500px />

Use MobaXterm to connect to an elastic cloud server Enter the decompression MobaXterm_Portable_v21.x folder, Open the MobaXterm_Personal_21.x.exe file, Select the "Session" of the menu bar, Then enter the "Session settings" page, Remote link selection "SSH" protocol, Enter Figure 2-12 The elastic public network IP address displayed when the ECS elastic cloud server is created, Select the specified user name "Specify username", User name is "root", Select OK for submission after the configuration is complete.

<img src=".\assets\8.png" alt="8" width= 500px />

<img src=".\assets\9.png" alt="9" width= 600px />

MobaXterm Login to ECS requires a password. In step 4 of ECS elastic cloud server, the elastic cloud server root user password has been customized in the advanced configuration. You can enter here. MobaXterm The remote link to the elastic cloud server is successful, and the cloud environment of the elastic cloud server should be further configured later.

<img src=".\assets\10.png" alt="10" width= 600px />

## 2. Code and Data Download

### 2.1. Get Code Files

Use git to download the source code of the training script from mindspore, switch to the home directory, create a working directory such as / work, and execute the following command:

```bash
git clone https://gitee.com/mindspore/models.git
```

The deepspeech2 project code for this experiment is located at models / research / audio / deepspeech2.

Training and inference-related parameters in the config.py file.

<img src=".\assets\16.png" alt="16" width= 500px />

### 2.2. Dataset and its Preprocessing

#### 2.2.1. Download the LibriSpeech Dataset

The link to download the data set is: http://www.openslr.org/12.

+ **training set**

​	Train-clean-100: [6.3G] (100 Hour No Noise Speech Training Set) (just download this file) 

+ **validation set**

​	dev-clean.tar.gz [337M] (No Noise) 

​	dev-other.tar.gz [314M] (with noise) 

+ **test set**

​	test-clean.tar.gz [346M] (Test set, No Noise) 

​	test-other.tar.gz [328M] (Test set, noisy)

LibriSpeech Data directory structure, is as follows:

```markdown
|──LibriSpeech
|── train
|   |─train-clean-100
|── val
|   |─dev-clean.tar.gz
|   |─dev-other.tar.gz
|── test_other
|   |─test-other.tar.gz
|──test_clean
|   |─test-clean.tar.gz
```

#### 2.2.2. Install the Python3.9.0

Installing python dependency and software such as gcc.

```bash
sudo apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev libgmp-dev sox libjpeg8-dev
```

<img src=".\assets\20.png" alt="20" width= 600px />

Use wget to download the python3.9.0 source package, which can be downloaded to any directory of the installation environment with the command:

```bash
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz
tar -zxvf Python-3.9.0.tgz
```

<img src=".\assets\17.png" alt="17" width= 600px />

<img src=".\assets\18.png" alt="18" width= 500px />

Go to the decompression folder and execute the configuration, compile, and installation commands:

```bash
cd Python-3.9.0
chmod +x  configure 
./configure --prefix=/usr/local/python3.9.0 --enable-shared
make
sudo make install
```

<img src=".\assets\19.png" alt="19" width= 500px />

<img src=".\assets\21.png" alt="21" width= 500px />

Query whether there is libpython3.9.so.1.0 under / usr / lib 64 or / usr / lib, skip this step or back up the libpython3.9.so.1.0 file with the following command.

```bash
cp /usr/local/python3.9.0/lib/libpython3.9.so.1.0 /usr/lib
```

<img src=".\assets\22.png" alt="22" width= 600px />

Perute the following command setting soft link:

```bash
sudo ln -s /usr/local/python3.9.0/bin/python3.9 /usr/bin/python
sudo ln -s /usr/local/python3.9.0/bin/pip3.9 /usr/bin/pip
sudo ln -s /usr/local/python3.9.0/bin/python3.9 /usr/bin/python3.9
sudo ln -s /usr/local/python3.9.0/bin/pip3.9 /usr/bin/pip3.9
```

<img src=".\assets\23.png" alt="23" width= 600px />

After the installation is complete, perform the following command to view the installation version, and if the relevant version information is returned, the installation is successful.

```bash
python3.9 --version
pip3.9  --version
```

<img src=".\assets\24.png" alt="24" width= 600px />

#### 2.2.3. Install the MindSpore and the Required Dependency Package

Install mindspore, install according to the actual server architecture, please refer to https://www.mindspore.cn/install.

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/x86_64/mindspore-1.6.0-cp39-cp39-linux_x86_64.whl \
    --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

<img src=".\assets\25.png" alt="25" width= 500px />

Pip source installation, you can add a mirror source installation when the dependent package file is large, such as pip install-i https://pypi.tuna.tsinghua.edu.cn/simple sox.

```bash
pip3.9 install wget
pip3.9 install tqdm
pip3.9 install sox
```

<img src=".\assets\26.png" alt="26" width= 500px />

<img src=".\assets\27.png" alt="27" width= 500px />

#### 2.2.4. The Data Preprocessing SeanNaren Scripts was Downloaded

After MobaXterm / Finalshell (recommended) connects to the ECS server, switch to the home directory, create the working directory, and then use the scripts in SeanNaren to process the data. SeanNaren Script link: https://github.com/SeanNaren/deepspeech.pytorch.

```bash
cd ../home
mkdir work
cd work
git clone https://github.com/SeanNaren/deepspeech.pytorch.git
```

Here, you need to access the external network, it is recommended to download directly to the local and then upload the zip file before decompression

<img src=".\assets\28.png" alt="28" width= 600px />

#### 2.2.5. LibriSpeech Data Preprocessing

The training set of train-clean-100 was downloaded locally via the dataset link http://www.openslr.org/12, validation sets dev-clean.tar.gz and dev-other.tar.gz, and test sets test-clean.tar.gz and test-other.tar.gz. Upload the local data set to the MobaXterm server.

The structure of the data set is shown below:

<img src=".\assets\32.png" alt="32" width= 500px />

Copy the librispeech.py in the data directory of the deepspeech.pytorch to the deepspeech.pytorch directory and execute the following command:

```bash
cd deepspeech.pytorch
cp ./data/librispeech.py ./
```

<img src=".\assets\30.png" alt="30" width= 600px />

Modify the librispeech.py code data set path, refer to step 3, set the directory structure in the current directory, and change the code path to the data set actual path, as shown in the figure below:

<img src=".\assets\31.png" alt="31" width= 500px />

Execute the data set processing command, and execute the command as follows.

```bash
python librispeech.py
```

<img src=".\assets\33.png" alt="33" width= 500px />

After the data processing, the data directory structure is as follows:

```markdown
├─ LibriSpeech_dataset
    │  ├── train
    │  │   ├─ wav
	│  │   └─ txt
    │  ├── val
    │  │    ├─ wav
    │  │    └─ txt
    │  ├── test_clean  
    │  │    ├─ wav
    │  │    └─ txt  
    │  └── test_other
    │       ├─ wav
    │       └─ txt
    └─ libri_test_clean_manifest.json, libri_test_other_manifest.json, libri_train_manifest.json, libri_val_manifest.json
```

Go from the json file to the csv file, create the json_to_csv.py in the deepspeech.pytorch directory, and copy the code to the file with the following code:

```bash
touch json_to_csv.py
```

```python
# json_to_csv.py:
import json
import csv
import argparse


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument("--json", type=str, default="", help="")
parser.add_argument("--csv", type=str, default="", help="")
config = parser.parse_args()

def trans(jsonfile, csvfile):
    jsonData = open(jsonfile)
    csvfile = open(csvfile, "a")
    for i in jsonData:
        dic = json.loads(i[0:])
        root_path = dic["root_path"]
        for j in dic["samples"]:
            wav_path = j["wav_path"]
            transcript_path =j["transcript_path"]
            res_wav = root_path + '/' + wav_path
            res_txt = root_path + '/' + transcript_path
            res = [res_wav, res_txt]
            writer = csv.writer(csvfile)
            writer.writerow(res)
    jsonData.close()
    csvfile.close()

if __name__ == "__main__":
    trans(config.json, config.csv)
```

Run the command as shown in the following figure:

```bash
python json_to_csv.py --json libri_test_clean_manifest.json --csv libri_test_clean_manifest.csv
python json_to_csv.py --json libri_test_other_manifest.json --csv libri_test_other_manifest.csv
python json_to_csv.py --json libri_train_manifest.json --csv libri_train_manifest.csv
python json_to_csv.py --json libri_val_manifest.json --csv libri_val_manifest.csv
```

<img src=".\assets\34.png" alt="34" width= 600px />

## 3. Model Training Results and Evaluation

### 3.1. Model Training

Switch to the models / official / audio / DeepSpeech2 directory, Model training requires the creation of the deepspeech_pytorch directory under the DeepSpeech2 directory and the decoder.py file under the deepspeech_pytorch directory.

```bash
mkdir deepspeech_pytorch
cd deepspeech_pytorch
touch decoder.py
```

Copy the code to the decoder.py file with the following code:

```python
#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))
    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self,
                 labels,
                 lm_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4,
                 blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        labels = list(labels)  # Ensure labels are a list before passing to decoder
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)
        def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence,
                       size,
                       remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
            Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets
```

Model configuration Modify the config.py under the src. After the modification, "ctrl + s" is saved and exits. 

Modify batch_size to 1 (one-process data size that is related to server device performance). 

Modify epochs to 1 (about 48h, can be adjusted according to the actual demand).

Modifies the train_manifest to the libri_train_manifest.csv actual path.

Modifies test_manifest to libri_test_clean_manifest.csv to the actual path.

Modify the window type of eval_config to change hanning to hann.

Install the model python dependency:

```bash
cd /home/work/models/official/audio/DeepSpeech2
pip3.9 install -r requirements.txt
pip3.9 install Levenshtein
pip3.9 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.7.1
pip3.9 install numpy==1.20.0
pip install numba==0.53.1
```

<img src=".\assets\35.png" alt="35" width= 600px />

Download the pre-training model, the download link is https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/ASR/DeepSpeech.ckpt, and the download command is as follows:

```bash
wget https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/ASR/DeepSpeech.ckpt
```

<img src=".\assets\36.png" alt="36" width= 600px />

Modify the run_standalone_train_cpu.sh in the scripts directory to load the pre-training model The modification is as follows:

```python
PATH_CHECKPOINT=$1
python ./train.py --device_target 'CPU' --pre_trained_model_path $PATH_CHECKPOINT
```

<img src=".\assets\37.png" alt="37" width= 600px />

Train the model in the DeepSpeech2 directory and enter the following command.

```bash
bash scripts/run_standalone_train_cpu.sh PATH_CHECKPOINT
# PATH_CHECKPOINT: Pre-training file path
```

<img src=".\assets\38.png" alt="38" width= 600px />

So that, the model is now being trained.

### 3.2. Model Training Results and Evaluation

#### 3.2.1. View the Training Log

If we want to view the training log, the current directory under the train.log. Enter the command as follows:

```bash
tail –f train.log
```

<img src=".\assets\40.png" alt="40" width= 600px />

So you can observe the training log all the time.

#### 3.2.2. Training Results and Evaluation

Model evaluation, enter the following command for evaluation.

```bash
bash scripts/run_eval_cpu.sh [PATH_CHECKPOINT]
# [PATH_CHECKPOINT] The Model checkpoint file
```

View the evaluation log, the eval.log in the current directory. Enter the command as follows:

```bash
tail –f eval.log
```

On my computer, the evaluation result is shown in the figure below:

<img src=".\assets\42.png" alt="42" width= 500px />

ASR refers to the automatic speech recognition technology (Automatic Speech Recognition), which is a technology to convert human speech into text. WER is the word error rate, Word Error Rate (WER) is an important indicator used to evaluate ASR performance, used to evaluate the error rate between predicted text and standard text, so the biggest characteristic of the word error rate is that the smaller, the better.

The two indicators on my computer are probably the same as those above in the experimental manual, and the model evaluation effect is good.

Here I have a problem, when the loss becomes nan or inf after the model training for a long time, it will automatically stop. Just like the following two pictures, I trained twice, one until the step length was more than 500, and the other was more than 600, about half an hour.

<img src=".\assets\43.png" alt="43" width= 600px />

<img src=".\assets\44.png" alt="44" width= 600px />

After many times of training, it is inevitable that this situation, I combine the data here to think there are the following several possibilities:

1. The learning rate is too high, which may lead to the gradient explosion, and the parameter update amplitude is too large, which makes the model weight becomes unstable. In the early stage of training, the update range of model weights is large. If the learning rate is too high, it is easy to lead to too large gradient and cause numerical overflow.
2. Gradient explosion, in deep networks the gradient may grow exponentially with backpropagation, leading in numerical spillover. When dealing with long sequences, the gradients can easily accumulate and eventually explode.
3. If the value of the loss function is unstable, the loss function may return a large value or directly return an invalid value. If the output of the model is logits and the value is too large, the numerical overflow may be caused when calculating the Softmax. Cross-entropy loss is prone to extreme values with probability approaching 0 or 1.
4. Sequence length or batch size problems, increasing memory requirements and computational complexity when handling long speech sequences or large volumes of data, may trigger numerical overflows. The longer the sequence length, the more information needed to process, easily leading to gradient problems. When the batch size is too large, the gradient of each update may be too intense.

### 3.3. Model Export

Model export that requires the following code to modify the export.py file.

```python
config = train_config
context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
with open(config.DataConfig.labels_path) as label_file:
labels = json.load(label_file)
```

<img src=".\assets\45.png" alt="45" width= 600px />

Enter the following command for converting and exporting the model file.

<img src=".\assets\46.png" alt="46" width= 600px />

Then enter the command ll in the directory, and you can see the model file:

<img src=".\assets\47.png" alt="47" width= 500px />

## 4. Answer the Question

### 4.1. Question Description

1 round of model training, need 50 hours, how to improve the speed?

### 4.2. Problem Thinking

+ **Replace the stronger hardware**

Using either a GPU or a TPU.

GPU: When training a deep learning model, the GPU is the most commonly used acceleration hardware. Compared with CPU, GPU can greatly improve the training speed when processing matrix operations. A NVIDIA GPU, such as A100, V100, or RTX 3090, is recommended. 

TPU: If you use the Google Cloud, or any other TPU-enabled platform, the TPU performance may be stronger and is suitable for large-scale deep learning tasks. 

Multi-GPU parallel training: If a single GPU is not enough, multiple GPU can be used for parallel training (data parallel or model parallel). Frameworks such as TensorFlow, PyTorch, and Mindspore all support this training approach.

Using a high-performance server:

If train on a cloud server, consider using stronger instances (such as NVIDIA Tesla V100, A100, or TPU instance). Cloud computing platforms: AWS, Google Cloud, and Azure all provide powerful computing resources that can be expanded as needed.

+ **Training with a mixed-precision approach**

FP16 (Hybrid Precision Training): Hybrid precision training (FP16) can accelerate training while saving memory. It uses most of the calculations using 16-bit floating-points instead of 32-bit, thereby increasing computation speed and reducing memory footprint.

+ **Data preprocessing and loading optimization**

Data preprocessing acceleration: If the data preprocessing before the training (such as audio feature extraction, image enhancement, etc.) is too time-consuming, it will significantly affect the training speed.

+ **Adjust the bulk size**

Increase the batch size (Batch Size): Increasing the batch size can reduce the time required to calculate each step. However, increasing the batch size increases memory consumption and needs to be adjusted for hardware resources.

Increasing batch size: Sometimes a large batch size can cause video memory outages. In this case, the "progressive batch size" strategy can be adopted to gradually increase the batch size until the video memory reaches the maximum capacity.

+ **Distributed training**

Data parallel: use multiple GPU or computing nodes, using data parallel training. Each compute node or GPU processes a different batch of the training data to accelerate the training by synchronously updating the model parameters.

Hybrid parallelism: combine model parallelism and data parallelism, distribute the model to multiple devices, and accelerate the training through data parallelism. This applies to very large models.

+ **Model architecture optimization**

Pruning (Pruning): Pruning is a technique that reduces the number of model parameters to accelerate reasoning and training by removing redundant neurons or connections.

Knowledge distillation (Knowledge Distillation): Train a small model to simulate the output of a large pre-training model to accelerate the inference and training process. 

Network compression: smaller networks (such as MobileNet, EfficientNet) are used to replace complex network models. For some tasks, using a lighter architecture significantly reduces the training time.

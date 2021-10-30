


# acoustic-fencing

<h1 align="center">
  <br>
Acoustic Fence Using Multi-Microphone Speaker Separation
  <br>
</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/orel-ben-reuven/">Orel Ben-Reuven</a> •
    <a href="https://www.linkedin.com/in/tomer-fait-69020b21b">Tomer Fait</a> •
    <a href="">Amir Ivry</a>
  </p>
<h4 align="center">
<p>Project A & Special @ Sipl Lab</p>
<p>Technion - Israel Institute of Technology</p>
</h4>

<h4 align="center">
<a href="Report/Acoustic_Fence_Report.pdf">Project Report</a> :
<a href="Report/Acoustic_Fence_Presentation.pdf">Project Presentation</a> :
<a href="results_samples/">Demo</a>
</h4>




https://user-images.githubusercontent.com/14962234/139452641-68e6b849-0694-450b-8507-cc4b411fb480.mp4



# Acoustic-Fencing 

> **Acoustic Fence Using Multi-Microphone Speaker Separation**<br> Orel Ben-Reuven, Tomer Fait, Amir Ivry<br>
>
> **Abstract:** *The goal of an acoustic fencing algorithm is to separate speakers by their physical location in space. In this project, we examine an algorithm that solves this problem, define suitable performance criteria, and test the algorithm in varied environments, both simulated and real. The real recordings were acquired by us with suitable acoustic equipment. We examine a speech separation algorithm based on spectral masking inferred from the speaker’s direction. The algorithm assumes the existence of a dominant speaker in each time-frequency (TF) bin and classifies these bins by employing a deep convolutional neural network. Traditional evaluation criteria do not independently quantify the effects of the desired signal distortion and the undesired signal attenuation, and often result in a single numeric value for both effects. In this project, we propose a method for evaluating these phenomena separately by applying the separation mask to the original separated signals. This mask is time-dependent and represents the network’s gain, such that by applying it to the desired signal (for example), we can evaluate the network’s effect on the signal. We tested the algorithm and evaluation criteria on simulated signals with varied room sizes, speakers locations, and reverberation times.  Following the success in the simulation, we continued to test the algorithm on real recordings acquired in the lab employing a microphone array and mouth simulator. To evaluate the generalization of the system, a test set was comprised of recordings acquired in rooms that were not present in the training set. In conclusion, this research describes an acoustic fencing algorithm and evaluation criteria with a high correlation to human perception and shows successful performance in a real acoustic environment. Furthermore, the system’s low resource consumption and fast response times might indicate that it is suitable as a practical algorithm in a real system.*

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Acoustic-Fencing](#acoustic-fencing)
* [Separation Results](#separation-results)
* [How To Use](#how-to-use)
	* [Installation](#installation)
	* [Usage](#usage)
		* [Requirements for Simulated Recordings](#requirements-for-simulated-recordings)
		* [Requirements for Real Recordings](#requirements-for-real-recordings)
		* [Train and Seperate](#train-and-seperate)
* [References](#references)

## Separation Results
Some separation results are available in the ***/results_samples*** folder. The results are split into simulation data and real recordings and are categorized by their acquisition room. Each separation result contains three files: `mix.wav` which contains the mix signals as they were captured by the microphone array, and `speaker_0.wav, speaker_1.wav` which contain the separated signals as the acoustic-fence system output. 

## How To Use

### Installation:
Clone the repo and install the dependencies:
* `git clone https://github.com/Orelbenr/acoustic-fencing.git`
* `cd acoustic-fencing`
* `pip install -r requirements.txt`

### Usage:
Our implementation supports both simulated recording from some database or real recording of speakers in their separation zones.

#### 1 - Requirements for Simulated Recordings
For this type of usage, one needs:
 - [x]  A directory containing audio files of recordings from a database. 
> Note that the files don't need to be separated into separation zones, or contain any reverberation or noise; these will be added in the simulation.

 - [x] A room impulse response matrix.
This matrix defines the room dimensions and locations of the microphones and speakers. 
The matrix sould be of the form [k, m, n, N]  where: 
	* k - length of the RIR,
	* m - number of microphones,
	* n - number of reception zones,
	* N - number of situations.

	We added a Matlab script in ***\create_rir_matrix\simulation.mat*** which construct such a matrix. Parameters can be adjusted in their responding section in the code.

> An example of such a matrix is located in ***\create_rir_matrix\output_dir\demo*** 
>  This matrix represent 5 situations from various rooms.   


#### 2 - Requirements for Real Recordings
For this type of usage, one needs to have recordings from a microphone array. The recordings should be categorized into folders according to their reception zones. Each sentence should be a folder containing .wav files of all of the microphones' recordings of that sentence. 

> The names of the folders can be arbitrary, but they are used for the output sentences names.

An example of such directory containing 2 reception zones, 2 sentences in each zone, and 3 microphones in the array:

![folders](https://user-images.githubusercontent.com/14962234/139452318-8fe678de-f2b0-43bf-b578-7ba0a6160b90.png)

#### 3 - Train and Seperate
To train the network and separate recordings into reception zones, one only needs to modify ***\Code\main.py***. 
First, fill in the desired paths in the ***directories*** section. Then, choose your ***run*** configuration.
The program lets you execute a consecutive list of ***runs***. Each ***run*** can be customized according to the following params:

> Some params may not be relevant in every run. Those params are marked as Simulation, Real Recordings, Train, Test.
> In that case - their value can be arbitrary.

| Parameter | Description | Example |
|--|--|--|
|run_name | A name used for the output library and more. | 'demo_run'
|test_out_name | A name for the test results folder | 'test_results' 
|rir_name | (Simulation) The name of the RIR .mat file| 'demo'
micN | The number of microphones | 9
zoneN | The number of reception zones| 2
spN | The number of simulatanous speakers | 2
batch | (Train) The network batch size| 32
lr | (Train) Initial learning rate| 1e-3
perm_skip| Dialate the training and testing set by a factor  (to reduce runtime). | 0
 seg_len | The size of a time segment for the net input| 100
 epochs | (Train) Number of ephoces for Training stage| 30
sc_step | (Train) Scheduler step ephoc length| 10
sc_gamma | (Train) Scheduler step factor | 0.5
train | If True - train the network | True
test| If True - run seperation on test set | True
files2save | (Test) Number of sentences to save as an example | 5
evaluate | (Test) If True - evaluate entire test dataset, else only seperate file2save | True
is_simulation | If True - Simulation run, else - Real Recordings run | True
old_model | (Train) The full path of an old model to continue training from | None


## References
* [1] Chazan, Shlomo E., et al. ["Multi-Microphone Speaker Separation based on Deep DOA Estimation."](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8903121) 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.
* [2] J.B. Allen and D.A. Berkley, "Image method for efficiently simulating small-room acoustics," Journal Acoustic Society of America, 65(4), April 1979, p 943.
* [3] https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator
* [4] https://github.com/schmiph2/pysepm/
* [5] https://github.com/zhixuhao/unet/
* [6] Written with [StackEdit](https://stackedit.io/).

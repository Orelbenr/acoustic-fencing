import numpy as np
from scipy import signal as ss
import soundfile as sf
from typing import List, Tuple
nparr = np.ndarray


def our_stft(signals: List[nparr]) -> nparr:
    """
    This function applies STFT transformation to every signal in a list, and returns the result in
    a concatenated 3d numpy array
    :param signals: list of numpy arrays that are to be transformed
    :return: matrix of size [stft_freq,stft_time,len(signals)] of transformed spectograms
    """
    stft_n = 512  # STFT window size
    return np.stack([ss.stft(sig, nperseg=stft_n)[2] for sig in signals], axis=2)


def norm_conv(x: nparr, h: nparr) -> nparr:
    """
    ( This function is not used )
    :param x:
    :param h:
    :return:
    """
    energy_x = np.sum(np.abs(x))
    f = np.convolve(x, h)
    energy_f = np.sum(np.abs(f))
    if energy_f > 1e-6:  # is not zero
        f = f*(energy_x/energy_f)
    return f


def wav_read16(file_path: str) -> nparr:
    signal, sample_rate = sf.read(file_path)
    assert sample_rate == 16_000
    return signal


def load_example(selected_files: Tuple[int], file_list: List,
                 rirs: nparr = None, rir_idx: int = None, zone_dict: nparr = None) -> List[List[nparr]]:
    """
    :param selected_files: touple of the indexes of the files to load from file list
    :param file_list: list of sound files
    :param rirs: matrix of RIRs of shape [rir_len,mic_num,speaker_num,sample_num]
    :param rir_idx: index of rir
    :param zone_dict: translates a speaker index to a zone index
    :return: a list of speakers where each item is a list of microphones where each contains
    a sound file padded to a consistent length
    """
    if rirs is not None:
        # Simulation: read selected files from the file_list (a list of strings)
        # and convolve them with the appropriate RIR
        speakers = [[np.convolve(wav_read16(file_list[file_idx]), rir) for rir in rirs[:, :, speaker_idx, rir_idx].T]
                    for speaker_idx, file_idx in enumerate(selected_files)]
    else:
        # Real Recording: file_list is a list of zones, each zone is a list of sentences, each sentence is a list
        # of microphones (a string representing a file).
        speakers = []
        for sp_idx, sentence_idx in enumerate(selected_files):
            zone_idx = zone_dict[sp_idx]
            cyc_sentence_idx = sentence_idx % len(file_list[zone_idx])
            mics = []
            for mic in file_list[zone_idx][cyc_sentence_idx]:
                mics.append(wav_read16(mic))
            speakers.append(mics)

    # find the maximum length of a file
    max_len = np.max([np.size(mic) for sp in speakers for mic in sp])

    # pad each file
    for i, speaker in enumerate(speakers):
        for j, mic in enumerate(speaker):
            speakers[i][j] = np.pad(mic, (0, max_len - len(mic)), 'constant')

    return speakers


def create_tags(speakers: List[List[nparr]], zone_dict: nparr) -> nparr:
    """
    :param speakers: a list of speakers where each item is a list of microphones where each contains
    a sound file padded to a consistent length
    :param zone_dict: numpy array of the number of the zone for the corresponding index.
    :return: the tags matrix
    """
    spectograms = our_stft([speakers[i][0] for i, _ in enumerate(speakers)])
    selected_tag = np.argmax(np.abs(spectograms), axis=2)
    selected_tag = np.take(zone_dict, selected_tag)
    return selected_tag


def create_example(speakers: List[List[nparr]]) -> nparr:
    """
    :param speakers: a list of speakers where each item is a list of microphones where each contains
    a sound file padded to a consistent length
    :return: matrix of size [features(freq),file_len(time),(mic_num-1)*2] which contains the cosine and a sine of the
     normalized stft phase of microphones relative to the reference mic
    """
    mics = [sum(mic) for mic in zip(*speakers)]                        # sum the speakers for each microphone.
    z = our_stft(mics)                                                 # calculate STFT for all microphones

    angles = np.angle(z)                                               # get STFT phase
    angles = angles[:, :, 1:] - angles[:, :, 0][:, :, np.newaxis]      # normalize by reference microphone

    layers = np.concatenate((np.sin(angles), np.cos(angles)), axis=2)  # return the sin and cos as the input features
    return layers


def awgn(speakers: List[List[nparr]], avgSNR: float = 35) -> List[List[nparr]]:
    """
    Add AWGN noise to each input signal microphone wise. It adds the same noise to each microphone.
    :param speakers : input signals
    :param avgSNR : desired noise average SNR (35 db)
    :return noisy_speakers : noised speakers (s + n)
   """
    snr = np.random.normal(avgSNR, 5)
    gamma = 10**(snr/10)
    ref_mic = sum(speakers[:][0])
    N0 = np.mean(np.abs(ref_mic)**2) / gamma
    num_mics = len(speakers[0])
    num_speakers = len(speakers)
    sp_shape = speakers[0][0].shape

    n = [np.sqrt(N0 / 2) * np.random.standard_normal(sp_shape) / num_speakers for _ in range(num_mics)]
    noisy_speakers = [[mic + n[mic_num] for mic_num, mic in enumerate(sp)] for sp in speakers]
    return noisy_speakers


def rand_sir(speakers: List[List[nparr]], std: float) -> List[List[nparr]]:
    """
    Randomize the SIR between the speakers.
    :param speakers: input signals
    :param std: standard deviation of sir values
    :return: modified_speakers: speakers after SIR randomization
    """
    num_speakers = len(speakers)
    ref_sp = np.random.randint(num_speakers-1)
    sirs = np.abs(std * np.random.randn(num_speakers-1))
    sirs = np.insert(sirs, ref_sp, 0)
    powers = [np.std(sp[0]) for sp in speakers]
    ref_power = powers[ref_sp]
    wanted_powers = [ref_power / (10**(sir/20)) for sir in sirs]
    modified_speakers = [[mic*wanted_powers[sp_num]/powers[sp_num] for mic in sp] for sp_num, sp in enumerate(speakers)]
    return modified_speakers

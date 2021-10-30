import logging
import numpy as np
from helpers.preprocessing import our_stft
from scipy import signal as ss
import os
import soundfile as sf
from os.path import join as pjoin
import pysepm as pm
from typing import List

nparr = np.ndarray


def our_istft(z: nparr) -> List[nparr]:
    """
    :param z: matrix of size [stft_freq,stft_time,len(signals)] of transformed spectrograms
    :return: signals: list of numpy arrays
    """
    stft_n = 512
    return [ss.istft(cur_z, nperseg=stft_n)[1] for cur_z in np.transpose(z, (2, 0, 1))]


def synthesis(files_dict: dict, masks: nparr, save_path: str, is_saved=False, is_evaluated=False) -> dict:
    # initialize some variables
    num_speakers = len(files_dict['speakers'])  # files_dict['speakers'] is a list of numpy where each numpy is a speaker in the reference mic
    num_zones = masks.shape[2]
    pass_block = None  # is a List of speakers where each element is a list of numpy: one for pass and one for block
    eval_dict = {'pesq': 0, 'pass_snr': 0, 'block_ratio': 0, 'in_sir': 0, 'out_sir': 0, 'sir_gain': 0}
    # amplify mic wav to (-0.5,0.5) range
    max_amp = np.max(np.abs(sum(files_dict['speakers']))) * 2
    max_amp = 1 if max_amp < 1e-6 else max_amp
    files_dict['speakers'] = [sp / max_amp for sp in files_dict['speakers']]

    if not (is_saved or is_evaluated):
        return eval_dict

    # normal reconstruction
    mixed_ref = sum(files_dict['speakers'])  # mix all the speakers to get the reference mic
    z_ref = our_stft([mixed_ref])            # STFT of ref mic
    masks = masks[:, :z_ref.shape[1], :]     # cut masks to match z-ref (remove padding)
    masked = masks * z_ref                   # apply mask
    rec_zones = our_istft(masked)            # reconstruct separated speakers

    if is_evaluated:
        z = our_stft(files_dict['speakers'])  # STFT of all input speakers
        pass_block = []
        for sp_id in range(num_speakers):
            cur_z = z[:, :, sp_id]
            speaker_zone = files_dict['zone_dict'][sp_id]
            # mask wanted zone
            pass_mask = masks[:, :, speaker_zone]
            pass_masked = pass_mask * cur_z
            # mask for unwanted zones
            block_mask = sum([masks[:, :, i] for i in range(num_zones) if i != speaker_zone])
            block_masked = block_mask * cur_z
            # reconstruct masked STFTs
            masked = np.stack((pass_masked, block_masked), axis=2)
            pass_block.append((our_istft(masked)))

        # apply evaluations and save them in a dictionary
        eval_dict = evaluate(files_dict, rec_zones, pass_block)

    #  save separated files
    if is_saved:
        save_results(save_path, files_dict, mixed_ref, rec_zones, pass_block=pass_block, eval_dict=eval_dict)

    return eval_dict


def save_results(save_path: str,  files_dict: dict, mixed_ref: nparr,
                 rec_zones: List[nparr], pass_block=None, eval_dict=None):
    # initialize some variables
    num_zones = len(rec_zones)
    fs = 16_000

    # create folders
    dir_path = pjoin(save_path, '_'.join(files_dict['file_names']))
    os.makedirs(dir_path, exist_ok=True)

    # save original files
    wav_write(pjoin(dir_path, 'mix.wav'), mixed_ref, fs)   # save mixed file
    for sp_id, sp in enumerate(files_dict['speakers']):   # save separated files
        wav_write(pjoin(dir_path, 'input_speaker_{}.wav'.format(sp_id)), sp, fs)

    # translate zone dict to speaker dict and save source separated speakers
    zone_dict = files_dict['zone_dict']                             # maps speakers to zones
    speaker_dict = ['zone_{}'.format(i) for i in range(num_zones)]  # maps zones to speakers
    for sp_id, speakers_zone in enumerate(zone_dict):
        speaker_dict[speakers_zone] += '_speaker_{}'.format(sp_id)

    # save separated reconstructed zones
    for zone_id, zone in enumerate(rec_zones):
        wav_write(pjoin(dir_path, speaker_dict[zone_id] + '.wav'), zone, fs)

    # save pass/block
    if pass_block is not None:
        # save pass and block wavs
        for sp_id, speaker_pb in enumerate(pass_block):
            wav_write(pjoin(dir_path, 'speaker_{}_pass.wav'.format(sp_id)), speaker_pb[0], fs)
            wav_write(pjoin(dir_path, 'speaker_{}_block.wav'.format(sp_id)), speaker_pb[1], fs)
        # save evaluations of  specific file
        eval_log = open(pjoin(dir_path, 'file_evaluation_log.txt'), "w")
        for k in eval_dict.keys():
            log_lines = ['speaker_{}: {} = {} \n'.format(sp_id, k, val) for sp_id, val in enumerate(eval_dict[k])]
            eval_log.writelines(log_lines)
        eval_log.close()


def evaluate(files_dict: dict, rec_zones: List[nparr], pass_block: List[List[nparr]]):
    # initialize some variables
    fs = 16_000
    num_speakers = len(files_dict['speakers'])
    evaluations = {'pesq': np.zeros(num_speakers),
                   'pass_snr': np.zeros(num_speakers),
                   'block_ratio': np.zeros(num_speakers),
                   'in_sir': np.zeros(num_speakers),
                   'out_sir': np.zeros(num_speakers),
                   'sir_gain': np.zeros(num_speakers)}
    zone_dict = files_dict['zone_dict']

    # calculate evaluations
    for sp_id, sp in enumerate(files_dict['speakers']):
        # pesq
        cur_zone = rec_zones[zone_dict[sp_id]][:len(sp)]  # current reconstructed result List with len num speakers
        _, pesq = pm.pesq(sp, cur_zone, fs)
        # block ration
        block_ratio = 10*np.log(np.sum(np.abs(pass_block[sp_id][1])) / np.sum(np.abs(sp)))
        # pass snr
        normed_pass = energy_norm(pass_block[sp_id][0][:len(sp)], sp)
        snr = pm.fwSNRseg(sp, normed_pass, fs)
        # input sir
        other_sp_std = np.std(sum([t_sp for t_id, t_sp in enumerate(files_dict['speakers']) if t_id != sp_id]))
        in_sir = np.std(sp) / other_sp_std
        in_sir = 20 * np.log10(in_sir)
        # output sir
        other_rec_std = np.std(sum([t_sp for t_id, t_sp in enumerate(list(zip(*pass_block))[1]) if t_id != sp_id]))
        out_sir = np.std(pass_block[sp_id][0]) / other_rec_std
        out_sir = 20 * np.log10(out_sir)
        # sir gain
        sir_gain = out_sir - in_sir
        # add results to the dictionary
        evaluations['pesq'][sp_id] = np.round_(pesq, 2)
        evaluations['block_ratio'][sp_id] = np.round_(block_ratio, 2)
        evaluations['pass_snr'][sp_id] = np.round(snr, 2)
        evaluations['in_sir'][sp_id] = np.round(in_sir, 2)
        evaluations['out_sir'][sp_id] = np.round(out_sir, 2)
        evaluations['sir_gain'][sp_id] = np.round(sir_gain)

    return evaluations


def energy_norm(target: nparr, ref: nparr):
    gain = np.sum(ref**2) / np.sum(target**2)
    return target * gain


def wav_write(file_name: str, signal: nparr, fs: int):
    logger = logging.getLogger('my_logger')
    max_val = np.max(np.abs(signal))
    if max_val > 1:
        logger.warning('Clipped while saving. value {} in file {}'.format(max_val, file_name))
    sf.write(file_name, signal, fs)

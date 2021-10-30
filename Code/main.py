import sys
from train import train
from separate import separate
import os
from os.path import join as pjoin
import logging
from datetime import datetime


def get_logger(logger_name, file_name):
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(file_name)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger


def main():

    # directories
    output_dir = '../output'     # Folder to save outputs (Created by program)
    train_dir = '../data/TRAIN'  # Directory contiaing train set - .wav files (Created by user)
    test_dir = '../data/TEST'    # Directory contiaing test set - .wav files (Created by user)
    rir_dir = '../RIRs'          # Directory of Simulation rirs .mat file (Created by user)

    runs = [
        # run1
        {'run_name': 'demo_run', 'test_out_name': 'test_results', 'rir_name': 'demo', 'micN': 9,
         'zoneN': 2, 'spN': 2, 'batch': 128, 'lr': 1e-3, 'perm_skip': 0, 'seg_len': 100, 'epochs': 30, 'sc_step': 10,
         'sc_gamma': 0.5, 'train': True, 'test': True, 'files2save': 5, 'evaluate': True, 'is_simulation': True, 'old_model': None}
        # run2 ...
    ]
    for i, run in enumerate(runs):
        # create required directories
        cur_out_dir = pjoin(output_dir, run['run_name'])
        os.makedirs(cur_out_dir, exist_ok=True)
        train_rir = pjoin(rir_dir, 'train_' + run['rir_name'] + '.mat')
        test_rir = pjoin(rir_dir, 'test_' + run['rir_name'] + '.mat')
        model_path = pjoin(cur_out_dir, 'trained_model', 'unet_model.pt')  # created by train
        test_out_dir = pjoin(cur_out_dir, run['test_out_name'])

        # logging
        logger = get_logger(logger_name='my_logger', file_name=pjoin(cur_out_dir, 'log.txt'))
        now = datetime.now()
        logger.info('Run {}/{} Started - {}'.format(i, len(runs), now.strftime("%d/%m/%Y %H:%M:%S")))

        if run['train']:
            train(cur_out_dir, train_dir, train_rir, mic_num=run['micN'], zone_num=run['zoneN'], sp_num=run['spN'],
                  batch_size=run['batch'], perm_skip=run['perm_skip'], seg_len=run['seg_len'], learning_rate=run['lr'],
                  num_epochs=run['epochs'], sched_step_size=run['sc_step'], sched_gamma=run['sc_gamma'],
                  is_simulation=run['is_simulation'], old_model=run['old_model'])

        if run['test']:
            separate(test_out_dir, test_dir, test_rir, model_path, mic_num=run['micN'], zone_num=run['zoneN'],
                     sp_num=run['spN'], perm_skip=run['perm_skip'], seg_len=run['seg_len'],
                     save_num=run['files2save'], is_evaluated=run['evaluate'], is_simulation=run['is_simulation'])

    logger.info('\nProgram Finished Successfully. Yey!')
    logger.info('-'*30)


if __name__ == '__main__':
    main()


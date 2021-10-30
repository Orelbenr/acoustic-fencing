import torch
from torch.utils.data import DataLoader
from helpers.data import TestAudioDataset, RealTestAudioDataset
from helpers.unet_model import UNet
from helpers.synthesis_and_evaluation import *
import logging
logger = logging.getLogger('my_logger')


def separate(output_dir, test_dir, test_rir, model_path, mic_num, zone_num, sp_num, perm_skip, seg_len, save_num,
             is_evaluated, is_simulation):
    # create requires directories
    os.makedirs(output_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, 'test_results')
    # os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create dataset and data loaders
    if is_simulation:
        test_dataset = TestAudioDataset(test_dir, test_rir, perm_skip=perm_skip, seg_len=seg_len, seed=2021)
    else:
        test_dataset = RealTestAudioDataset(test_dir, zone_num, sp_num, perm_skip=perm_skip, seg_len=seg_len, seed=2021)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)

    # Load Model
    net = UNet((mic_num - 1) * 2, zone_num, False).double()
    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Generate masks and separate
    masks = []
    file_counter = 0
    mean_eval_dict = {}
    var_eval_dict = {}
    logger.info('\nStart Testing')
    logger.info('-' * 20)
    for data, files, eof in test_loader:
        data = data.to(device)

        cur_mask = net(data)
        masks.append(cur_mask)

        if eof.item() is True:
            # move torch to numpy and count the file
            file_counter += 1
            mask = torch.cat(masks, dim=3)
            mask = mask.detach().cpu().numpy().squeeze()
            mask = np.moveaxis(mask, 0, -1)

            files['speakers'] = [i.numpy().squeeze() for i in files['speakers']]
            files['zone_dict'] = files['zone_dict'].numpy().squeeze()
            files['file_names'] = [i[0] for i in files['file_names']]

            # print some logs
            if (file_counter % 100) == 0:
                logger.info('evaluated {} files'.format(file_counter))
            if file_counter <= save_num:
                logger.info('saving file {} / {} ...'.format(file_counter, save_num))

            # save and evaluate
            cur_eval = synthesis(files, mask, output_dir,
                                 is_saved=(file_counter <= save_num), is_evaluated=is_evaluated)
            for k in cur_eval.keys():
                mean_eval_dict[k] = mean_eval_dict.get(k, 0) + cur_eval[k]
                var_eval_dict[k] = var_eval_dict.get(k, 0) + cur_eval[k]**2

            # re initialize masks
            masks = []

    # save evaluation
    if is_evaluated:
        logger.info('Finished Testing')
        logger.info('Saving evaluation logger...')
        eval_log = open(pjoin(output_dir, 'evaluation_log.txt'), "w")
        for k in mean_eval_dict.keys():
            mean_eval_dict[k] = mean_eval_dict[k] / file_counter
            var_eval_dict[k] = np.sqrt(var_eval_dict[k] / file_counter - mean_eval_dict[k]**2)  # this is std not variance
            log_lines = ['speaker_{}: {} = {} +- {} \n'.format(sp_id, k, val, var_eval_dict[k][sp_id]) for sp_id, val in enumerate(mean_eval_dict[k])]
            eval_log.writelines(log_lines)
        eval_log.close()


if __name__ == '__main__':
    # Folder to save outputs
    OUTPUT_DIR = r'..\Acustic_Fencing\output'
    # Folder of the test wav files
    TEST_DIR = r'..\Acustic_Fencing\resources\temp_test'
    # The test rir mat file name and path
    TEST_RIR = r'..\Acustic_Fencing\resources\rir_samples.mat'
    # Save model file name and path
    MODEL_PATH = r'..\Acustic_Fencing\output\trained_model\unet_model.pt'
    logging.basicConfig(level=logging.INFO)
    
    # separate(OUTPUT_DIR, TEST_DIR, TEST_RIR, MODEL_PATH, mic_num=9, zone_num=2, sp_num=2,
    #          perm_skip=100, seg_len=100, save_num=2, is_evaluated=True, is_simulation=True)
    #
    separate(OUTPUT_DIR, TEST_DIR, None, MODEL_PATH, mic_num=9, zone_num=2, sp_num=2,
             perm_skip=0, seg_len=100, save_num=2, is_evaluated=True, is_simulation=False)
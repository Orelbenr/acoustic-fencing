import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helpers.data import AudioDataset, RealAudioDataset
from helpers.unet_model import UNet
import logging

logger = logging.getLogger('my_logger')


def train(output_dir, train_dir, train_rir, mic_num, zone_num, sp_num, batch_size, perm_skip, seg_len=100,
          learning_rate=1e-1, num_epochs=1, sched_step_size=7, sched_gamma=0.1, is_simulation=True, old_model=None):
    # create requires directories
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'trained_model')
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('training on: ' + str(device))

    # create dataset and data loaders
    if is_simulation:
        train_dataset = AudioDataset(train_dir, train_rir, is_train=True, train_ratio=0.8,
                                     perm_skip=perm_skip, seg_len=seg_len, seed=2021)
    else:
        train_dataset = RealAudioDataset(train_dir, zone_num, sp_num, is_train=True, train_ratio=0.8,
                                         perm_skip=perm_skip, seg_len=seg_len, seed=2021)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)

    if is_simulation:
        valid_dataset = AudioDataset(train_dir, train_rir, is_train=False, train_ratio=0.8,
                                     perm_skip=perm_skip, seg_len=seg_len, seed=2021)
    else:
        valid_dataset = RealAudioDataset(train_dir, zone_num=2, sp_num=2, is_train=False, train_ratio=0.8,
                                     perm_skip=perm_skip, seg_len=seg_len, seed=2021)

    valid_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    logger.info('REMEMBER TO FIX VALID LOADER!!!!!!')

    dataloaders = {'train': train_loader,
                   'val': valid_loader}

    # Load Model
    net = UNet((mic_num - 1) * 2, zone_num, False).double()
    net = net.to(device)
    if old_model is not None:
        net.load_state_dict(torch.load(old_model, map_location=device))

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)

    # Do the Magic
    model, loss_vec, acc_vec = train_model(net, criterion, optimizer, scheduler, dataloaders,
                                           device, save_path, num_epochs)

    # Plot
    plot_loss(save_path, loss_vec, 'Loss')
    plot_loss(save_path, acc_vec, 'Accuracy')


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, save_path, num_epochs=25):
    since = time.time()
    last_time = since

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    loss_vec = {'train': [], 'val': []}
    acc_vec = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{} , lr = {:.0e}'.format(epoch, num_epochs - 1, scheduler.get_last_lr()[0]))
        logger.info('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            counter = 0
            for data, tags in dataloaders[phase]:
                counter += tags.size(0)

                data = data.to(device)
                tags = tags.to(device).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, tags)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == tags)  # type: torch.tensor
                # print
                if (counter % 100) == 0:
                    logger.info('Epoch {}, Iteration {} - {} Loss: {:.4f}'.format(
                        epoch, counter, phase, running_loss / counter))

            if counter == 0:
                raise ValueError('There was no data in the data loader. try reducing perm_skip.')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / counter
            epoch_acc = running_corrects.double().item() / (counter * tags.size(1) * tags.size(2))

            loss_vec[phase].append(epoch_loss)
            acc_vec[phase].append(epoch_acc)

            time_elapsed = time.time() - last_time
            last_time = time.time()
            if phase == 'train':
                logger.info('Epoch training finished ({:.0f}m {:.0f}s)'.format(
                    time_elapsed // 60, time_elapsed % 60))
                logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                logger.info('Starting validation...')

            # deep copy the model
            if phase == 'val':
                logger.info('Epoch validation finished ({:.0f}m {:.0f}s)'.format(
                    time_elapsed // 60, time_elapsed % 60))
                logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    logger.info('Validation improved - Saving model...')
                    torch.save(best_model_wts, os.path.join(save_path, 'unet_model.pt'))
                    torch.save(best_model_wts, os.path.join(save_path, f'model_backup_epoch{epoch}'))

        logger.info('')

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val loss: {:4f}'.format(best_loss))

    # save losses
    np.save(os.path.join(save_path, 'losses'), loss_vec)
    np.save(os.path.join(save_path, 'accuracies'), loss_vec)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_vec, acc_vec


def plot_loss(save_path, losses, title):
    plt.figure()
    plt.plot(losses['train'], label='Training')
    plt.plot(losses['val'], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend(frameon=False)
    plt.title('{} by Epochs'.format(title))
    fig_name = os.path.join(save_path, '{}_graph.png'.format(title))
    plt.savefig(fig_name)


if __name__ == '__main__':
    # Folder to save outputs
    OUTPUT_DIR = r'..\Acustic_Fencing\output'
    # Folder of the train wav files
    TRAIN_DIR = r'..\Acustic_Fencing\resources\real_temp' 
    # The train rir mat file name and path
    TRAIN_RIR = r'..\Acustic_Fencing\resources\rir_samples.mat'

    # train(OUTPUT_DIR, TRAIN_DIR, TRAIN_RIR, mic_num=9, zone_num=2, sp_num=2,
    #       batch_size=128, perm_skip=100, seg_len=100, learning_rate=1e-3, num_epochs=100, sched_step_size=101,
    #       sched_gamma=0.5, is_simulation=True)
    logging.basicConfig(level=logging.INFO)
    train(OUTPUT_DIR, TRAIN_DIR, None, mic_num=9, zone_num=2, sp_num=2,
          batch_size=3, perm_skip=0, seg_len=100, learning_rate=1e-3, num_epochs=3, sched_step_size=101,
          sched_gamma=0.5, is_simulation=False, old_model=None)

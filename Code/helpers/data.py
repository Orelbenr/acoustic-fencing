import os
from os.path import join as pjoin
import scipy.io as sio
from itertools import permutations, product
from torch.utils.data import DataLoader, IterableDataset
from helpers.preprocessing import *
import itertools


class AudioDataset(IterableDataset):

    def __init__(self, data_dir, rir_path, is_train=True, train_ratio=0.8, perm_skip=100, seg_len=100, seed=2021):
        super(AudioDataset).__init__()
        self.is_train = is_train
        self.seg_len = seg_len
        self.rng = np.random.default_rng(seed)
        self.perm_skip = perm_skip

        # Find wavs and load rirs
        self.wav_list = [pjoin(root, file) for root, dirs, files in os.walk(data_dir) for file in files
                         if file.endswith('.wav')]
        rirs_mat = sio.loadmat(rir_path)
        self.H = rirs_mat['H']
        self.zone_dict = rirs_mat['zone_dict']

        # Split train and validation
        train_len = int(np.ceil(train_ratio * len(self.wav_list)))
        self.wav_list = self.wav_list[:train_len] if is_train else self.wav_list[train_len:]

        train_len = int(np.ceil(train_ratio * self.H.shape[3]))
        self.H = self.H[:, :, :, :train_len] if is_train else self.H[:, :, :, train_len:]
        self.zone_dict = self.zone_dict[:train_len, :] if is_train else self.zone_dict[train_len:, :]

        # Generate Permutations
        self.perm = permutations(list(range(len(self.wav_list))), self.H.shape[2])
        # Initialize generator
        self.mini_batch_gen = TrainMiniBatchGenerator(self.wav_list, self.perm, self.H,
                                                      self.zone_dict, self.rng, self.perm_skip, self.seg_len)

    def __iter__(self):
        return iter(self.mini_batch_gen)


class TrainMiniBatchGenerator:
    def __init__(self, wav_list, perm, H, zone_dict, rng, perm_skip=100, seg_len=100):
        self.wav_list = wav_list
        self.perm = iter(perm)
        self.H = H
        self.zone_dict = zone_dict
        self.seg_len = seg_len
        self.rng = rng
        self.perm_skip = perm_skip
        self.data = 0
        self.tags = 0

    def __iter__(self):
        self.next_file()
        self.seg_idx = 0
        self.seg_num = self.data.shape[1] // self.seg_len
        return self

    def __next__(self):
        """
        return data, tag
        fixed data\tag size!
        cycle the end of data from the start
        """
        if self.seg_idx >= self.seg_num:
            self.next_file()
            self.seg_idx = 0
            self.seg_num = self.data.shape[1] // self.seg_len

        seg_start = self.seg_idx * self.seg_len
        seg_end = (self.seg_idx + 1) * self.seg_len
        self.seg_idx += 1
        return (np.moveaxis(self.data[:, seg_start:seg_end, :], -1, 0),
                self.tags[:, seg_start:seg_end])

    def next_file(self):
        # choose files and rirs
        rand_next(self.perm, self.perm_skip, self.rng)
        cur_sample_idx = next(self.perm)
        rir_idx = self.rng.integers(low=0, high=self.H.shape[3])

        # load files
        speakers = load_example(cur_sample_idx, self.wav_list, self.H, rir_idx)

        # create tags and data
        self.tags = create_tags(speakers, self.zone_dict[rir_idx, :])
        self.data = create_example(awgn(speakers))

        # loop back to fill seg len
        remainder = self.data.shape[1] % self.seg_len
        if remainder > 0:
            cycle_len = self.seg_len - remainder
            self.data = np.concatenate((self.data, self.data[:, :cycle_len, :]), axis=1)
            self.tags = np.concatenate((self.tags, self.tags[:, :cycle_len]), axis=1)


class TestAudioDataset(IterableDataset):
    def __init__(self, data_dir, rir_path, perm_skip=100, seg_len=100, seed=2022):
        super(TestAudioDataset).__init__()
        self.seg_len = seg_len
        self.rng = np.random.default_rng(seed)
        self.perm_skip = perm_skip

        self.wav_list = [pjoin(root, file) for root, dirs, files in os.walk(data_dir) for file in files
                         if file.endswith('.wav')]
        self.file_names = [os.path.basename(file) for file in self.wav_list]

        rirs_mat = sio.loadmat(rir_path)
        self.H = rirs_mat['H']
        self.zone_dict = rirs_mat['zone_dict']

        self.perm = permutations(list(range(len(self.wav_list))), self.H.shape[2])

        self.mini_batch_gen = TestMiniBatchGenerator(self.wav_list, self.file_names, self.perm, self.H,
                                                     self.zone_dict, self.rng, self.perm_skip, self.seg_len)

    def __iter__(self):
        return iter(self.mini_batch_gen)


class TestMiniBatchGenerator:
    """
    return data, file_name, EOF,
    and use with batch_size = 1 or None (whatever work)
    this way the data can vary in length
    """

    def __init__(self, wav_list, file_names, perm, H, zone_dict, rng, perm_skip=100, seg_len=100):
        self.wav_list = wav_list
        self.file_names = file_names
        self.perm = iter(perm)
        self.H = H
        self.zone_dict = zone_dict
        self.seg_len = seg_len
        self.rng = rng
        self.perm_skip = perm_skip
        self.current_files = {'file_names': [], 'zone_dict': [], 'speakers': []}
        self.data = 0

    def __iter__(self):
        self.next_file()
        self.seg_idx = 0
        self.seg_num = int(np.ceil(self.data.shape[1] / self.seg_len))
        self.EOF = False
        return self

    def __next__(self):
        if self.seg_idx >= self.seg_num:
            self.next_file()
            self.seg_idx = 0
            self.seg_num = int(np.ceil(self.data.shape[1] / self.seg_len))
            self.EOF = False

        seg_start = self.seg_idx * self.seg_len
        seg_end = (self.seg_idx + 1) * self.seg_len
        self.seg_idx += 1

        data_seg = np.moveaxis(self.data[:, seg_start:seg_end, :], -1, 0)
        if seg_end >= self.data.shape[1]:
            data_seg = np.moveaxis(self.data[:, seg_start:, :], -1, 0)
            self.EOF = True

        return data_seg, self.current_files, self.EOF

    def next_file(self):
        # choose files and rirs
        rand_next(self.perm, self.perm_skip, self.rng)
        cur_sample_idx = next(self.perm)
        rir_idx = self.rng.integers(low=0, high=self.H.shape[3])
        self.current_files['file_names'] = [self.file_names[i] for i in cur_sample_idx]
        self.current_files['zone_dict'] = self.zone_dict[rir_idx, :]
        # load files
        speakers = load_example(cur_sample_idx, self.wav_list, self.H, rir_idx)
        speakers = awgn(speakers)
        self.current_files['speakers'] = [speakers[i][0] for i in range(len(speakers))]
        # create data
        self.data = create_example(speakers)

        # pad to fill seg len
        remainder = self.data.shape[1] % self.seg_len
        if remainder > 0:
            pad_len = self.seg_len - remainder
            self.data = np.pad(self.data, ((0, 0), (0, pad_len), (0, 0)), 'constant')


class RealAudioDataset(IterableDataset):

    def __init__(self, data_dir, zone_num, sp_num, is_train=True,
                 train_ratio=0.8, perm_skip=0, seg_len=100, seed=2021):
        super(RealAudioDataset).__init__()
        self.is_train = is_train
        self.seg_len = seg_len
        self.rng = np.random.default_rng(seed)
        self.perm_skip = perm_skip

        # Find wavs and load rirs
        # Train/Zone1/Sentence1/mic1.wav
        self.wav_list = [[[pjoin(data_dir, zone, sentence, mic) for mic in os.listdir(pjoin(data_dir, zone, sentence))]
                          for sentence in os.listdir(pjoin(data_dir, zone))]
                         for zone in os.listdir(data_dir)]
        # Sort the Files so Mic9 is first and then Mic1,2,3..,8
        for zone in self.wav_list:
            for s_id, sentence in enumerate(zone):
                zone[s_id].sort()
                zone[s_id].append(zone[s_id].pop(0))

        # Split train and validation
        for zone_idx, zone in enumerate(self.wav_list):
            train_len = int(np.ceil(train_ratio * len(zone)))
            if is_train:
                self.wav_list[zone_idx] = self.wav_list[zone_idx][:train_len]
            else:
                self.wav_list[zone_idx] = self.wav_list[zone_idx][train_len:]

        # Generate Permutations
        max_len = max([len(zone) for zone in self.wav_list])
        self.perm = product(range(max_len), repeat=sp_num)
        # Initialize generator
        self.mini_batch_gen = RealTrainMiniBatchGenerator(self.wav_list, self.perm, sp_num, zone_num,
                                                          self.rng, self.perm_skip, self.seg_len)

    def __iter__(self):
        return iter(self.mini_batch_gen)


class RealTrainMiniBatchGenerator:
    def __init__(self, wav_list, perm, sp_num, zone_num, rng, perm_skip=0, seg_len=100):
        self.wav_list = wav_list
        self.perm = iter(perm)
        self.seg_len = seg_len
        self.zone_num = zone_num
        self.sp_num = sp_num
        self.rng = rng
        self.perm_skip = perm_skip
        self.data = 0
        self.tags = 0

    def __iter__(self):
        self.next_file()
        self.seg_idx = 0
        self.seg_num = self.data.shape[1] // self.seg_len
        return self

    def __next__(self):
        """
        return data, tag
        fixed data\tag size!
        cycle the end of data from the start
        """
        if self.seg_idx >= self.seg_num:
            self.next_file()
            self.seg_idx = 0
            self.seg_num = self.data.shape[1] // self.seg_len

        seg_start = self.seg_idx * self.seg_len
        seg_end = (self.seg_idx + 1) * self.seg_len
        self.seg_idx += 1
        return (np.moveaxis(self.data[:, seg_start:seg_end, :], -1, 0),
                self.tags[:, seg_start:seg_end])

    def next_file(self):
        # choose files and rirs
        cur_sample_idx = next(self.perm)
        rand_next(self.perm, self.perm_skip, self.rng)
        zone_dict = self.rng.choice(self.zone_num, size=self.sp_num, replace=False)

        # load files
        speakers = load_example(cur_sample_idx, self.wav_list, zone_dict=zone_dict)

        # create tags and data
        self.tags = create_tags(speakers, zone_dict)
        self.data = create_example(awgn(rand_sir(speakers, 3)))

        # loop back to fill seg len
        remainder = self.data.shape[1] % self.seg_len
        if remainder > 0:
            cycle_len = self.seg_len - remainder
            self.data = np.concatenate((self.data, self.data[:, :cycle_len, :]), axis=1)
            self.tags = np.concatenate((self.tags, self.tags[:, :cycle_len]), axis=1)


class RealTestAudioDataset(IterableDataset):
    def __init__(self, data_dir, zone_num, sp_num, perm_skip=0, seg_len=100, seed=2022):
        super(RealTestAudioDataset).__init__()
        self.seg_len = seg_len
        self.rng = np.random.default_rng(seed)
        self.perm_skip = perm_skip

        self.wav_list = [[[pjoin(data_dir, zone, sentence, mic) for mic in os.listdir(pjoin(data_dir, zone, sentence))]
                          for sentence in os.listdir(pjoin(data_dir, zone))]
                         for zone in os.listdir(data_dir)]
        # Sort the Files so Mic9 is first and then Mic1,2,3..,8
        for zone in self.wav_list:
            for s_id, sentence in enumerate(zone):
                zone[s_id].sort()
                zone[s_id].append(zone[s_id].pop(0))

        self.file_names = [[sentence[0].split(os.sep)[-2] for sentence in zone] for zone in self.wav_list]

        max_len = max([len(zone) for zone in self.wav_list])
        self.perm = product(range(max_len), repeat=sp_num)

        self.mini_batch_gen = RealTestMiniBatchGenerator(self.wav_list, self.file_names, self.perm, sp_num,
                                                         zone_num, self.rng, self.perm_skip, self.seg_len)

    def __iter__(self):
        return iter(self.mini_batch_gen)


class RealTestMiniBatchGenerator:
    """
    return data, file_name, EOF,
    and use with batch_size = 1 or None (whatever work)
    this way the data can vary in length
    """

    def __init__(self, wav_list, file_names, perm, sp_num, zone_num, rng, perm_skip=0, seg_len=100):
        self.wav_list = wav_list
        self.file_names = file_names
        self.perm = iter(perm)
        self.zone_num = zone_num
        self.sp_num = sp_num
        self.seg_len = seg_len
        self.rng = rng
        self.perm_skip = perm_skip
        self.current_files = {'file_names': [], 'zone_dict': [], 'speakers': []}
        self.data = 0

    def __iter__(self):
        self.next_file()
        self.seg_idx = 0
        self.seg_num = int(np.ceil(self.data.shape[1] / self.seg_len))
        self.EOF = False
        return self

    def __next__(self):
        if self.seg_idx >= self.seg_num:
            self.next_file()
            self.seg_idx = 0
            self.seg_num = int(np.ceil(self.data.shape[1] / self.seg_len))
            self.EOF = False

        seg_start = self.seg_idx * self.seg_len
        seg_end = (self.seg_idx + 1) * self.seg_len
        self.seg_idx += 1

        data_seg = np.moveaxis(self.data[:, seg_start:seg_end, :], -1, 0)
        if seg_end >= self.data.shape[1]:
            data_seg = np.moveaxis(self.data[:, seg_start:, :], -1, 0)
            self.EOF = True

        return data_seg, self.current_files, self.EOF

    def next_file(self):
        # choose files and rirs
        cur_sample_idx = next(self.perm)
        rand_next(self.perm, self.perm_skip, self.rng)
        zone_dict = self.rng.choice(self.zone_num, size=self.sp_num, replace=False)
        pick_sentences = lambda z, s: self.file_names[zone_dict[z]][s % len(self.file_names[zone_dict[z]])]
        self.current_files['file_names'] = [pick_sentences(zone_id, sentence_id)
                                            for zone_id, sentence_id in enumerate(cur_sample_idx)]
        self.current_files['zone_dict'] = zone_dict
        # load files
        speakers = load_example(cur_sample_idx, self.wav_list, zone_dict=zone_dict)
        speakers = rand_sir(speakers, 4)
        self.current_files['speakers'] = [speakers[i][0] for i in range(len(speakers))]
        # create data
        self.data = create_example(speakers)

        # pad to fill seg len
        remainder = self.data.shape[1] % self.seg_len
        if remainder > 0:
            pad_len = self.seg_len - remainder
            self.data = np.pad(self.data, ((0, 0), (0, pad_len), (0, 0)), 'constant')


def rand_next(iterator, avg_next_num, rng):
    if avg_next_num > 0:
        n = rng.integers(low=0, high=avg_next_num) + avg_next_num // 2
        next(itertools.islice(iterator, n, n), None)


if __name__ == '__main__':

    DIR = r'....\darpa-timit-acousticphonetic-continuous-speech\data\TRAIN\DR1'
    RIR_PATH = r'..\Acustic_Fencing\resources\test_1r_2s_2z.mat'

    REAL_DIR = r'..\Acustic_Fencing\resources\real_temp'
    train_dataset = RealAudioDataset(REAL_DIR, perm_skip=0, sp_num=2, zone_num=2)
    train_loader = DataLoader(train_dataset, batch_size=3)

    counter = 0
    for data, tags in train_loader:
        print(data.shape, tags.shape)
        counter += 1
        if counter > 20:
            break
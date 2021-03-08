import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os, random
train_on_gpu = torch.cuda.is_available()
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')
CPU = torch.device('cpu')

class HParams:
    def __init__(self):
        self.n_fft = 1024
        self.hop_length = 160
        self.num_mels = 80
        self.win_size = 400
        self.conv_in = self.n_fft // 2 + 1

        # Mel and Linear spectrograms normalization/scaling and clipping
        self.signal_normalization = True
        self.allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
        self.symmetric_mels = False  # Whether to scale the data to be symmetric around 0
        self.max_abs_value = 4  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]

        self.filter_width = 3
        self.layers = 4
        self.conv_channels = 16
        self.num_att_units = 128
        self.vad_reg_weight = 1e-5
        self.vad_decay_learning_rate = True #boolean, determines if the learning rate will follow an exponential decay
        self.vad_start_decay = 50000 #Step at which learning decay starts
        self.vad_decay_steps = 35000 #Determines the learning rate decay slope (UNDER TEST)
        self.vad_decay_rate = 0.8 #learning rate decay   (UNDER TEST)
        self.vad_initial_learning_rate = 1e-3 #starting learning rate
        self.vad_final_learning_rate = 5e-6 #minimal learning rate

        self.vad_adam_beta1 = 0.9 #AdamOptimizer beta1 parameter
        self.vad_adam_beta2 = 0.999 #AdamOptimizer beta2 parameter
        self.vad_adam_epsilon = 1e-6 #AdamOptimizer Epsilon parameter

        self.vad_zoneout_rate=0.1 #zoneout rate for all LSTM cells in the network
        self.vad_dropout_rate=0.5 #dropout rate for all convolutional layers + prenet
        self.vad_moving_average_decay=.99

        self.vad_clip_gradients=True #whether to clip gradients

        self.sample_rate = 16000
        self.w = 19
        self.u = 9
        self.batch_size = 512

hparams = HParams()


def data_transform_targets_bdnn(inputs, w, u, DEVICE=DEVICE):
    neighbors_1 = torch.arange(-w, -u, u)
    neighbors_2 = torch.tensor([-1, 0, 1])
    neighbors_3 = torch.arange(1+u, w+1, u)

    neighbors = torch.cat((neighbors_1, neighbors_2, neighbors_3), dim=0)

    pad_size = 2*w + inputs.shape[0]
    pad_inputs = torch.zeros(pad_size).to(DEVICE)
    pad_inputs[0:inputs.shape[0]] = inputs

    trans_inputs = torch.vstack([torch.unsqueeze(torch.roll(pad_inputs, int(-1*neighbors[i]), dims=0)
                                [0:inputs.shape[0]], dim=0) for i in range(neighbors.shape[0])])
    trans_inputs = trans_inputs.permute([1, 0])

    return trans_inputs


def data_transform(inputs, w=hparams.w, u=hparams.u, min_abs_value=1e-7, DEVICE=DEVICE):
    neighbors_1 = torch.arange(-w, -u, u)
    neighbors_2 = torch.tensor([-1, 0, 1])
    neighbors_3 = torch.arange(1+u, w+1, u)

    neighbors = torch.cat((neighbors_1, neighbors_2, neighbors_3), dim=0)

    pad_size = 2*w + inputs.shape[0]
    pad_inputs = torch.ones((pad_size, inputs.shape[1])).to(DEVICE) * min_abs_value
    pad_inputs[0:inputs.shape[0], :] = inputs

    trans_inputs = torch.vstack([torch.unsqueeze(torch.roll(pad_inputs, int(-1*neighbors[i]), dims=0)
                                [0:inputs.shape[0], :], dim=0) for i in range(neighbors.shape[0])])
    trans_inputs = trans_inputs.permute([1, 0, 2])

    return trans_inputs


def bdnn_prediction(logits, threshold=0.5, w=hparams.w, u=hparams.u):
    bdnn_batch_size = int(logits.shape[0] + 2 * w)
    result = np.zeros((bdnn_batch_size, 1))
    indx = np.arange(bdnn_batch_size) + 1
    indx = data_transform_targets_bdnn(torch.from_numpy(indx), w, u, DEVICE=CPU)
    indx = indx[w:(bdnn_batch_size - w), :]
    indx_list = np.arange(w, bdnn_batch_size - w)

    for i in indx_list:
        indx_temp = np.where((indx - 1) == i)
        pred = logits[indx_temp]
        pred = np.sum(pred) / pred.shape[0]
        result[i] = pred

    result = result[w:-w]
    soft_result = np.float32(result)
    result = np.float32(result) >= threshold

    return result.astype(np.float32), soft_result


def prediction(targets, pipenet_output, postnet_output, w=hparams.w, u=hparams.u):
    # _, soft_prediction = bdnn_prediction(postnet_output.size[0], F.sigmoid(postnet_output))
    pipenet_prediction = torch.round(F.sigmoid(pipenet_output))
    postnet_prediction = torch.round(F.sigmoid(postnet_output))

    pipenet_targets = targets.clone().detach()
    # targets = torch.max(targets, dim=-1)
    raw_indx = int(np.floor(int(2 * (w - 1) / u + 3) / 2))  # =3
    raw_labels = pipenet_targets[:, raw_indx]
    raw_labels = torch.reshape(raw_labels, shape=(-1, 1))

    postnet_accuracy = torch.mean(postnet_prediction.eq_(raw_labels))
    pipenet_accuracy = torch.mean(pipenet_prediction.eq_(raw_labels))
    return postnet_accuracy, pipenet_accuracy


class SpeechDataset(Dataset):
    def __init__(self, metadata_filename, metadata, hparams):
        super().__init__()
        self._hparams = hparams
        self._mel_dir = os.path.join(os.path.dirname(metadata_filename), 'mels')
        self._metadata = metadata
        timesteps = sum([int(x[2]) for x in self._metadata])
        sr = hparams.sample_rate
        hours = timesteps / sr / 3600
        print('Loaded metadata for {} examples_SE ({:.2f} hours)'.format(len(self._metadata), hours))
        self.len_ = len(self._metadata)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        meta = self._metadata[index]
        start_frame = int(meta[4])
        end_frame = int(meta[5])

        mel_input = np.load(os.path.join(self._mel_dir, meta[1]))
        mel_input = np.divide((mel_input - mel_input.min(axis=0)), (mel_input.max(axis=0) - mel_input.min(axis=0)))
        target = np.asarray([0] * (len(mel_input)))
        target[start_frame:end_frame] = 1

        mel_input = torch.as_tensor(mel_input, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        mel_input = data_transform(mel_input, self._hparams.w, self._hparams.u, mel_input.min(), DEVICE=torch.device('cpu'))
        target = data_transform_targets_bdnn(target, self._hparams.w, self._hparams.u, DEVICE=torch.device('cpu'))
        mel_input = mel_input[self._hparams.w:-self._hparams.w, :, :]
        target = target[self._hparams.w:-self._hparams.w]
        return mel_input, target


def train_valid_split(metadata_filename, hparams, test_size=0.05, seed=0):
    with open(metadata_filename, encoding='utf-8') as f:
        data = [line.strip().split('|') for line in f]
        timesteps = sum([int(x[2]) for x in data])
        sr = hparams.sample_rate
        hours = timesteps / sr / 3600
        print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(data), hours))
    random.seed(seed)
    training_idx = []
    validation_idx = []

    aug_list = [[idx, x] for idx, x in enumerate(data) if len(x[0].split('_')) > 4]
    snr_list = list(set([x[0].split('_')[4] for idx, x in aug_list]))
    # print('snr_list:', snr_list)
    noise_list = list(set([x[0].split('_')[5] for idx, x in aug_list]))
    # print('noise_list:', noise_list)
    clean_idx = list(range(len(data)))
    for idx, x in aug_list:
        clean_idx.remove(idx)

    random.shuffle(clean_idx, random.random)
    validation_split_idx = int(np.ceil(test_size * len(clean_idx)))
    training_idx += clean_idx[validation_split_idx:]
    validation_idx += clean_idx[0:validation_split_idx]

    meta_idx = {}
    for n in noise_list:
        meta_idx[n] = {}
        for s in snr_list:
            meta_idx[n][s] = [idx for idx, x in aug_list if x[0].split('_')[5] == n and x[0].split('_')[4] == s]
            random.shuffle(meta_idx[n][s])
            validation_split_idx = int(np.ceil(test_size * len(meta_idx[n][s])))
            training_idx += meta_idx[n][s][validation_split_idx:]
            validation_idx += meta_idx[n][s][0:validation_split_idx]

    if bool(set(training_idx) & set(validation_idx)):
        raise ValueError('Training and validation data are overlapped!')

    random.shuffle(training_idx)
    random.shuffle(validation_idx)

    return data, training_idx, validation_idx


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: {:.3f}Million, Trainable: {:.3f}Million'.format(
            total_num / 1_000_000, trainable_num / 1_000_000))
    # return {'Total': total_num, 'Trainable': trainable_num}


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
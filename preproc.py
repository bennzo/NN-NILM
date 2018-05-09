import numpy as np
import utilities
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch

opt = {
    'Fs' : 2*650+50,
    'sample_time' : 0.1,
    'noise' : True,
    'noise_percentage' : 0.1,
    'train_test_ratio' : 0.9
}

class fftDataset(Dataset):
    def __init__(self, data=None, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        return {'data': torch.from_numpy(data),'label': torch.from_numpy(label)}

def data_init(data_dir):
    data = np.array([])
    labels = np.array([])
    raw_data, raw_labels = utilities.load_sum(data_dir)
    data_size = np.shape(raw_labels)[0]

    Fs = opt['Fs']
    win = int(opt['Fs']*opt['sample_time'])

    for i in range(0, data_size, win):
        # Filter out windows with a change in active appliances
        label = raw_labels[i]
        skip = False
        for t in range(win):
            if np.all(label == raw_labels[t]):
                skip == True
                break
        if skip:
            continue

        I_t = raw_data[i:i + win]
        I_fft = fft(I_t,opt['noise'])
        I_fft_amp, I_fft_phase = fft_amp_phase(I_fft)

        if i == 0:
            data = fft2input(I_fft_amp, I_fft_phase, Fs)
            labels = label
        else:
            data = np.vstack((data, fft2input(I_fft_amp, I_fft_phase, Fs)))
            labels = np.vstack((labels, label))

    n = np.shape(data)[0]
    shuffle_idx = np.random.RandomState(0).permutation(np.array(range(0,n)))

    train_idx = shuffle_idx[0:int(n*opt['train_test_ratio'])]
    test_idx = shuffle_idx[int(n*opt['train_test_ratio']):n]

    train_data_unscaled = data[train_idx].astype(float)
    train_labels = labels[train_idx]

    test_data_unscaled = data[test_idx].astype(float)
    test_labels = labels[test_idx]

    return train_data_unscaled, train_labels, test_data_unscaled, test_labels

def fft(I_t, noise=False):
    n = len(I_t)  # length of the signal
    if noise:
        I_t += np.random.normal(0,1,n)*(I_t*opt['noise_percentage'])
    I_fft = np.fft.fft(I_t)/(n/2)   # fft computing and normalization
    I_fft = I_fft[range(int(n/2))]  # one side frequency range
    return I_fft

def fft_amp_phase(I_fft, threshold=0.1):
    I_fft_amp = np.abs(I_fft)                   # calculate amplitude
    I_fft_phase = np.angle(I_fft)               # calculate phase
    I_fft_phase[I_fft_amp < threshold] = 0      # non maximum supression
    I_fft_amp[I_fft_amp < threshold] = 0
    I_fft_phase = I_fft_phase * (360 / (2 * np.pi))+90
    return I_fft_amp, I_fft_phase

def fft2input(I_fft_amp, I_fft_phase, Fs):
    harmonies = np.array([1, 3, 5, 7, 9, 11, 13])*round((len(I_fft_amp)*2)/(Fs/50))
    harmonies = harmonies.astype(int)
    IN = np.concatenate((I_fft_amp[harmonies],I_fft_phase[harmonies]))
    return IN


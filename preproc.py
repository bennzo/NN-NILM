import numpy as np
import itertools
import random
import utilities as util
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch

def data_init(data_dir):
    data = np.array([])
    labels = np.array([])
    raw_data, raw_labels = util.load_sum(data_dir)
    data_size = np.shape(raw_labels)[0]

    Fs = util.preproc_config['Fs']
    win = int(util.preproc_config['Fs']*util.preproc_config['sample_time'])

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
        I_fft = fft(I_t,util.preproc_config['noise'])
        I_fft_amp, I_fft_phase = fft_amp_phase(I_fft)

        if i == 0:
            data = fft2input(I_fft_amp, I_fft_phase, Fs)
            labels = label
        else:
            data = np.vstack((data, fft2input(I_fft_amp, I_fft_phase, Fs)))
            labels = np.vstack((labels, label))

    n = np.shape(data)[0]
    shuffle_idx = np.random.RandomState(0).permutation(np.array(range(0,n)))

    train_idx = shuffle_idx[0:int(n*util.preproc_config['train_test_ratio'])]
    test_idx = shuffle_idx[int(n*util.preproc_config['train_test_ratio']):n]

    train_data_unscaled = data[train_idx].astype(float)
    train_labels = labels[train_idx]

    test_data_unscaled = data[test_idx].astype(float)
    test_labels = labels[test_idx]

    return train_data_unscaled, train_labels, test_data_unscaled, test_labels


def data_init_syn(data_dir):
    raw_data, raw_labels = util.load_sum(data_dir)

    num_classes = util.nn_config['num_classes']
    data_size = raw_data.size
    sig_len = data_size//(2**num_classes)

    manual = 0
    if manual:
        train_comb = [
            0b10000,
            0b01000,
            0b00100,
            #0b00010,
            0b00001,
            # 0b10100,
             0b10010,
            # 0b00011,
            # 0b00111,
            # 0b10101,
            # 0b10111,
            # 0b10011,
            # 0b11111,
            # 0b10101,
            # 0b11011
        ]
        test_comb = [
             0b10000,
             0b01100,
           # 0b00100,
          #  0b00010,
             0b11000,
             0b10100,
      #      0b00111
        ]
    else:
        train_perm = 32
        test_perm = 12
        train_comb = ['0b' + s for s in ["".join(seq) for seq in itertools.product("01", repeat=5)]]
        random.shuffle(train_comb)
        test_comb = ['0b' + s for s in ["".join(seq) for seq in itertools.product("01", repeat=5)]]
        random.shuffle(test_comb)
        for _ in range(32 - train_perm):
            train_comb.pop()
        for _ in range(32 - test_perm):
            test_comb.pop()

    raw_train_data = np.array([], dtype=float)
    raw_train_label = np.array([]).reshape((0,num_classes))
    raw_test_data = np.array([], dtype=float)
    raw_test_label = np.array([]).reshape((0,num_classes))

    for comb in train_comb:
        if not manual:
            comb = int(comb, 2)
        raw_train_data = np.concatenate((raw_train_data, raw_data[comb*sig_len:(1 + comb)*sig_len]))
        raw_train_label = np.vstack((raw_train_label, raw_labels[comb*sig_len:(1 + comb)*sig_len]))
    train_data_unscaled, train_labels = data2input(raw_train_data, raw_train_label)

    # ------------- Predefined test -------------- #
    for comb in test_comb:
        if not manual:
            comb = int(comb, 2)
        raw_test_data = np.concatenate((raw_test_data, raw_data[comb*sig_len:(1 + int(comb))*sig_len]))
        raw_test_label = np.vstack((raw_test_label, raw_labels[comb*sig_len:(1 + comb)*sig_len]))
    test_data_unscaled, test_labels   = data2input(raw_test_data, raw_test_label)
    # -------------------------------------------- #

    # ------------- Holdout Validation test ------------- #
    # n = train_data_unscaled.shape[0]
    # shuffle_idx = np.random.RandomState(0).permutation(np.array(range(0,n)))
    # train_idx = shuffle_idx[0:int(n*util.preproc_config['train_test_ratio'])]
    # test_idx = shuffle_idx[int(n*util.preproc_config['train_test_ratio']):n]
    #
    # test_data_unscaled = train_data_unscaled[test_idx].astype(float)
    # test_labels = train_labels[test_idx]
    # train_data_unscaled = train_data_unscaled[train_idx].astype(float)
    # train_labels = train_labels[train_idx]
    # --------------------------------------------------- #

    return train_data_unscaled, train_labels, test_data_unscaled, test_labels


def fft(I_t, noise=False):
    n = len(I_t)  # length of the signal
    if noise:
        I_t += np.random.normal(0,1,n)*(I_t*util.preproc_config['noise_percentage'])
    I_fft = np.fft.fft(I_t)/(n/2)                                                       # fft computing and normalization
    I_fft = I_fft[range(int(n/2))]                                                      # one side frequency range
    return I_fft


def fft_amp_phase(I_fft):
    I_fft_amp = np.abs(I_fft)                                                           # calculate amplitude
    I_fft_phase = np.angle(I_fft)                                                       # calculate phase
    I_fft_phase[I_fft_amp < util.preproc_config['threshold']] = 0                       # non maximum supression
    I_fft_amp[I_fft_amp < util.preproc_config['threshold']] = 0
    I_fft_phase = I_fft_phase * (360 / (2 * np.pi))
    return I_fft_amp, I_fft_phase


def fft2input(I_fft_amp, I_fft_phase, Fs):
    harmonies = np.array([1, 3, 5, 7, 9, 11, 13])*round((len(I_fft_amp)*2)/(Fs/50))
    harmonies = harmonies.astype(int)
    IN = np.concatenate((I_fft_amp[harmonies],I_fft_phase[harmonies]))
    return IN


def data2input(raw_data,raw_labels):
    data_size = raw_data.size
    Fs = util.preproc_config['Fs']
    #win = int(util.preproc_config['Fs']*util.preproc_config['sample_time'])
    win = 128

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
        I_fft = fft(I_t,util.preproc_config['noise'])
        I_fft_amp, I_fft_phase = fft_amp_phase(I_fft)

        if i == 0:
            data = fft2input(I_fft_amp, I_fft_phase, Fs)
            labels = label
        else:
            data = np.vstack((data, fft2input(I_fft_amp, I_fft_phase, Fs)))
            labels = np.vstack((labels, label))
    return data,labels


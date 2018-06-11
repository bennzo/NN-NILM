import numpy as np
import itertools
import random
import utilities as util
import preproc as pp

train_comb_default = [
    '0b00000',
    '0b00001',
    '0b00010',
    '0b00011',
    '0b00100',
    '0b00101',
    '0b00110',
    '0b00111',
    '0b01000',
    '0b01001',
    '0b01010',
    '0b01011',
    '0b01100',
    '0b01101',
    '0b01110',
    '0b01111',
    '0b10000',
    '0b10001',
    '0b10010',
    '0b10011',
    '0b10100',
    '0b10101',
    '0b10110',
    '0b10111',
    '0b11000',
    '0b11001',
    '0b11010',
    '0b11011',
    '0b11100',
    '0b11101',
    '0b11110',
    '0b11111'
            ]
test_comb_default = [
    '0b00000',
    '0b00001',
    '0b00010',
    '0b00011',
    '0b00100',
    '0b00101',
    '0b00110',
    '0b00111',
    '0b01000',
    '0b01001',
    '0b01010',
    '0b01011',
    '0b01100',
    '0b01101',
    '0b01110',
    '0b01111',
    '0b10000',
    '0b10001',
    '0b10010',
    '0b10011',
    '0b10100',
    '0b10101',
    '0b10110',
    '0b10111',
    '0b11000',
    '0b11001',
    '0b11010',
    '0b11011',
    '0b11100',
    '0b11101',
    '0b11110',
    '0b11111'
            ]


# Data initialization for simulated signals
# the output is a composition of signal summarization across all on/off combinations
def data_init_simulated(data_dir):
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
        I_fft = pp.fft(I_t,util.preproc_config['noise'])
        I_fft_amp, I_fft_phase = pp.fft_amp_phase(I_fft)

        if i == 0:
            data = pp.fft2input(I_fft_amp, I_fft_phase, Fs)
            labels = label
        else:
            data = np.vstack((data, pp.fft2input(I_fft_amp, I_fft_phase, Fs)))
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

# Data initialization for measured inputs
# the output is a composition of signal summarization across predefined on/off combinations
def data_init_comb(data_dir, train_comb=train_comb_default, test_comb=test_comb_default):
    raw_data, raw_labels = util.load_sum(data_dir)

    num_classes = util.nn_config['num_classes']
    data_size = raw_data.size
    sig_len = data_size//(2**num_classes)

    manual = True
    if not manual:
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
        comb = int(comb, 2)
        raw_train_data = np.concatenate((raw_train_data, raw_data[comb*sig_len:(1 + comb)*sig_len]))
        raw_train_label = np.vstack((raw_train_label, raw_labels[comb*sig_len:(1 + comb)*sig_len]))
    train_data_unscaled, train_labels = data2input(raw_train_data, raw_train_label)

    # ------------- Predefined test -------------- #
    for comb in test_comb:
        comb = int(comb, 2)
        raw_test_data = np.concatenate((raw_test_data, raw_data[comb*sig_len:(1 + comb)*sig_len]))
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

# Data initialization for measured signals
# the output is a composition of signal summarization across all on/off combinations
def data_init_measured(data_dir):
    raw_data, raw_labels = util.load_sum(data_dir)
    train_data_unscaled, train_labels = data2input(raw_data, raw_labels)

    # ------------- Holdout Validation test ------------- #
    n = train_data_unscaled.shape[0]
    shuffle_idx = np.random.RandomState(0).permutation(np.array(range(0,n)))
    train_idx = shuffle_idx[0:int(n*util.preproc_config['train_test_ratio'])]
    test_idx = shuffle_idx[int(n*util.preproc_config['train_test_ratio']):n]

    test_data_unscaled = train_data_unscaled[test_idx].astype(float)
    test_labels = train_labels[test_idx]
    train_data_unscaled = train_data_unscaled[train_idx].astype(float)
    train_labels = train_labels[train_idx]
    # --------------------------------------------------- #

    return train_data_unscaled, train_labels, test_data_unscaled, test_labels

def data2input(raw_data,raw_labels):
    data_size = raw_data.size
    Fs = util.preproc_config['Fs']
    win = int(util.preproc_config['Fs']*util.preproc_config['sample_time']) # For Simulated data
    #win = 128

    data = np.zeros((int(np.ceil(data_size/win)),14))
    labels = np.zeros((int(np.ceil(data_size/win)),5))

    k = 0
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
        I_fft = pp.fft(I_t,util.preproc_config['noise'])
        I_fft_amp, I_fft_phase = pp.fft_amp_phase(I_fft)

        data[k, :] = pp.fft2input(I_fft_amp, I_fft_phase, Fs)
        labels[k, :] = label
        k = k + 1
        # if i == 0:
        #     data = pp.fft2input(I_fft_amp, I_fft_phase, Fs)
        #     labels = label
        # else:
        #     data = np.vstack((data, pp.fft2input(I_fft_amp, I_fft_phase, Fs)))
        #     labels = np.vstack((labels, label))


    return data,labels

def signal2input(signal):
    I_fft = pp.fft(signal, util.preproc_config['noise'])
    I_fft_amp, I_fft_phase = pp.fft_amp_phase(I_fft)
    input = pp.fft2input(I_fft_amp, I_fft_phase, util.preproc_config['Fs'])
    return input

def data_init_comb_test(data_dir, train_perm, test_perm):
    raw_data, raw_labels = util.load_sum(data_dir)

    train_comb = train_comb_default[0:train_perm]
    test_comb = test_comb_default[0:test_perm]

    num_classes = util.nn_config['num_classes']
    data_size = raw_data.size
    sig_len = data_size//(2**num_classes)

    manual = True
    if not manual:
        train_comb = ['0b' + s for s in ["".join(seq) for seq in itertools.product("01", repeat=5)]]
        random.shuffle(train_comb)
        test_comb = ['0b' + s for s in ["".join(seq) for seq in itertools.product("01", repeat=5)]]
        random.shuffle(test_comb)
        for _ in range(32 - train_perm):
            train_comb.pop()
        for _ in range(32 - test_perm):
            test_comb.pop()

    raw_train_data = np.zeros((sig_len*train_perm),dtype=float)
    raw_train_label = np.zeros((sig_len*train_perm,num_classes), dtype=float)
    raw_test_data = np.zeros((sig_len * test_perm), dtype=float)
    raw_test_label = np.zeros((sig_len * test_perm, num_classes), dtype=float)

    i = 0
    for comb in train_comb:
        comb = int(comb, 2)
        raw_train_data[i:i+sig_len] = raw_data[comb * sig_len:(1 + comb) * sig_len]
        raw_train_label[i:i+sig_len,:] = raw_labels[comb * sig_len:(1 + comb) * sig_len]
        i = i + sig_len
    train_data_unscaled, train_labels = data2input(raw_train_data, raw_train_label)

    # ------------- Predefined test -------------- #
    i = 0
    for comb in test_comb:
        comb = int(comb, 2)
        raw_test_data[i:i+sig_len] = raw_data[comb * sig_len:(1 + comb) * sig_len]
        raw_test_label[i:i+sig_len,:] = raw_labels[comb * sig_len:(1 + comb) * sig_len]
        i = i + sig_len
    test_data_unscaled, test_labels   = data2input(raw_test_data, raw_test_label)
    # -------------------------------------------- #

    return train_data_unscaled, train_labels, test_data_unscaled, test_labels

def data_init_for_plot(data_dir, test_perm):
    raw_data, raw_labels = util.load_sum(data_dir)
    test_comb = test_comb_default[0:test_perm]
    #random.shuffle(test_comb)
    # test_comb = np.sort(test_comb[0:test_perm])
    num_classes = util.nn_config['num_classes']
    data_size = raw_data.size
    sig_len = data_size//(2**num_classes)

    manual = True
    if not manual:
        test_comb = ['0b' + s for s in ["".join(seq) for seq in itertools.product("01", repeat=5)]]
        random.shuffle(test_comb)
        for _ in range(32 - test_perm):
            test_comb.pop()

    raw_test_data = np.zeros((sig_len * test_perm), dtype=float)
    raw_test_label = np.zeros((sig_len * test_perm, num_classes), dtype=float)

    i = 0
    for comb in test_comb:
        comb = int(comb, 2)
        raw_test_data[i:i+sig_len] = raw_data[comb * sig_len:(1 + comb) * sig_len]
        raw_test_label[i:i+sig_len,:] = raw_labels[comb * sig_len:(1 + comb) * sig_len]
        i = i + sig_len
    test_data_unscaled, test_labels   = data2input(raw_test_data, raw_test_label)

    # raw_test_data = np.array([], dtype=float)
    # raw_test_label = np.array([]).reshape((0,num_classes))

    # ------------- Predefined test -------------- #
    # for comb in test_comb_default:
    #     comb = int(comb, 2)
    #     raw_test_data = np.concatenate((raw_test_data, raw_data[comb*sig_len:(1 + comb)*sig_len]))
    #     raw_test_label = np.vstack((raw_test_label, raw_labels[comb*sig_len:(1 + comb)*sig_len]))
    # test_data_unscaled, test_labels   = data2input(raw_test_data, raw_test_label)
    # -------------------------------------------- #

    return test_data_unscaled, test_labels
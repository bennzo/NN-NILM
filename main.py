import numpy as np
import matplotlib.pyplot as plt
import utilities
import preproc
import network
import data

test_comb = [
    '0b11100',
    '0b11011',
    '0b11111',
    '0b10101',
    '0b00100',
    '0b10111'
            ]

train_data_unscaled, train_labels, test_data_unscaled, test_labels = data.data_init_comb('data/real_world_new/', train_comb=test_comb, test_comb=test_comb)
_,_, NET = network.train(train_data_unscaled, train_labels, test_data_unscaled, test_labels)
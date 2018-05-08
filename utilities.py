import numpy as np
import preproc as pp


def gen_I(n):
    harmony_n = np.random.randint(2,8)
    f = np.array([1,3,5,7,9,11,13]) * 50  # frequency space

    # ------ Signal properties ------ #
    F = np.sort(np.random.choice(f, harmony_n, replace=False))      # frequency vector
    P = np.random.uniform(-1 / 2 * np.pi, 1 / 2 * np.pi, harmony_n) # phase vector
    A = np.sort(np.random.randint(1, 10, size=harmony_n))[::-1]     # amplitude vector

    # ------ Generate Signal ------ #
    Fs = pp.opt['Fs'];                                                                                  # sampling rate
    Ts = 1.0 / Fs;                                                                              # sampling interval
    T_fin = 200                                                                                # measuring window
    t = np.arange(0, T_fin, Ts)                                                                 # time vector
    I_t = np.array([np.sin(2 * np.pi * F * t[i] + P) for i in range(0, np.size(t))]).dot(A)     # signal

    # ------ Determine on/off ------ #
    #on = np.sort(np.random.choice(T_fin*Fs-1, 16, replace=False))
    # label = np.ones((T_fin*Fs))
    # for i in range(0,np.size(on),2):
    #     I_t[on[i]:on[i+1]] = 0
    #     label[on[i]:on[i+1]] = 0

    on = np.ones((T_fin*Fs), dtype=int)
    for i in range(0, (T_fin*Fs), int(T_fin*Fs/(2**n))*2):
        on[i:i+int(T_fin*Fs/(2**n))] = 0
    label = on
    I_t = I_t*on

    save_signal(n,F,A,P,I_t,label)
    return I_t, label

def gen_sum():
    S_sum, label_sum = gen_I(1)
    for i in range(2,6):
        temp_sum, temp_label = gen_I(i)
        S_sum += temp_sum
        label_sum = np.vstack((label_sum,temp_label))

    np.savetxt("signal_sum_val.txt", S_sum, fmt='%.7f', delimiter='\n')
    np.savetxt("signal_sum_label.txt", np.transpose(label_sum), fmt='%i', delimiter=',')


def save_signal(i,F,A,P,I,label):
    np.savetxt("signal_{}_prop.txt".format(i), (F, A, P), fmt='%.3f', delimiter=',')
    np.savetxt("signal_{}_val.txt".format(i), I, fmt='%.7f', delimiter='\n')
    np.savetxt("signal_{}_label.txt".format(i), label, fmt='%i', delimiter='\n')

def load_signal(i, folder_name):
    F, A, P = np.loadtxt(folder_name + "signal_{}_prop.txt".format(i), delimiter=',')
    I = np.loadtxt(folder_name + "signal_{}_val.txt".format(i))
    return F, A, P, I

def load_sum(folder_name):
    I = np.loadtxt(folder_name + "signal_sum_val.txt")
    I_label = np.loadtxt(folder_name + "signal_sum_label.txt", delimiter=',')
    return I, I_label

def compare_input(I1,I2):
    return np.sqrt(((I1 - I2)**2).sum())

if __name__ == "__main__":
    gen_sum()

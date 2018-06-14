import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import utilities as util
import data


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = nn.functional.relu(self.fc4(out))
        out = nn.functional.relu(self.fc5(out))
        out = nn.functional.sigmoid(self.fc6(out))
        return out

def train(train_data, train_labels, test_data, test_labels):
    # Pytorch datasets

    # Amp Only
    if (util.nn_config['input_size'] == 7):
        train_data = train_data[:,:7]
        test_data = test_data[:, :7]

    train_dataset = data_utils.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
    test_dataset = data_utils.TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

    train_loader = data_utils.DataLoader(train_dataset, batch_size=util.nn_config['batch_size'], shuffle=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=util.nn_config['batch_size'], shuffle=True)

    print("Initializing NN")
    net = Net(util.nn_config['input_size'], util.nn_config['num_classes'])

    # Loss and Optimizer
    criterion = nn.BCELoss()
    #criterion = nn.MultiLabelMarginLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=util.nn_config['learning_rate'], momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    train_accuracy_1 = np.zeros(util.nn_config['num_epochs'])
    # Train the Model
    for epoch in range(util.nn_config['num_epochs']):
        correct = 0
        total = 0
        for i, (signals, appliances) in enumerate(train_loader):
            # Convert torch tensor to Variable
            signals = Variable(signals)
            appliances = Variable(appliances)
            # for sig in signals:
            #     if (((sig[0:7]-test_input[0:7]).abs().sum() < 0.001).any()):
            #         aaa = 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(signals)
            total += appliances.size(0)
            correct += np.sum(np.all(appliances.data == torch.round(outputs.data), axis = 1))

            loss = criterion(outputs, appliances)
            loss.backward()
            optimizer.step()

        # print loss at the end of each batch
        # correct = 0
        # total = 0
        # for signals, appliances in train_loader:
        #     signals = Variable(signals)
        #     appliances = Variable(appliances)
        #     outputs = net(signals)
        #     total += appliances.size(0)
        #     correct += np.sum(np.all(appliances.data == torch.round(outputs.data), axis = 1))
        train_accuracy_1[epoch] = correct / total
        print('Accuracy of the network on the training set after the {0} epoch: {1:.2f} %'.format(epoch + 1, 100 * correct / total))

    # Test the Model
    correct = 0
    total = 0
    hamming_dist = 0
    for signals, appliances in test_loader:
        signals = Variable(signals)
        outputs = net(signals)
        total += appliances.size(0)
        hamming_dist += np.count_nonzero(appliances == torch.round(outputs.data))/util.nn_config['num_classes']
        correct += np.sum(np.all(appliances == torch.round(outputs.data), axis=1))

    test_hamming_loss = 100 * hamming_dist / total
    test_exact_match = 100 * correct / total
    print('Exact Match Accuracy of the network on the test set: {0:.2f} %'.format(test_exact_match))
    print('Hamming Accuracy of the network on the test set: {0:.2f} %'.format(test_hamming_loss))

    # Save the Model
    torch.save(net.state_dict(), 'model.pkl')

    return test_exact_match, test_hamming_loss, net

def disaggregate(signal):
    model = Net(util.nn_config['input_size'], util.nn_config['num_classes'])
    model.load_state_dict(torch.load('model.pkl'))
    model.eval()

    input = data.signal2input(signal)

    # Amp only
    if (util.nn_config['input_size'] == 7):
        input = input[:7]
    output = model(Variable(torch.from_numpy(input).float()))
    print(output.data.numpy().round().astype(int))

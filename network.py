import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import sklearn.preprocessing
import preproc
import matplotlib.pyplot as plt


# Hyper Parameters
input_size = 14
output_size = 5
num_classes = 5
num_epochs = 100
batch_size = 30
learning_rate = 1e-3

# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.sigmoid(self.fc3(out))
        return out


# Build data arrays
train_data, train_labels, test_data, test_labels = preproc.data_init('data\lab-noiseless-perfect\\')

# Scale data
# train_data = sklearn.preprocessing.scale(train_data, axis=0, with_std=False)
# test_data = sklearn.preprocessing.scale(test_data, axis=0, with_std=False)

# Pytorch datasets
train_dataset = data_utils.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
test_dataset = data_utils.TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Initializing NN")
net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

train_accuracy_1 = np.zeros(num_epochs)
# Train the Model
for epoch in range(num_epochs):
    for i, (signals, appliances) in enumerate(train_loader):
        # Convert torch tensor to Variable
        signals = Variable(signals)
        appliances = Variable(appliances)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(signals)
        loss = criterion(outputs, appliances)
        loss.backward()
        optimizer.step()
    # print loss at the end of each batch
    correct = 0
    total = 0
    for signals, appliances in train_loader:
        signals = Variable(signals)
        appliances = Variable(appliances)
        outputs = net(signals)
        total += appliances.size(0)
        correct += np.sum(np.all(appliances.data == torch.round(outputs.data), axis = 1))
    train_accuracy_1[epoch] = correct / total
    print('Accuracy of the network on the training set after the {0} epoch: {1:.2f} %'.format(epoch + 1, 100 * correct / total))

# Test the Model
correct = 0
total = 0
for signals, appliances in test_loader:
    signals = Variable(signals)
    outputs = net(signals)
    total += appliances.size(0)
    correct += np.sum(np.all(appliances == torch.round(outputs.data), axis=1))

print('Accuracy of the network on the test set: {0:.2f} %'.format(100 * correct / total))


# Save the Model
#torch.save(net.state_dict(), 'model_1.pkl')

# fig1 = plt.figure()
# p1 = fig1.add_subplot(111)
# p1.plot(range(0,110,10), noise_acc, 'b')
# p1.grid(True)
# p1.set_title('Accuracy for noise')
# p1.set_xlabel('Noise Percentage')
# p1.set_ylabel('Accuracy')
# plt.show()
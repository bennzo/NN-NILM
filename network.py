import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import preproc

# Hyper Parameters
input_size = 14
output_size = 5
num_classes = 5
num_epochs = 10
batch_size = 20
learning_rate = 1e-3

train_data, train_labels, test_data, test_labels = preproc.data_init('data\lab-noiseless\\')

# Independent datasets
# train_dataset = preproc.fftDataset(train_data, train_labels, transform=preproc.ToTensor())
# test_dataset = preproc.fftDataset(test_data, test_loader, transform=preproc.ToTensor())
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=False)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# Pytorch datasets
train_dataset = data_utils.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
test_dataset = data_utils.TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


print("running section a network")
net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

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
        _, predicted = torch.max(outputs.data, 1)
        total += appliances.size(0)
        correct += (predicted == appliances).sum()
    train_accuracy_1[epoch] = correct / total
    print('Accuracy of the network on the training set after the %d epoch: %d %%' % (epoch + 1, 100 * correct / total))

# Test the Model
correct = 0
total = 0
for signals, appliances in test_loader:
    signals = Variable(signals)
    outputs = net(signals)
    _, predicted = torch.max(outputs.data, 1)
    total += appliances.size(0)
    correct += (predicted == appliances).sum()
    total += appliances.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model_1.pkl')

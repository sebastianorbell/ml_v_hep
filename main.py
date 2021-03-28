from dataClass import jetDataset, imshow
import matplotlib.pyplot as plt
from torchCNN import Net
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


root = '/Users/sebastianorbell/PycharmProjects/ml_v_hep/data/numpy'

qcd_file = '/train_QCD_preproccesed_data.npy'
ww_file = '/train_WW_preproccesed_data.npy'
classes = ['QCD', 'WW']

transform = transforms.Compose(
    [transforms.ToTensor()])

'''transforms.Normalize((0.5), (0.5))])'''

trainDataset = jetDataset(root+qcd_file, root+ww_file, root, transform=transform)

trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=4,
                                          shuffle=True)


# get some random training images
dataiter = iter(trainloader)
data = dataiter.next()

# show images
labels = [classes[int(i)] for i in data['label'].numpy()]
imshow(np.squeeze(data['image'].numpy()), labels=labels)
# print labels
print(' '.join('%5s' % classes[int(data['label'][j])] for j in range(4)))


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

loss_list = []
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a DICT of {'inputs', 'labels'}
        inputs, labels = data['image'], data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_list.append(running_loss)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

plt.plot(loss_list)
plt.show()
print('Finished Training')

PATH = 'savedModels/jet.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(trainloader)
data = dataiter.next()

# show images
labels = [classes[int(i)] for i in data['label'].numpy()]
imshow(np.squeeze(data['image'].numpy()), labels=labels)
# print labels
print(' '.join('%5s' % classes[int(data['label'][j])] for j in range(4)))


net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(data['image'])

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data['image'], data['label']
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
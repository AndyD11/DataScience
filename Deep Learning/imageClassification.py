# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random 
import time

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# ==========================================
#    Load Training Data and Testing Data
# ==========================================

class_names = [name[11:] for name in glob.glob('data/train/*')]
class_names = dict(zip(range(len(class_names)), class_names))
print(class_names)

def img_norm(img):
    return 2 * (np.float32(img) / 255 - 0.5) # normalize img pixels to [-1, 1]

def load_dataset(path, img_size, num_per_class=-1, batch_num=1, shuffle=False, augment=False, is_color=False):
    
    data = []
    labels = []
    
    if is_color:
        channel_num = 3
    else:
        channel_num = 1
        
    # read images and resizing
    for id, class_name in class_names.items():
        img_path_class = glob.glob(path + class_name + '/*.jpg')
        if num_per_class > 0:
            img_path_class = img_path_class[:num_per_class]
        labels.extend([id]*len(img_path_class))
        for filename in img_path_class:
            if is_color:
                img = cv2.imread(filename)
            else:
                img = cv2.imread(filename, 0)
            
            # resize the image
            img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
            
            if is_color:
                img = np.transpose(img, [2, 0, 1])
            
            # norm pixel values to [-1, 1]
            data.append(img_norm(img))
    
     # norm data to zero-centered
    mean_img = np.mean(np.array(data), 0)
    data = data - mean_img
    data = [data[i] for i in range(data.shape[0])]
    
    # augment data
    if augment:
        aug_data = [np.flip(img, 1) for img in data]
        data.extend(aug_data)
        labels.extend(labels)

    # randomly permute (this step is important for training)
    if shuffle:
        bundle = list(zip(data, labels))
        random.shuffle(bundle)
        data, labels = zip(*bundle)
    
    # divide data into minibatches of TorchTensors
    if batch_num > 1:
        batch_data = []
        batch_labels = []
        
        print(len(data))
        print(batch_num)
        
        for i in range(int(len(data) / batch_num)):
            minibatch_d = data[i*batch_num: (i+1)*batch_num]
            minibatch_d = np.reshape(minibatch_d, (batch_num, channel_num, img_size[0], img_size[1]))
            batch_data.append(torch.from_numpy(minibatch_d))

            minibatch_l = labels[i*batch_num: (i+1)*batch_num]
            batch_labels.append(torch.LongTensor(minibatch_l))
        data, labels = batch_data, batch_labels 
    
    return zip(batch_data, batch_labels)

# load data into size (64, 64)
img_size = (64, 64)
batch_num = 50 # training sample number per batch 

# load training dataset
trainloader_small = list(load_dataset('data/train/', img_size, batch_num=batch_num, shuffle=True, augment=True))
train_num = len(trainloader_small)
print("Finish loading %d minibatches(=%d) of training samples." % (train_num, batch_num))

# load testing dataset
testloader_small = list(load_dataset('data/test/', img_size, num_per_class=100, batch_num=batch_num))
test_num = len(testloader_small)
print("Finish loading %d minibatches(=%d) of testing samples." % (test_num, batch_num))

# show some images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(npimg.shape) > 2:
        npimg = np.transpose(img, [1, 2, 0])
    plt.figure
    plt.imshow(npimg, 'gray')
    plt.show()
img, label = trainloader_small[0][0][11][0], trainloader_small[0][1][11]
label = int(np.array(label))
print(class_names[label])
imshow(img)

# ==========================================
#       Define Network Architecture
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(32*12*12, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = x.view(50, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn = Net().to(device)

loss_function = nn.CrossEntropyLoss()
gradients = optim.SGD(cnn.parameters(), lr=0.1)

# ==========================================
#         Optimize/Train Network
# ==========================================
training_start_time = time.time()

for epoch in range(50): 
    for data in trainloader_small:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        gradients.zero_grad()
        outputs = cnn(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        gradients.step()
            
training_end_time = time.time()
print("Train time: " + str(training_end_time-training_start_time))

# ==========================================
#            Evaluating Network
# ==========================================
testing_start_time = time.time()

total = 0
correct = 0
for data in testloader_small:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = cnn(inputs)
    _, predicted_labels = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted_labels == labels).sum().item()

testing_end_time = time.time()
print("Test time: " + str(testing_end_time-testing_start_time))

print('Accuracy of the network on the 1500 test images: %d %%' % (
    100 * correct / total))

# reload data with a larger size
img_size = (224, 224)
batch_num = 50 # training sample number per batch 

# load training dataset
trainloader_large = list(load_dataset('data/train/', img_size, batch_num=batch_num, shuffle=True, augment=False, is_color=True))
train_num = len(trainloader_large)
print("Finish loading %d minibatches(=%d) of training samples." % (train_num, batch_num))

# load testing dataset
testloader_large = list(load_dataset('data/test/', img_size, num_per_class=100, batch_num=batch_num, is_color=True))
test_num = len(testloader_large)
print("Finish loading %d minibatches(=%d) of testing samples." % (test_num, batch_num))

# ==========================================
#       Fine-Tune Pretrained Network
# ==========================================
alexnet = models.alexnet(pretrained=True)

alexnet.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 15),
        )

alexnet = alexnet.to(device)

loss_function = nn.CrossEntropyLoss()
gradients = optim.SGD(alexnet.parameters(), lr=0.01)

# ==========================================
#         Optimize/Train Network
# ==========================================
training_start_time = time.time()

for epoch in range(30): 
    for data in trainloader_large:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        gradients.zero_grad()
        outputs = alexnet(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        gradients.step()
            
training_end_time = time.time()
print("Train time: " + str(training_end_time-training_start_time))

# ==========================================
#            Evaluating Network
# ==========================================
testing_start_time = time.time()

total = 0
correct = 0
for data in testloader_large:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = alexnet(inputs)
    _, predicted_labels = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted_labels == labels).sum().item()

testing_end_time = time.time()
print("Test time: " + str(testing_end_time-testing_start_time))

print('Accuracy of the network on the 1500 test images: %d %%' % (
    100 * correct / total))


from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch

#import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
#from torchsummary import summary
import matplotlib.pyplot as plt

from PIL import Image
import sys
import random
#from Utils import *
#from Models import *


# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#NOTE: Preprocessing and preparing data to be ready to be fed to the network!

#NOTE Set the path to the image file
#NOTE: Read the train set
data_dir1_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\F3' # PC directory
#data_dir1_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\F3' #Laptop directory
data_dir2_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\C3'
#data_dir2_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\C3' #Laptop directory
data_dir3_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\O1'
#data_dir3_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\O1' #Laptop directory
data_dir4_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\\ECGF3'
#data_dir4_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\ECGF3' #Laptop directory

#NOTE: Read the test set
data_dir1_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\F3' # PC directory
#data_dir1_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\F3' #Laptop directory
data_dir2_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\C3'
#data_dir2_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\C3' #Laptop directory
data_dir3_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\O1'
#data_dir3_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\O1' #Laptop directory
data_dir4_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\ECGF3'
#data_dir4_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\ECGF3' #Laptop directory
# Define the transformation to apply to the image
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # Resize the image to a specific size
    transforms.ToTensor(),  # Convert the image to a tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

#NOTE: Create the ImageFolder dataset for training set
dataset1_train = ImageFolder(root=data_dir1_train, transform=transform)
dataset2_train = ImageFolder(root=data_dir2_train, transform=transform)
dataset3_train = ImageFolder(root=data_dir3_train, transform=transform)
dataset4_train = ImageFolder(root=data_dir4_train, transform=transform)

#NOTE: Create the ImageFolder dataset for testing set
dataset1_test = ImageFolder(root=data_dir1_test, transform=transform)
dataset2_test = ImageFolder(root=data_dir2_test, transform=transform)
dataset3_test = ImageFolder(root=data_dir3_test, transform=transform)
dataset4_test = ImageFolder(root=data_dir4_test, transform=transform)

#classes = dataset1.classes
#print(classes)
#images = dataset1.imgs
#print(len(images))
# Example of accessing a specific image and label

#image_path, label = dataset1.imgs[0]
#print("Image path:", image_path[-11:-4])
#print("Label:", label)

sequence_length = 16
#NOTE: Merging the images to create a 12 channel image for training set
All_labels = []
All_images = []
minlen = min(len(dataset1_train), len(dataset2_train), len(dataset3_train), len(dataset4_train))
for item in range(int(100/sequence_length)):
    sequence_labels = []
    sequence_images = []
    for iter in range(item*sequence_length, (item+1)*sequence_length):
        img1 = dataset1_train[iter][0]
        img2 = dataset2_train[iter][0]
        img3 = dataset3_train[iter][0]
        img4 = dataset4_train[iter][0]
        if dataset1_train[iter][1] == dataset2_train[iter][1] == dataset3_train[iter][1] == dataset4_train[iter][1]:
            label = dataset1_train[iter][1]
        concatenated_image = torch.cat([img1, img2, img3, img4], dim=0)
        sequence_labels.append(label)
        sequence_images.append(concatenated_image)
    All_labels.append(torch.tensor(sequence_labels))
    All_images.append(torch.stack(sequence_images))
train_dataset = TensorDataset(torch.stack(All_images), torch.stack(All_labels))

#NOTE: Merging the images to create a 12 channel image for testing set
All_labels = []
All_images = []
minlen = min(len(dataset1_test), len(dataset2_test), len(dataset3_test), len(dataset4_test))
for item in range(int(100/sequence_length)):
    sequence_labels = []
    sequence_images = []
    for iter in range(item*sequence_length, (item+1)*sequence_length):
        img1 = dataset1_test[iter][0]
        img2 = dataset2_test[iter][0]
        img3 = dataset3_test[iter][0]
        img4 = dataset4_test[iter][0]
        if dataset1_test[iter][1] == dataset2_test[iter][1] == dataset3_test[iter][1] == dataset4_test[iter][1]:
            label = dataset1_test[iter][1]
        concatenated_image = torch.cat([img1, img2, img3, img4], dim=0)
        sequence_labels.append(label)
        sequence_images.append(concatenated_image)
    All_labels.append(torch.tensor(sequence_labels))
    All_images.append(torch.stack(sequence_images))
test_dataset = TensorDataset(torch.stack(All_images), torch.stack(All_labels))

#print(dataset_tupleslist[0][0].shape)
'''
class CustomDataset(Dataset):
    def __init__(self,  dataset_tupleslist):
        self.datset_tuples = dataset_tupleslist

    def __getitem__(self, index):
        image, label = self.datset_tuples[index]
        return image, label
    def __len__(self):
        return len(self.datset_tuples)
    
#convert the list of tuples into a custom dataset
train_dataset = CustomDataset(train_dataset_tupleslist)
test_dataset = CustomDataset(test_dataset_tupleslist)
'''

'''#NOTE::::: It has a big problem!!! It should choose the training and test sets from both classes, but it only choose first 80%
#from class 0 and last 20% from class 1 !!!! change it by reading training set and test sets individualy::: you can put training and test sets
# in different folders in the directory and read it to make them different from the begining! the best idea is this I think!
dataset_length = len(custom_dataset)
train_length = int(0.8 * dataset_length)  # 80% for training, 20% for testing
train_dataset = Subset(custom_dataset, range(train_length))
test_dataset = Subset(custom_dataset, range(train_length, dataset_length))'''

#create a dataloader using the custom datset
batch_size = 32
TrainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
TestLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


'''
for batch_data, batch_label in TrainLoader:

    print(batch_data)
    print(batch_label)
    print(batch_data.shape)
    break
# Iterate over the data loader to access the images and labels
'''
'''
sequences = {}
for images, labels in TrainLoader:
    subject_id = labels.item()
    if subject_id not in sequences:
        sequences[subject_id] = []

    sequences[subject_id].append(images)
    #print(sequences)


sequence_tensors = []
for subject_id, sequence in sequences.items():
    sequence_tensor = torch.stack(sequence, dim = 0)
    sequence_tensors.append(sequence_tensor)

input_tensor = torch.stack(sequence_tensors, dim = 0)
print(input_tensor.shape)
'''

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 1
num_channels = 12
image_height = 250
image_width = 125
#input_size = image_height * image_width * num_channels
hidden_size = 128
num_layers = 2
output_size = 2
learning_rate = 0.001

# Define the LSTM network
class LSTMNet(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, stride=1, padding=1):
        super(LSTMNet, self).__init__()
        # prelayer, fix the input size
        self.padding = nn.ZeroPad2d((0, 3, 3, 3))
        # convolutional layer
        # 1st layer of CNN
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2end layer of CNN
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3rd layer of CNN
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 4th layer of CNN
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 5th layer of CNN
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=stride, padding=padding)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 6th layer of CNN
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=stride, padding=padding)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_size = 2*4*1024
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, hidden_size, batch_first=True, num_layers= num_layers, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.max = nn.LogSoftmax()

    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0], x.shape[1], 1024, 4, 2).cuda()
        else:
            tmp = torch.zeros(x.shape[0], x.shape[1], 1024, 4, 2).cpu()
        for i in range(16): # sequence length
            input = x[:, i]
            #print('input shape:', input.shape)
            input = self.padding(input)

            input = self.relu1(self.conv1(input))
            input = self.pool1(input)

            input = self.relu2(self.conv2(input))
            input = self.pool2(input)

            input = self.relu3(self.conv3(input))
            input = self.pool3(input)

            input = self.relu4(self.conv4(input))
            input = self.pool4(input)

            input = self.relu5(self.conv5(input))
            input = self.pool5(input)
            
            input = self.relu6(self.conv6(input))
            tmp[:, i] = self.pool6(input)
            del input
        print(tmp.shape)
        input = tmp.reshape(x.shape[0], x.shape[1], 4 * 1024 * 2)
        del tmp
        #input = torch.cat((input[1:], input[2:]))
        print('new input:', input.shape)
        h0 = torch.zeros(num_layers, input.size(0), hidden_size).to(input.device)
        c0 = torch.zeros(num_layers, input.size(0), hidden_size).to(input.device)
        lstm_out, _ = self.lstm(input, (h0, c0))
        #print('LSTM output shape:', lstm_out.shape)
        output = self.fc(lstm_out[:, :, :])
        softmaxoutput = self.max(output)
        softmaxoutput = torch.argmax(softmaxoutput, dim=2)
        print('output shape:',output.shape)
        return output, softmaxoutput

# Create the LSTM network
model = LSTMNet(hidden_size, output_size, num_layers).to(device)
#summary(model, (12, 250, 125))
print('after summary')
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for images, labels in TrainLoader:
        print('checking the image shape before giving the input to the network:')
        print(images.shape)
        #images = images.view(-1, sequence_length, input_size).to(device) # changing the input images in the shape of the input to the
        # network
        labels = labels.to(device)
        #labels = labels.float()
        # Convert labels to LongTensor
        labels = torch.tensor(labels, dtype=torch.long)
        print(labels.shape)
        '''num_zeros = torch.sum(labels == 0).item()
        num_ones = torch.sum(labels == 1).item()
        if num_zeros > num_ones:
            labels = torch.zeros(int(batch_size/sequence_length),dtype=torch.int64)
        else:
            labels = torch.ones(int(batch_size/sequence_length),dtype=torch.int64)'''

        print('checking the image shape after resizing the input image before giving the input to the network:')
        print(images.shape)

        
        (predictions, softmaxoutput) = model(images)
        print(softmaxoutput)
        predictions = predictions.float()
        predictions = torch.tensor(predictions, requires_grad= True)
        predictions = predictions.view(-1, 2)
        print('pred:::',predictions)
        print(predictions.shape)
        print('True labels:', labels)
        loss = criterion(predictions, labels.view(-1))
        acc = torch.sum(softmaxoutput == labels).item() / (batch_size * sequence_length)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(TrainLoader):.4f}, Acc: {epoch_acc/len(TrainLoader):.4f}')

# Evaluate the model
def evaluate(model, loader):
    model.eval()
    epoch_acc = 0

    with torch.no_grad():
        for images, labels in loader:
            #images = images.view(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            labels = labels.float()

            '''num_zeros = torch.sum(labels == 0).item()
            num_ones = torch.sum(labels == 1).item()
            if num_zeros > num_ones:
                labels = torch.zeros(int(batch_size/sequence_length),dtype=torch.int64)
            else:
                labels = torch.ones(int(batch_size/sequence_length),dtype=torch.int64)'''

            (predictions, softmaxoutput) = model(images)
            acc = torch.sum(softmaxoutput == labels).item() / (batch_size * sequence_length)

            epoch_acc += acc

    return epoch_acc / len(loader)

test_acc = evaluate(model, TestLoader)
print(f'Test Accuracy: {test_acc:.4f}')


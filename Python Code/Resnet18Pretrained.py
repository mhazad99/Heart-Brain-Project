from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch

#import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix


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
#data_dir1_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\F3' # PC directory
data_dir1_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\F3' #Laptop directory
#data_dir2_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\C3'
data_dir2_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\C3' #Laptop directory
#data_dir3_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\O1'
data_dir3_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\O1' #Laptop directory
#data_dir4_train = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Train\\ECGF3'
data_dir4_train = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\ECGF3' #Laptop directory

#NOTE: Read the test set
#data_dir1_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\F3' # PC directory
data_dir1_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\F3' #Laptop directory
#data_dir2_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\C3'
data_dir2_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\C3' #Laptop directory
#data_dir3_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\O1'
data_dir3_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\O1' #Laptop directory
#data_dir4_test = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Spectrograms\Test\ECGF3'
data_dir4_test = 'D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Spectrograms\ECGF3' #Laptop directory

# Transform for resnet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

#NOTE: Merging the images to create a 12 channel image for training set
train_dataset_tupleslist = []
minlen = min(len(dataset1_train), len(dataset2_train), len(dataset3_train), len(dataset4_train))
for item in range(1000):
    img1 = dataset1_train[item][0]
    img2 = dataset2_train[item][0]
    img3 = dataset3_train[item][0]
    img4 = dataset4_train[item][0]
    if dataset1_train[item][1] == dataset2_train[item][1] == dataset3_train[item][1] == dataset4_train[item][1]:
        label = dataset1_train[item][1]
    concatenated_image = torch.cat([img1, img2, img3, img4], dim=0)
    train_dataset_tupleslist.append((concatenated_image, label))

#NOTE: Merging the images to create a 12 channel image for testing set
test_dataset_tupleslist = []
minlen = min(len(dataset1_test), len(dataset2_test), len(dataset3_test), len(dataset4_test))
for item in range(200):
    img1 = dataset1_test[item][0]
    img2 = dataset2_test[item][0]
    img3 = dataset3_test[item][0]
    img4 = dataset4_test[item][0]
    if dataset1_test[item][1] == dataset2_test[item][1] == dataset3_test[item][1] == dataset4_test[item][1]:
        label = dataset1_test[item][1]
    concatenated_image = torch.cat([img1, img2, img3, img4], dim=0)
    test_dataset_tupleslist.append((concatenated_image, label))

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


'''#NOTE::::: It has a big problem!!! It should choose the training and test sets from both classes, but it only choose first 80%
#from class 0 and last 20% from class 1 !!!! change it by reading training set and test sets individualy::: you can put training and test sets
# in different folders in the directory and read it to make them different from the begining! the best idea is this I think!
dataset_length = len(custom_dataset)
train_length = int(0.8 * dataset_length)  # 80% for training, 20% for testing
train_dataset = Subset(custom_dataset, range(train_length))
test_dataset = Subset(custom_dataset, range(train_length, dataset_length))'''

#create a dataloader using the custom dataset
batch_size = 64
TrainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
TestLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
num_channels = 12
image_height = 250
image_width = 125
input_size = image_height * image_width * num_channels
output_size = 2
learning_rate = 0.001

#Resnet18
resnet18 = models.resnet18(weights = True).to(device)
# Modify the first convolutional layer to accept 12 channels
num_input_channels = 12
resnet18.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=learning_rate, momentum= 0.9)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    resnet18.train()
    all_labels = []
    all_predicted = []
    for images, labels in TrainLoader:
        labels = labels.to(device)
        images = images.to(device)
        # Convert labels to LongTensor
        #labels = torch.tensor(labels, dtype=torch.long)
        optimizer.zero_grad()
        predictions = resnet18(images)
        predictions = torch.tensor(predictions, requires_grad = True)
        #predictions.clone().detach().requires_grad_(True)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        '''probabilities = F.softmax(predictions, dim=1)
        _, predicted = torch.max(probabilities, 1)  # Get class predictions'''
        
        predicted = predictions.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        all_labels.extend(labels)
        all_predicted.extend(predicted)
        #predicted = predicted.detach().numpy()
        #predictions.extend(predicted.cpu().numpy())
        #labels = labels.detach().numpy()
        #true_labels.extend(labels.cpu().numpy())
        epoch_loss += loss.item()
    # Calculate accuracy and confusion matrix at the end of each epoch
    accuracy = accuracy_score(all_labels, all_predicted)
    confusion = confusion_matrix(all_labels, all_predicted)

    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(TrainLoader):.4f}, Acc: {accuracy:.4f}')
    print(confusion)

# Evaluate the model
def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            images = images.to(device)
            #labels = torch.tensor(labels, dtype=torch.long)
            #labels = labels.to('cuda')
            #images = images.to('cuda')
            output = model(images)
            _, predicted = torch.max(output, 1)
            predicted = predicted.cpu().numpy()
            labels =  labels.cpu().numpy()
            all_labels.extend(labels)
            all_predicted.extend(predicted)

    accuracy = accuracy_score(all_labels, all_predicted)
    confusion = confusion_matrix(all_labels, all_predicted)

    return accuracy, confusion

test_acc, confusion = evaluate(resnet18, TestLoader)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
print('Test Confusion Matrix:')
print(confusion)

torch.save(resnet18.state_dict(), 'Myresnet18.pth')




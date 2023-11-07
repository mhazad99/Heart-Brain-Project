import os
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
import math
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import torchvision.models as models
import seaborn as sns
import wandb

wandb.init(project = "classificationwithcoherences")

def DataPreparation(data_dir):
    data_and_labels = []
    All_labels = []
    All_segments = []
    C3_Coherences = []
    F3_Coherences = []
    O1_Coherences = []
    C3_labels = []
    F3_labels = []
    O1_labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat') and 'C3' in root:
                if 'Control' in root:
                    label = 0
                elif 'Depression' in root:
                    label = 1
                mat_file = os.path.join(root, file)
                mat = loadmat(mat_file)
                for item in range(len(mat['MagCoherence'][0])):
                    MagCoherenceC3 = mat['MagCoherence'][0:256, item] ### This value is based on the frequencies that we want to capture in our analysis.(up to 10Hz with sampling rate of 32)
                    C3_Coherences.append(MagCoherenceC3)
                    C3_labels.append(label)
            elif file.endswith('.mat') and 'F3' in root:
                if 'Control' in root:
                    label = 0
                elif 'Depression' in root:
                    label = 1
                mat_file = os.path.join(root, file)
                mat = loadmat(mat_file)
                for item in range(len(mat['MagCoherence'][0])):
                    MagCoherenceF3 = mat['MagCoherence'][0:256, item]
                    F3_Coherences.append(MagCoherenceF3)
                    F3_labels.append(label)
            elif file.endswith('.mat') and 'O1' in root:
                if 'Control' in root:
                    label = 0
                elif 'Depression' in root:
                    label = 1
                mat_file = os.path.join(root, file)
                mat = loadmat(mat_file)
                for item in range(len(mat['MagCoherence'][0])):
                    MagCoherenceO1 = mat['MagCoherence'][0:256, item]
                    O1_Coherences.append(MagCoherenceO1)
                    O1_labels.append(label)
    merged_array = np.stack((C3_Coherences, F3_Coherences, O1_Coherences), axis=1)
    #merged_array = merged_array.squeeze(3)
    # Convert data to PyTorch tensors
    merged_array = torch.Tensor(merged_array)
    O1_labels = torch.Tensor(O1_labels).long()
    dataset = TensorDataset(merged_array, O1_labels)
    return dataset

    
# Step 2: List and read the MATLAB files from the directory and its subdirectories, along with labels
Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Train'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Test'

Train_dataset = DataPreparation(Train_data_dir)
Test_dataset = DataPreparation(Test_data_dir)


###### Visualize the data ######

'''num_zeros = (Train_dataset[:][1] == 0).sum().item()
num_ones = (Train_dataset[:][1] == 1).sum().item()

class0Data = Train_dataset[:num_zeros][0]
class1Data  = Train_dataset[num_zeros:][0]

# Create histograms or density plots for each feature
num_features = class0Data.shape[2]

for feature_index in range(32):
    plt.figure(figsize=(8, 4))
    
    # Plot histograms for each class
    sns.histplot(class0Data[:, 0, feature_index], kde=True, label='Class 0', color='blue')
    sns.histplot(class1Data[:, 0, feature_index], kde=True, label='Class 1', color='red')
    
    plt.title(f'Distribution of Feature {feature_index}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.show()'''

#create a dataloader using the custom datset
batch_size = 256
TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = True)
TestLoader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)

#print(len(TrainLoader))
#print(len(TestLoader))
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OneDCNN(nn.Module):
    def __init__(self):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=3)

        '''self.fc1 = nn.Linear(64*10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)'''

        self.fc = nn.Sequential(
            nn.Linear(64*10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        '''x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)'''
        x = self.fc(x)
        return x

# Initialize the model
model = OneDCNN().to(device)

# Log model architecture
wandb.watch(model)  # your_model is your 1DCNN model

# Log hyperparameters
config = wandb.config
config.learning_rate = 0.001
config.batch_size = 256
config.epochs = 200

# Add more hyperparameters as needed
######### Using L1 and L2 regularization to prevent overfitting ##########
# Define loss and optimizer
# Define the L1 and L2 regularization strengths
l1_strength = 0.01  # Adjust as needed
l2_strength = 0.01  # Adjust as needed

# Define the loss function with L1 and L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    correct = 0
    total = 0
    model.train()
    for segments, labels in TrainLoader:
        optimizer.zero_grad()
        outputs = model(segments)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        '''regularization_loss = 0
        for param in model.parameters():
            if param.requires_grad:
                regularization_loss += l1_strength * torch.norm(param, p=1) + l2_strength * torch.norm(param, p=2)

        loss = criterion(outputs, labels) + regularization_loss # Calculating loss with regularization parameter!'''

        loss = criterion(outputs, labels) # Calculating loss without regularization parameter!
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item(), "train_accuracy": (correct / total)})

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}')

# Evaluation
with torch.no_grad():
    correct = 0
    total = 0
    for segments, labels in TestLoader:
        model.eval()
        outputs = model(segments)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total

print("Test Accuracy:", accuracy)
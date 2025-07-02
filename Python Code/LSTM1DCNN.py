import os
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


# Step 1: Create a dictionary to map subdirectory names to labels
label_mapping = {
    r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Train\Control Cases': 0,
    r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Train\Depression Cases': 1,
    r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Test\Control Cases': 0,
    r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Test\Depression Cases': 1,
}

def DataPreparation(data_dir):
    data_and_labels = []
    All_labels = []
    All_segments = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat') and 'EEG' in file:
                if 'Control' in root:
                    EEGlabel = 0
                elif 'Depression' in root:
                    EEGlabel = 1
                mat_file = os.path.join(root, file)
                mat = loadmat(mat_file)
                CleanedEEGinfo = mat['CleanedEEGinfo']
                CleanedC3 = CleanedEEGinfo['CleanedC3'][0][0].T
                CleanedF3 = CleanedEEGinfo['CleanedF3'][0][0].T
                CleanedO1 = CleanedEEGinfo['CleanedO1'][0][0].T
                print(CleanedF3.shape)
            elif file.endswith('.mat') and 'ECG' in file:
                if 'Control' in root:
                    ECGlabel = 0
                elif 'Depression' in root:
                    ECGlabel = 1
                mat_file = os.path.join(root, file)
                mat = loadmat(mat_file)
                ECG = mat['ECG']
                ECGCleaned = ECG['ECGCleaned'][0][0]
                print(ECGCleaned.shape)
        if len(files) != 0:
            # Concatenate the channels along the last dimension (features)
            if len(CleanedC3) == len(ECGCleaned):
                ConcatenateData = np.concatenate((CleanedC3, CleanedF3, CleanedO1, ECGCleaned), axis=1, dtype = np.double)
            else:
                ConcatenateData = np.concatenate((CleanedC3, CleanedF3, CleanedO1, ECGCleaned[0:len(CleanedC3)]), axis=1, dtype = np.double)

            #Z-score normalization
            mean = np.mean(ConcatenateData, axis = 0)
            std = np.std(ConcatenateData, axis = 0)
            NormalizedConcatenateData = (ConcatenateData - mean)/std

            if EEGlabel == ECGlabel:
                label = EEGlabel
            sequence_length = 5*60*256 # 5min
            OL = 2*60*256 # 2min overelap
            X = sequence_length - OL # 3min
            # Create sequences with a sliding window
            sequences = [NormalizedConcatenateData[i*X:i*X + sequence_length] for i in range(0, math.floor(len(NormalizedConcatenateData)/X)-1)] # segmenting data in 5min sequences with 2min overlap
            if label == 0:
                labels = np.zeros(len(sequences), dtype=np.int64)
            else:
                labels = np.ones(len(sequences), dtype=np.int64)

            All_labels.append(torch.tensor(labels))
            All_segments.append(torch.tensor(sequences, dtype = torch.double))
        else:
            pass

    All_labels_list = [item for inner_list in All_labels for item in inner_list]
    All_segments_list = [item for inner_list in All_segments for item in inner_list]

    dataset = TensorDataset(torch.stack(All_segments_list), torch.stack(All_labels_list))
    return dataset

# Step 2: List and read the MATLAB files from the directory and its subdirectories, along with labels
Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\ExaminData\Train'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\ExaminData\Test'

Train_dataset = DataPreparation(Train_data_dir)
Test_dataset = DataPreparation(Test_data_dir)

#create a dataloader using the custom datset
batch_size = 32
TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = False)
TestLoader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super(LSTM1DCNN, self).__init__()
        
        # 1DCNN layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)       
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=hidden_size1, batch_first=True, num_layers= num_layers, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=hidden_size2, batch_first=True, num_layers= num_layers, dropout=0.2)       
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size2, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes)
        )
        self.max = nn.LogSoftmax()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))
        #x = self.cnn(x)
        #print('After CNN: ', x.shape)
        # Reshape for LSTM (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)
        #print('Before LSTM: ', x.shape)
        h01 = torch.zeros(num_layers, x.size(0), hidden_size1).to(x.device)
        c01 = torch.zeros(num_layers, x.size(0), hidden_size1).to(x.device)
        lstm_out1, _ = self.lstm1(x, (h01, c01))
        lstm_out2, _ = self.lstm2(lstm_out1)
        #print('LSTM output shape:', lstm_out.shape)
        output = self.fc(lstm_out2[:, -1, :])
        softmaxoutput = self.max(output)
        softmaxoutput = torch.argmax(softmaxoutput, dim=1)
        #print('output shape:',output.shape)
        return output, softmaxoutput

input_size = 4  # Number of features in the input
hidden_size1 = 32  # Number of hidden units in the LSTM layer
hidden_size2 = 16  # Number of hidden units in the LSTM layer
num_layers = 2  # Number of LSTM layers
num_classes = 2  # Number of output classes
num_epochs = 1
learning_rate = 0.001
sequence_length = 5*60*256

model = LSTM1DCNN(input_size, hidden_size1, hidden_size2, num_layers, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for segments, labels in TrainLoader:
        print('checking the image shape before giving the input to the network:')
        print(segments.shape)
        plt.plot(segments[1, :, :])
        plt.show()
        #segments = segments.view(-1, sequence_length, input_size).to(device) # changing the input segments in the shape of the input to the
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
        print(segments.shape)
        #segments = segments.double()
        #labels = labels.double()

        (predictions, softmaxoutput) = model(segments.float())
        print(softmaxoutput)
        predictions = predictions.float()
        predictions = torch.tensor(predictions, requires_grad= True)
        predictions = predictions.view(-1, 2)
        print('pred:::',predictions)
        print(predictions.shape)
        print('True labels:', labels)
        loss = criterion(predictions, labels.view(-1))
        acc = torch.sum(softmaxoutput == labels).item() / (batch_size)
        
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
        for segments, labels in loader:
            #segments = segments.view(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            labels = labels.float()

            '''num_zeros = torch.sum(labels == 0).item()
            num_ones = torch.sum(labels == 1).item()
            if num_zeros > num_ones:
                labels = torch.zeros(int(batch_size/sequence_length),dtype=torch.int64)
            else:
                labels = torch.ones(int(batch_size/sequence_length),dtype=torch.int64)'''

            (predictions, softmaxoutput) = model(segments.float())
            acc = torch.sum(softmaxoutput == labels).item() / (batch_size)

            epoch_acc += acc

    return epoch_acc / len(loader)

test_acc = evaluate(model, TestLoader)
print(f'Test Accuracy: {test_acc:.4f}')


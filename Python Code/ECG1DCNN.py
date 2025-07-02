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




# Number of samples per class
num_samples = 500  # Half for each class
test_size = 0.2

# Create data for class 0 with a clear pattern
class_0_data = np.random.normal(loc=2, scale=1, size=(num_samples, 1))

# Create data for class 1 with a different pattern
class_1_data = np.random.normal(loc=8, scale=1, size=(num_samples, 1))

# Labels: Class 0 is labeled as 0, Class 1 is labeled as 1
labels = np.concatenate([np.zeros(num_samples), np.ones(num_samples)])

# Combine the data from both classes
data = np.concatenate([class_0_data, class_1_data], axis=0)

# Shuffle the data and labels
permutation = np.random.permutation(2 * num_samples)
data = data[permutation]
labels = labels[permutation]

data = data.reshape(-1, 1, 1)  # Reshape to (batch_size, 1, 1)

# Convert data to PyTorch tensors
data = torch.Tensor(data)
labels = torch.Tensor(labels).long()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

Train_dataset = TensorDataset(X_train, y_train)
Test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = False)
TestLoader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)

# Define a simple 1DCNN model

class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 2)  # Two output classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
# Initialize the model
model = Simple1DCNN()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for segments, labels in TrainLoader:
        optimizer.zero_grad()
        outputs = model(segments)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

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

'''# Initialize the SVM classifier
SVM = SVC(kernel='linear', C=1.0, random_state=42)

# Fit the classifier to the training data
SVM.fit(X_train, y_train)

# Make predictions on the test data
y_pred = SVM.predict(X_test)

# Calculate and print the accuracy of the classifier
SVMaccuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print("SVM Accuracy:", SVMaccuracy)

# Initialize the MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print("MLP Accuracy:", accuracy)
print("MLP Confusion Matrix:\n", conf_mat)'''

train_data = torch.Tensor(X_train)
train_labels = torch.Tensor(y_train)
test_data = torch.Tensor(X_test)
test_labels = torch.Tensor(y_test)

Train_dataset = TensorDataset(train_data, train_labels)
Test_dataset = TensorDataset(test_data, test_labels)

# Step 1: Create a dictionary to map subdirectory names to labels
r'''label_mapping = {
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

            NormalizedConcatenateData = NormalizedConcatenateData[:, 3]

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
Test_dataset = DataPreparation(Test_data_dir)'''

#create a dataloader using the custom datset
batch_size = 32
TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = False)
TestLoader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)

#print(len(TrainLoader))
#print(len(TestLoader))
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super(LSTM1DCNN, self).__init__()
        
        # 1DCNN layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)     
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256*10, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
            #nn.Sigmoid()
        )
        self.max = nn.LogSoftmax()

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))
 
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        output = self.fc(x)
        softmaxoutput = self.max(output)
        softmaxoutput = torch.argmax(softmaxoutput, dim=1)

        return output, softmaxoutput

input_size = 1  # Number of features in the input
hidden_size1 = 32  # Number of hidden units in the LSTM layer
hidden_size2 = 16  # Number of hidden units in the LSTM layer
num_layers = 2  # Number of LSTM layers
num_classes = 2  # Number of output classes
num_epochs = 200
learning_rate = 0.001
#sequence_length = 5*60*256
sequence_length = 100

model = LSTM1DCNN(input_size, hidden_size1, hidden_size2, num_layers, num_classes).to(device)


# Define the loss function and optimizer
#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_acc = 0
    model.train()

    for segments, labels in TrainLoader:
        optimizer.zero_grad()
        #print('checking the image shape before giving the input to the network:')
        #print(segments.shape)
        #plt.plot(segments[1, :])
        #plt.show()
        labels = labels.to(device)
        segments = segments.to(device)
        # Convert labels to LongTensor
        labels = torch.tensor(labels, dtype=torch.long)
        #print(labels.shape)
        #print('checking the image shape after resizing the input image before giving the input to the network:')
        #print(segments.shape)

        (predictions, softmaxoutput) = model(segments.float())
        #print(softmaxoutput)
        predictions = predictions.float()
        #predictions = torch.tensor(predictions, requires_grad= True)
        #predictions = predictions.view(-1, 2)
        #print('pred:::',predictions)
        #print(predictions.shape)
        #print('True labels:', labels)
        #print(labels.shape)
        #predictions = predictions.squeeze()
        #labels = labels.float()
        loss = criterion(predictions, labels.view(-1))
        #acc = torch.sum(softmaxoutput == labels).item() / batch_size
        
        loss.backward()
        optimizer.step()

        #epoch_acc += acc

    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluate the model
def evaluate(model, loader):
    model.eval()
    epoch_acc = 0
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for segments, labels in loader:
            labels = labels.to(device)
            labels = labels.float()
            segments = segments.to(device)

            (predictions, softmaxoutput) = model(segments.float())
            acc = torch.sum(softmaxoutput == labels).item() / batch_size

            epoch_acc += acc

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save the true labels and predictions for confusion matrix calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(all_labels, all_predictions)
    test_acc = (100 * correct / total)

    # Print the accuracy
    #print(f'Accuracy of the model on the validation images: {100 * correct / total:.2f}%')

    #return epoch_acc / len(loader)
    return test_acc, conf_mat

test_acc, conf_mat = evaluate(model, TestLoader)
#print(f'Test Accuracy: {test_acc:.4f}')
# Print the accuracy
print(f'Accuracy of the model on the test set: {test_acc:.2f}%')
# Print the confusion matrix
print("Confusion Matrix:")
print(conf_mat)


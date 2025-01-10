import os
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset
import torch
import math
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import torchvision.models as models
import wandb
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

wandb.init(project = "ClassificationWithBandPowers")

torch.manual_seed(42)  # Set a specific seed for reproducibility

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
                EEGList = [CleanedC3, CleanedF3, CleanedO1]
                PSDList = []
                for item in range(3):
                    DSIList = []
                    TSIList = []
                    ASIList = []
                    for iter in range(int(len(CleanedC3)/(256*300))):
                        # Calculate the PSD with Welch method
                        seg = EEGList[item][iter*256*300:(iter+1)*256*300]
                        seg = np.squeeze(seg)
                        frequencies, PSDEEG = welch(seg, fs=256, nperseg = len(seg))
                        # Find indices corresponding to the frequency band
                        Delta_band_indices = np.where((frequencies >= 0.5) & (frequencies <= 4))[0]       # We calculate the power for Delta frequency band!
                        Theta_band_indices = np.where((frequencies >= 4) & (frequencies <= 8))[0]       # We calculate the power for Theta frequency band!
                        Alpha_band_indices = np.where((frequencies >= 8) & (frequencies <= 13))[0]       # We calculate the power for Theta frequency band!
                        # Calculate power in the specified frequency band
                        '''plt.figure(figsize=(10, 4))
                        plt.title('Frequency Domain Signal')
                        plt.plot(frequencies, PSDEEG)
                        #plt.yscale('log')  # Set y-axis to logarithmic scale
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Amplitude')
                        plt.show()'''

                        power_in_Delta_band = np.sum(PSDEEG[Delta_band_indices])
                        power_in_Theta_band = np.sum(PSDEEG[Theta_band_indices])
                        power_in_Alpha_band = np.sum(PSDEEG[Alpha_band_indices])
                        DSI = power_in_Delta_band / (power_in_Theta_band + power_in_Alpha_band)
                        TSI = power_in_Theta_band / (power_in_Delta_band + power_in_Alpha_band)
                        ASI = power_in_Alpha_band / (power_in_Delta_band + power_in_Theta_band)
                        DSIList.append(DSI)
                        TSIList.append(TSI)
                        ASIList.append(ASI)
                    PSDList.append(DSIList)
                    PSDList.append(TSIList)
                    PSDList.append(ASIList)

                '''# Plot the signals in time and frequency domains
                plt.figure(figsize=(10, 4))
                plt.title('Frequency Domain Signal')
                plt.plot(freq_EEG[0:442200], PSDEEG[0:442200])
                #plt.yscale('log')  # Set y-axis to logarithmic scale
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')

                plt.tight_layout()
                plt.show()
                print(CleanedF3.shape)'''

                '''elif file.endswith('.mat') and 'ECG' in file:
                if 'Control' in root:
                    ECGlabel = 0
                elif 'Depression' in root:
                    ECGlabel = 1
                mat_file = os.path.join(root, file)
                mat = loadmat(mat_file)
                ECG = mat['ECG']
                ECGCleaned = ECG['ECGCleaned'][0][0]
                # Calculate the Fourier Transform
                ECGdft = np.fft.fft(ECGCleaned)
                fs = 256  # Example sampling frequency (adjust as needed)
                n = len(ECGCleaned)
                ECGdft = ECGdft[0:int(n/2)+1]
                PSDECG = (1/(fs*n)) * abs(ECGdft)**2
                PSDECG[1:-2] = 2 * PSDECG[1:-2]
                freq_ECG = np.arange(0, fs/2 + fs/n, fs/n)
                PSDECG = PSDECG[0:int(np.floor((len(CleanedC3)*2.5)/256))]

                #Plot the signals in time and frequency domains
                plt.figure(figsize=(10, 4))

                plt.title('Frequency Domain Signal')
                plt.plot(freq_ECG[0:442200], PSDECG[0:442200])
                #plt.yscale('log')  # Set y-axis to logarithmic scale
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')

                plt.tight_layout()
                plt.show()
                print(ECGCleaned.shape)'''

        if len(files) != 0 and any('.mat' in word for word in files):
            # Concatenate the channels along the last dimension (features)
            '''if len(PSDList[0]) == len(PSDECG):
                ConcatenateData = np.concatenate((PSDList[0], PSDList[1], PSDList[2], PSDECG), axis=1, dtype = np.double)
            else:'''
            ConcatenateData = np.concatenate((
                np.array(PSDList[0])[:, np.newaxis], np.array(PSDList[1])[:, np.newaxis], np.array(PSDList[2])[:, np.newaxis],
                np.array(PSDList[3])[:, np.newaxis], np.array(PSDList[4])[:, np.newaxis], np.array(PSDList[5])[:, np.newaxis],
                np.array(PSDList[6])[:, np.newaxis], np.array(PSDList[7])[:, np.newaxis], np.array(PSDList[8])[:, np.newaxis]
            ), axis=1, dtype=np.double)

            ####################### USE F3 channel to check the results of one channel first! ################################
            ConcatenateData = ConcatenateData[:, 0:9] # Use only PSDECG to check the results and add other channels later on!!!!
            #Z-score normalization
            mean = np.mean(ConcatenateData, axis = 0)
            std = np.std(ConcatenateData, axis = 0)
            NormalizedConcatenateData = (ConcatenateData - mean)/std

            #if EEGlabel == ECGlabel:
            label = EEGlabel
            sequence_length = 50
            OL = 25
            X = sequence_length - OL
            # Create sequences with a sliding window
            sequences = [ConcatenateData[i*X:i*X + sequence_length] for i in range(0, math.floor(len(ConcatenateData)/X)-1)]
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

    # Initialize variables to store indices of removed '1's
    removed_indices = []
    # Counter to keep track of every third occurrence of '1'
    count = 0
    # Iterate through the list in reverse order to safely remove elements
    for i in range(len(All_labels_list) - 1, -1, -1):
        # Check if the element is equal to 1
        if All_labels_list[i] == 1:
            count += 1
            # If the count is a multiple of 3, remove '1' and save its index
            if count % 6 == 0:
                removed_indices.append(i)
                All_labels_list.pop(i)  # Remove '1' from the list at index i
    # Remove corresponding elements from another_list based on indices
    adjusted_indices = sorted(removed_indices, reverse=True)  # Sort indices in reverse order to avoid index shift
    for i in adjusted_indices:
        del All_segments_list[i]

    dataset = TensorDataset(torch.stack(All_segments_list), torch.stack(All_labels_list))
    return dataset

# Step 2: List and read the MATLAB files from the directory and its subdirectories, along with labels
Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Train'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Test'
Validation_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Validation'

Train_dataset = DataPreparation(Train_data_dir)
Test_dataset = DataPreparation(Test_data_dir)
Validation_dataset = DataPreparation(Validation_data_dir)

'''###### Visualize the data ######

num_zeros = (Train_dataset[:][1] == 0).sum().item()
num_ones = (Train_dataset[:][1] == 1).sum().item()

class0Data = Train_dataset[:num_zeros][0]
class1Data  = Train_dataset[num_zeros:][0]

# Create histograms or density plots for each feature
num_features = class0Data.shape[1]

for feature_index in range(0, 1):
    plt.figure(figsize=(8, 4))
    
    # Plot histograms for each class
    sns.histplot(class0Data[:, 3], kde=True, label='Class 0', color='blue')
    sns.histplot(class1Data[:, 3], kde=True, label='Class 1', color='red')
    
    plt.title(f'Distribution of Feature {feature_index}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.show()
    plt.savefig("data_distribution_plot.png")

wandb.log({"data_distribution_plot": wandb.Image("data_distribution_plot.png")})'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OneD CNN without considering time dependencies of the segments
class OneDCNN(nn.Module):   ################ Input to the network: (Batch_size, Num_Channels, Sequence_len)
    def __init__(self):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=16, kernel_size=3)
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
            nn.Linear(32*4, 32), #### Change this layer's input when you are changing the sequence length!
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        #x = x.unsqueeze(1) ###### Uncomment it when you are using only one channel
        x = x.permute(0, 2, 1)   ######  Uncomment it when you are uisng more than one channel
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        #x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        '''x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)'''
        x = self.fc(x)
        return x


# LSTM1DCNN to consider time dependencies between segments
class LSTM1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_layer, num_classes, D):
        super(LSTM1DCNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size1 = hidden_size1
        self.D = D
        # 1DCNN layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3)
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
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=self.hidden_size1, batch_first=True, num_layers= self.num_layer, dropout=0.2, bidirectional=False)
        #self.lstm2 = nn.LSTM(input_size=32, hidden_size=hidden_size2, batch_first=True, num_layers= num_layers, dropout=0.2)
        #self.lstm3 = nn.LSTM(input_size=16, hidden_size=hidden_size3, batch_first=True, num_layers= num_layers, dropout=0.2)     
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.D*self.hidden_size1, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
        )


    def forward(self, x):
        
        #x = x.unsqueeze(1) #### Uncomment it when you are using only one channel of input!
        x = x.permute(0, 2, 1) #### comment it when you are using only one channel of input!
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        #x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))

        x = x.permute(0, 2, 1)
        h0 = torch.randn(self.D*self.num_layer, x.size(0), self.hidden_size1).to(x.device)
        h0 = torch.tensor(h0, dtype=torch.double)
        c0 = torch.randn(self.D*self.num_layer, x.size(0), self.hidden_size1).to(x.device)
        c0 = torch.tensor(c0, dtype=torch.double)
        lstm_out1, _ = self.lstm1(x, (h0, c0))
        #lstm_out2, _ = self.lstm2(lstm_out1)
        #lstm_out3, _ = self.lstm3(lstm_out2)

        output = self.fc(lstm_out1[:, -1, :])
        return output


# Add more hyperparameters as needed
######### Using L1 and L2 regularization to prevent overfitting ##########
# Define loss and optimizer
# Define the L1 and L2 regularization strengths
l1_strength = 0.01  # Adjust as needed
l2_strength = 0.01  # Adjust as needed

def HyperparameterTuning (lr, batch_size, num_layer, trainval_index, trainval_subset_loader):

    kf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)  # Use random_state for reproducibility
    # Store accuracy scores for each fold
    val_accuracy_scores = []
    train_val_dataset = trainval_subset_loader.dataset
    intargets = np.array([train_val_dataset[item][1] for item in range(len(train_val_dataset))])

    for train_index, val_index in kf.split(np.zeros(len(intargets)), intargets):

        # Create data samplers for training and validation subsets
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_index)

        # Create data loaders for training and validation subsets
        train_subset_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=train_sampler)
        val_subset_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=val_sampler)

        num_epochs = 100
        input_size = 9  # Number of features in the input
        hidden_size1 = 32  # Number of hidden units in the LSTM layer
        hidden_size2 = 16  # Number of hidden units in the LSTM layer
        hidden_size3 = 8
        num_layers = 2  # Number of LSTM layers
        num_classes = 2  # Number of output classes
        D = 1 # 2 if bidirectional ow 1

        # Initialize the model
        #model = OneDCNN().to(device)
        model = LSTM1DCNN(input_size, hidden_size1, hidden_size2, hidden_size3, num_layer, num_classes, D).to(device)
        model = model.double()
        # Log model architecture
        wandb.watch(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr= lr)
        
        for epoch in range(num_epochs):
            train_correct = 0
            train_total = 0
            model.train()
            for segments, labels in train_subset_loader:
                optimizer.zero_grad()
                labels = labels.to(device)
                segments = segments.to(device)
                #segments = torch.tensor(segments, dtype=torch.double)
                outputs = model(segments)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                ################################################################## comment the following lines if you don't use regularization
                '''regularization_loss = 0
                for param in model.parameters():
                    if param.requires_grad:
                        regularization_loss += l1_strength * torch.norm(param, p=1) + l2_strength * torch.norm(param, p=2)

                loss = criterion(outputs, labels) + regularization_loss # Calculating loss with regularization parameter!'''

                loss = criterion(outputs, labels) # Calculating loss without regularization parameter!
                loss.backward()
                optimizer.step()

            wandb.log({"Epoch": (epoch + 1)/(num_epochs), "train_loss": loss.item(), "train_accuracy": (train_correct / train_total)})
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_correct / train_total:.4f}')

        # Validation loop
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for segments, labels in val_subset_loader:
                model.eval()
                labels = labels.to(device)
                segments = segments.to(device)
                outputs = model(segments)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            val_accuracy = val_correct / val_total

        # Log hyperparameters and validation accuracy to wandb
        wandb.log({"lr": lr,  "batch_size": batch_size, "val_accuracy": val_accuracy, "val_loss": val_loss/len(val_subset_loader), "Num layers": num_layer})
        print("Validation Accuracy:", val_accuracy, "Validation Loss:", val_loss/len(val_subset_loader))

        val_accuracy_scores.append(val_accuracy)
    # Calculate mean and standard deviation of accuracy scores across all folds for each epoch
    mean_accuracy = sum(val_accuracy_scores) / len(val_accuracy_scores)

    return mean_accuracy


TrainLoader = DataLoader(Train_dataset, shuffle = False) ################## Shuffle is false to consider the time dependency in time series models!!!!
ValidationLoader = DataLoader(Validation_dataset, shuffle = False)
TestLoader = DataLoader(Test_dataset, shuffle = False)

# Combine training and validation datasets for cross-validation splitting
full_dataset = torch.utils.data.ConcatDataset([TrainLoader.dataset, ValidationLoader.dataset, TestLoader.dataset])
targets = np.array([full_dataset[item][1] for item in range(len(full_dataset))])
kfOuter = StratifiedKFold(n_splits= 10, shuffle=True, random_state=42)  # Use random_state for reproducibility
AllHyperparameters = {}
AllResults = []
ACCs = []
Precesions = []
Recalls = []
F1Scores = []
testsets = -1
for trainval_index, test_index in kfOuter.split(np.zeros(len(targets)), targets):
    testsets += 1
    # Create data samplers for training and validation subsets
    trainval_sampler = torch.utils.data.SubsetRandomSampler(trainval_index)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index)

    # Create data loaders for training and validation subsets
    trainval_subset_loader = DataLoader(full_dataset, sampler=trainval_sampler)
    test_subset_loader = DataLoader(full_dataset, sampler=test_sampler)


    best_hyperparams = {}
    best_accuracy = 0
    MeanAccuracies = []
    # Define hyperparameters to tune
    learning_rates = [ 0.001]
    batch_sizes = [128]
    Num_layers = [2]
    early_stopping_counter = 0
    for lr, batch_size, num_layer in product(learning_rates, batch_sizes, Num_layers):
        mean_accuracy = HyperparameterTuning(lr, batch_size, num_layer, trainval_index, trainval_subset_loader)
        MeanAccuracies.append(mean_accuracy)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_hyperparams = {"lr": lr, "batch_size": batch_size, "Num_layer": num_layer}
            
            '''# Attempt to change the value of "batch_size" with allow_val_change=True
            wandb.config.update({"batch_size": batch_size}, allow_val_change=True)
            wandb.config.update({"learning_rate": lr}, allow_val_change=True)
            wandb.config.update({"Num_layer": num_layer}, allow_val_change=True)
            # Log hyperparameters
            config = wandb.config
            config.learning_rate = best_hyperparams['lr']
            config.batch_size = best_hyperparams['batch_size']
            config.Num_layer = best_hyperparams['Num_layer']
            config.epochs = 150
            config.optimizer = 'Adam'
            config.segment_len = 500
            config.num_folds = 5
            wandb.log({"Best model accuracy:": best_accuracy})'''

    # Print the best hyperparameters and accuracy
    print("Best Hyperparameters:", best_hyperparams)
    print("Best Validation Accuracy:", best_accuracy)
    print('Mean Accuracy List:', MeanAccuracies)

    AllHyperparameters[testsets] = best_hyperparams

    num_epochs = 150
    input_size = 9  # Number of features in the input
    hidden_size1 = 32  # Number of hidden units in the LSTM layer
    hidden_size2 = 16  # Number of hidden units in the LSTM layer
    hidden_size3 = 8
    #num_layer = 4  # Number of LSTM layers
    num_classes = 2  # Number of output classes
    D = 1 # 2 if bidirectional ow 1

    #######################  Train the model after hyperparameter tuning ###############################
    batch_size = best_hyperparams["batch_size"]
    learning_rate = best_hyperparams["lr"]
    num_layer = best_hyperparams["Num_layer"]
    
    # Create data loaders for training and validation subsets
    trainval_subset_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=trainval_sampler)
    test_subset_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    #model2 = OneDCNN().to(device)  ######################## Uncomment it to use 1DCNN Model!
    model2 = LSTM1DCNN(input_size, hidden_size1, hidden_size2, hidden_size3, num_layer, num_classes, D).to(device)
    model2 = model2.double()
    # Log model architecture
    wandb.watch(model2)
    #learning_rate = 0.0001
    #batch_size = 64
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr= learning_rate)

    # Attempt to change the value of "batch_size" with allow_val_change=True
    #wandb.config.update({"batch_size": batch_size}, allow_val_change=True)
    #wandb.config.update({"learning_rate": lr}, allow_val_change=True)
    # Log hyperparameters
    '''config = wandb.config
    config.learning_rate = learning_rate
    config.batch_size = batch_size
    config.epochs = 130
    config.optimizer = 'Adam'
    config.segment_len = 500
    config.Num_layer = num_layer'''
    l1_strength = 0.01  # Adjust as needed
    l2_strength = 0.01  # Adjust as needed
    Results = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        model2.train()
        for segments, labels in trainval_subset_loader:
            optimizer.zero_grad()
            labels = labels.to(device)
            segments = segments.to(device)
            outputs = model2(segments)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            '''regularization_loss = 0
            for param in model2.parameters():
                if param.requires_grad:
                    regularization_loss += l1_strength * torch.norm(param, p=1) + l2_strength * torch.norm(param, p=2)

            loss = criterion(outputs, labels) + regularization_loss # Calculating loss with regularization parameter!'''

            loss = criterion(outputs, labels) # Calculating loss without regularization parameter!
            loss.backward()
            optimizer.step()

        #wandb.log({"Epoch": (epoch + 1)/(num_epochs), "train_loss": loss.item(), "train_accuracy": (train_correct / train_total)})
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_correct / train_total:.4f}')

    # Test the model
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        correct = 0
        total = 0
        test_loss = 0
        for segments, labels in test_subset_loader:
            model2.eval()
            labels = labels.to(device)
            segments = segments.to(device)
            outputs = model2(segments)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Save the true labels and predictions for confusion matrix calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            loss = criterion(outputs, labels)
            test_loss += loss.item()    

        conf_mat = confusion_matrix(all_labels, all_predictions)
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        wandb.log({"Confusion_Matrix": conf_mat, "validation_accuracy": accuracy, "precision": precision, "recall": recall, "f1 score": f1, "Test Loss": test_loss/len(TestLoader)})
        Results.append([conf_mat, accuracy, precision, recall, f1])
        ACCs.append(accuracy)
        Precesions.append(precision)
        Recalls.append(recall)
        F1Scores.append(f1)

    print(f"Iteration {testsets + 1} completed and results are as following")
    print("Test Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ConfusionMatrix:", conf_mat)
    AllResults.append(Results)
    print(AllResults)
    print(AllHyperparameters)

    print("Accuracy list: ",ACCs)
    print("Precesion list: ",Precesions)
    print("Recall list: ",Recalls)
    print("F1Score list: ",F1Scores)
    print('Average accuracies:', sum(ACCs)/len(ACCs))
    print('Average Precesions:', sum(Precesions)/len(Precesions))
    print('Average Recalls:', sum(Recalls)/len(Recalls))
    print('Average F1Scores:', sum(F1Scores)/len(F1Scores))

wandb.run.save()
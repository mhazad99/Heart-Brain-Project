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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
from scipy.interpolate import interp1d
from scipy import signal
from scipy.signal import welch


wandb.init(project = "classificationwithSiameseNetwork")

torch.manual_seed(42)  # Set a specific seed for reproducibility

def DataPreparation1(data_dir):
    HRs = []
    Labels = []
    # Specify the column you want to read from the CSV files
    desired_column = 'HeartRates'

    # Create an empty list to store the data from the specified column
    #segment_len = 1000
    # Traverse through the nested folders
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Check if the file is a CSV file
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if 'Control' in root:
                    label = 0
                elif 'Depression' in root:
                    label = 1
                # Read the CSV file using pandas
                df = pd.read_csv(file_path)
                
                # Check if the desired column exists in the DataFrame
                if desired_column in df.columns:
                    # Extract the data from the specified column and append to the list
                    temp = df[desired_column].tolist()
                    temp = [x for x in temp if (x > 40 and x < 180)]
                    num_chunks = len(temp) // 300
                    # Average every 300 elements and create a new list of averaged values
                    new_temp = [sum(temp[i*300:(i+1)*300]) / 300 for i in range(num_chunks)]

                HRs.append(np.array(new_temp))
                Labels.append(label)
    return HRs, Labels


def DataPreparation2(data_dir):
    BPs = []
    EEGLabels = []
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
                        seg = EEGList[item][iter*256*300:(iter+1)*256*300] # 5 min segments (5*60 = 300)
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
            #mean = np.mean(ConcatenateData, axis = 0)
            #std = np.std(ConcatenateData, axis = 0)
            #NormalizedConcatenateData = (ConcatenateData - mean)/std
            BPs.append(ConcatenateData)
            EEGLabels.append(EEGlabel)

    return BPs, EEGLabels

'''Plotting signals to check their charactaristics
fs = 256
T = np.arange(0, len(F3)/fs, 1/fs)
plt.plot(T, F3)
plt.show()
fs2 = 1.16
T2 = np.arange(0, len(EEGstore[1])/fs2, 1/fs2)
plt.plot(T2, EEGstore[1])
plt.show()
'''
def DataPreparation(BPs, EEGLabels, HRs, Labels):
    All_labels = []
    All_labels2 = []
    All_HR_segments = []
    All_BP_segments = []
    '''for item in range(len(Labels)):
        C3 = EEGs[item][0] 
        F3 = EEGs[item][1]
        O1 = EEGs[item][2]
        EEGstore = []
        for iter in range(3):
            fs = 256
            #T = np.arange(0, len(F3) / fs, 1 / fs)
            # Bandpass filter design
            lowcut = 0.5  # Lower cutoff frequency in Hz
            highcut = 4    # Upper cutoff frequency in Hz
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            # Design a 4th-order Butterworth bandpass filter
            b, a = signal.butter(N=4, Wn=[low, high], btype='band')
            # Apply bandpass filter to scaled EEG signal
            EEGchannel = np.squeeze(EEGs[item][iter])
            filtered_signal = signal.filtfilt(b, a, EEGchannel)

            # Plot original and filtered signals
            plt.figure(figsize=(10, 6))
            plt.plot(T, scaled_signal, label='Scaled EEG Signal')
            plt.plot(T, filtered_signal, label='Bandpass Filtered Signal', linewidth=2)
            plt.title('Scaled EEG Signal and Bandpass Filtering')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
            # Downsampling the longer signal to match the length of the shorter signal
            downsample_factor = len(filtered_signal) // len(HRs[item])  # Compute downsampling factor
            downsampled_signal_m = filtered_signal[::downsample_factor]
            downsampled_signal_m = np.reshape(downsampled_signal_m, (len(downsampled_signal_m), 1))
            EEGstore.append(downsampled_signal_m)'''
    newHRs = []
    for item in range(len(Labels)):
        # Downsampling the longer signal to match the length of the shorter signal
        #downsample_factor = len(HRs[item]) // len(BPs[item])  # Compute downsampling factor
        #downsampled_signal_m = HRs[item][::downsample_factor]
        downsampled_signal_m = np.reshape(HRs[item], (len(HRs[item]), 1))
        newHRs.append(downsampled_signal_m)
        if len(HRs[item]) == len(BPs[item]):
            newHRs[item] = newHRs[item].reshape(-1, 1)
            #ConcatenateData = np.concatenate((BPs[item], newHRs[item]), axis=1, dtype = np.double)
        elif len(HRs[item]) > len(BPs[item]): 
            newHRs[item] = np.array(newHRs[item][:len(BPs[item])])
            newHRs[item] = newHRs[item].reshape(-1, 1)
            #ConcatenateData = np.concatenate((BPs[item], newHRs), axis=1, dtype = np.double)
        else:
            BPs[item] = np.array(BPs[item][:len(newHRs[item])])
            newHRs[item] = newHRs[item].reshape(-1, 1)
            #ConcatenateData = np.concatenate((BPs, newHRs), axis=1, dtype = np.double)



        #Z-score normalization
        #mean = np.mean(ConcatenateData, axis = 0)
        #std = np.std(ConcatenateData, axis = 0)
        #NormalizedConcatenateData = (ConcatenateData - mean)/std
        #NormalizedConcatenateData = NormalizedConcatenateData[:, 0:10]

        ###### HR Sequences ########
        if EEGLabels[item] == Labels[item]:
            label = Labels[item]
        sequence_length = 50
        OL = 25
        X = sequence_length - OL
        # Create sequences with a sliding window
        sequences = [newHRs[item][i*X:i*X + sequence_length] for i in range(0, math.floor(len(newHRs[item])/X)-1)] # segmenting data in 5min sequences with 2min overlap
        if label == 0:
            labels = np.zeros(len(sequences), dtype=np.int64)
        else:
            labels = np.ones(len(sequences), dtype=np.int64)

        All_labels.append(torch.tensor(labels))
        All_HR_segments.append(torch.tensor(sequences, dtype = torch.double))


        All_labels_list = [item for inner_list in All_labels for item in inner_list]
        All_HR_segments_list = [item for inner_list in All_HR_segments for item in inner_list]

        ###### BP Sequences ########
        if EEGLabels[item] == Labels[item]:
            label = Labels[item]
        sequence_length = 50
        OL = 25
        X = sequence_length - OL
        # Create sequences with a sliding window
        sequences = [BPs[item][i*X:i*X + sequence_length] for i in range(0, math.floor(len(BPs[item])/X)-1)] # segmenting data in 5min sequences with 2min overlap
        if label == 0:
            labels = np.zeros(len(sequences), dtype=np.int64)
        else:
            labels = np.ones(len(sequences), dtype=np.int64)

        All_labels2.append(torch.tensor(labels))
        All_BP_segments.append(torch.tensor(sequences, dtype = torch.double))


        All_labels_list2 = [item for inner_list in All_labels for item in inner_list]
        All_BP_segments_list = [item for inner_list in All_BP_segments for item in inner_list]

        # Initialize variables to store indices of removed '1's
    removed_indices = []
    # Counter to keep track of every third occurrence of '1'
    count = 0
    # Iterate through the list in reverse order to safely remove elements
    for i in range(len(All_labels_list) - 1, -1, -1):
        # Check if the element is equal to 1
        if All_labels_list[i] == 0:
            count += 1
            # If the count is a multiple of 3, remove '1' and save its index
            if count % 6 == 0:
                removed_indices.append(i)
                All_labels_list.pop(i)  # Remove '1' from the list at index i
    # Remove corresponding elements from another_list based on indices
    adjusted_indices = sorted(removed_indices, reverse=True)  # Sort indices in reverse order to avoid index shift
    for i in adjusted_indices:
        del All_HR_segments_list[i]
        del All_BP_segments_list[i]

    dataset = TensorDataset(torch.stack(All_HR_segments_list), torch.stack(All_BP_segments_list), torch.stack(All_labels_list))
    return dataset


Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Train'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Test'
Validation_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Validation'

TrainHRs, TrainLabels = DataPreparation1(Train_data_dir)
TestHRs, TestLabels = DataPreparation1(Test_data_dir)
ValHRs, ValHRLabels = DataPreparation1(Validation_data_dir)

# Step 2: List and read the MATLAB files from the directory and its subdirectories, along with labels
Train_data_dir1 = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Train'
Test_data_dir1 = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Test'
Validation_data_dir1 = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Validation'

TrainEEGs, TrainEEGLabels = DataPreparation2(Train_data_dir1)
TestEEGs, TestEEGLabels = DataPreparation2(Test_data_dir1)
ValEEGs, ValEEGLabels = DataPreparation2(Validation_data_dir1)

Train_dataset = DataPreparation(TrainEEGs, TrainEEGLabels, TrainHRs, TrainLabels)
Test_dataset = DataPreparation(TestEEGs, TestEEGLabels, TestHRs, TestLabels)
Validation_dataset = DataPreparation(ValEEGs, ValEEGLabels, ValHRs, ValHRLabels)

###### Visualize the data ######

'''num_zeros = (Train_dataset[:][1] == 0).sum().item()
num_ones = (Train_dataset[:][1] == 1).sum().item()

class0Data = Train_dataset[:num_zeros][0]
class1Data  = Train_dataset[num_zeros:][0]

# Create histograms or density plots for each feature
num_features = class0Data.shape[1]

for feature_index in range(100, 101):
    plt.figure(figsize=(8, 4))
    
    # Plot histograms for each class
    sns.histplot(class0Data[:, feature_index], kde=True, label='Class 0', color='blue')
    sns.histplot(class1Data[:, feature_index], kde=True, label='Class 1', color='red')
    
    plt.title(f'Distribution of Feature {feature_index}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    #plt.show()
    plt.savefig("data_distribution_plot.png")

wandb.log({"data_distribution_plot": wandb.Image("data_distribution_plot.png")})'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OneD CNN without considering time dependencies of the segments
class OneDCNN(nn.Module):
    def __init__(self):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
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
            nn.Linear(64*17, 64), #### Change this layer's input whem you are changing the sequence length!
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
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))
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

################ Applying a Qusi-Siamese Network for combining the results of the HR, SWIs, and Coherence
class QuasiSiamese(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size1, num_layer, num_classes, D):
        super(QuasiSiamese, self).__init__()
        ###### First CNN-LSTM layer for HR classification
        self.num_layer = num_layer
        self.hidden_size1 = hidden_size1
        self.D = D
        # 1DCNN layer
        self.conv11 = nn.Conv1d(in_channels=input1_size, out_channels=16, kernel_size=3)
        self.batchnorm11 = nn.BatchNorm1d(16)
        self.relu11 = nn.ReLU()
        self.pool11 = nn.MaxPool1d(kernel_size=3)
        self.conv12 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.batchnorm12 = nn.BatchNorm1d(32)
        self.relu12 = nn.ReLU()
        self.pool12 = nn.MaxPool1d(kernel_size=3)
        self.conv13 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchnorm13 = nn.BatchNorm1d(64)
        self.relu13 = nn.ReLU()
        self.pool13 = nn.MaxPool1d(kernel_size=3)       
        # LSTM layer
        self.lstm11 = nn.LSTM(input_size=32, hidden_size=self.hidden_size1, batch_first=True, num_layers= self.num_layer, dropout=0.2, bidirectional=False)
        
        ###### Second CNN-LSTM for SWIs classification
        # 1DCNN layer
        self.conv21 = nn.Conv1d(in_channels=input2_size, out_channels=16, kernel_size=3)
        self.batchnorm21 = nn.BatchNorm1d(16)
        self.relu21 = nn.ReLU()
        self.pool21 = nn.MaxPool1d(kernel_size=3)
        self.conv22 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.batchnorm22 = nn.BatchNorm1d(32)
        self.relu22 = nn.ReLU()
        self.pool22 = nn.MaxPool1d(kernel_size=3)
        self.conv23 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchnorm23 = nn.BatchNorm1d(64)
        self.relu23 = nn.ReLU()
        self.pool23 = nn.MaxPool1d(kernel_size=3)       
        # LSTM layer
        self.lstm21 = nn.LSTM(input_size=32, hidden_size=self.hidden_size1, batch_first=True, num_layers= self.num_layer, dropout=0.2, bidirectional=False)

        ####### Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(2*self.D*self.hidden_size1, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
        )

    def forward(self, x1, x2):
        # Process the first input
        x1 = x1.squeeze(1) #### Uncomment it when you are using only one channel of input!
        x1 = x1.permute(0, 2, 1) #### comment it when you are using only one channel of input! (Not a case when you are using the squeeze function instead of unsqueeze!)
        x1 = self.pool11(self.relu11(self.batchnorm11(self.conv11(x1))))
        x1 = self.pool12(self.relu12(self.batchnorm12(self.conv12(x1))))
        #x1 = self.pool13(self.relu13(self.batchnorm13(self.conv13(x1))))

        x1 = x1.permute(0, 2, 1)
        h0 = torch.randn(self.D*self.num_layer, x1.size(0), self.hidden_size1).to(x1.device)
        h0 = torch.tensor(h0, dtype=torch.double)
        c0 = torch.randn(self.D*self.num_layer, x1.size(0), self.hidden_size1).to(x1.device)
        c0 = torch.tensor(c0, dtype=torch.double)
        x1, _ = self.lstm11(x1, (h0, c0))

        # Process the second input
        #x = x.unsqueeze(1) #### Uncomment it when you are using only one channel of input!
        x2 = x2.permute(0, 2, 1) #### comment it when you are using only one channel of input!
        x2 = self.pool21(self.relu21(self.batchnorm21(self.conv21(x2))))
        x2 = self.pool22(self.relu22(self.batchnorm22(self.conv22(x2))))
        #x2 = self.pool23(self.relu23(self.batchnorm23(self.conv23(x2))))

        x2 = x2.permute(0, 2, 1)
        h0 = torch.randn(self.D*self.num_layer, x2.size(0), self.hidden_size1).to(x2.device)
        h0 = torch.tensor(h0, dtype=torch.double)
        c0 = torch.randn(self.D*self.num_layer, x2.size(0), self.hidden_size1).to(x2.device)
        c0 = torch.tensor(c0, dtype=torch.double)
        x2, _ = self.lstm21(x2, (h0, c0))

        # Concatenate the processed inputs
        concatenated = torch.cat([x1[:, -1, :], x2[:, -1, :]], dim=1)

        # Final classification layer
        output = self.fc(concatenated)
        return output


#######################  Train the model after hyperparameter tuning ###############################
num_epochs = 100
input_size = 10  # Number of features in the input
input_size1 = 1
input_size2 = 9
hidden_size1 = 32  # Number of hidden units in the LSTM layer
hidden_size2 = 16  # Number of hidden units in the LSTM layer
hidden_size3 = 8
num_layers = 3  # Number of LSTM layers
num_classes = 2  # Number of output classes
D = 1 # 2 if bidirectional ow 1

#model = OneDCNN().to(device)  ######################## Uncomment it to use 1DCNN Model!
model = QuasiSiamese(input_size1, input_size2, hidden_size1, num_layers, num_classes, D).to(device)
model = model.double()
# Log model architecture
wandb.watch(model)
#batch_size = best_hyperparams["batch_size"]
#learning_rate = best_hyperparams["lr"]
learning_rate = 0.001
batch_size = 128
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = True)
ValidationLoader = DataLoader(Validation_dataset, batch_size = batch_size, shuffle = True)
##### Merge two train and validation set to retrain the model and achieve the high training performance
first_elements = torch.cat((Train_dataset[:][0], Validation_dataset[:][0]))
second_elements = torch.cat((Train_dataset[:][1], Validation_dataset[:][1]))
third_elements = torch.cat((Train_dataset[:][2], Validation_dataset[:][2]))
TrainValDataset = TensorDataset(first_elements, second_elements, third_elements)

TrainValLoader = DataLoader(TrainValDataset, batch_size=batch_size, shuffle=True)

################################################################################################# Perform k-fold cross-validation ############
# Combine training and validation datasets for cross-validation splitting
full_dataset = torch.utils.data.ConcatDataset([TrainLoader.dataset, ValidationLoader.dataset])

#kf = KFold(n_splits= 5, shuffle=True, random_state=42)  # Use random_state for reproducibility
# Store accuracy scores for each fold
#accuracy_scores = []

# Attempt to change the value of "batch_size" with allow_val_change=True
#wandb.config.update({"batch_size": batch_size}, allow_val_change=True)
#wandb.config.update({"learning_rate": lr}, allow_val_change=True)
# Log hyperparameters
config = wandb.config
config.learning_rate = learning_rate
config.batch_size = batch_size
config.epochs = num_epochs
config.optimizer = 'Adam'
config.segment_len = 300
config.num_folds = 5


for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    model.train()
    for HRsegments, BPsegments, labels in TrainValLoader:
        optimizer.zero_grad()
        labels = labels.to(device)
        HRsegments = HRsegments.to(device)
        BPsegments = BPsegments.to(device)
        outputs = model(HRsegments, BPsegments)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

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

TestLoader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)
# Test the model
with torch.no_grad():
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    val_loss = 0
    for HRsegments, BPsegments, labels in TestLoader:
        model.eval()
        labels = labels.to(device)
        HRsegments = HRsegments.to(device)
        BPsegments = BPsegments.to(device)
        outputs = model(HRsegments, BPsegments)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Save the true labels and predictions for confusion matrix calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        val_loss = criterion(outputs, labels)
    
    conf_mat = confusion_matrix(all_labels, all_predictions)
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    wandb.log({"Confusion_Matrix": conf_mat, "validation_accuracy": accuracy, "precision": precision, "recall": recall, "f1 score": f1, "Validation Loss": val_loss})

    print("Validation Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ConfusionMatrix:", conf_mat)

wandb.run.save()
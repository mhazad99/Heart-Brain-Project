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
import seaborn as sns
import wandb
from itertools import product
import pandas as pd


#wandb.init(project = "classificationwithhr")

torch.manual_seed(42)  # Set a specific seed for reproducibility

def DataPreparation(data_dir):
    All_labels = []
    All_segments = []
    # Specify the column you want to read from the CSV files
    desired_column = 'HeartRates'

    # Create an empty list to store the data from the specified column
    segment_len = 500
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
                    sequences = [temp[i*segment_len:(i+1)*segment_len] for i in range(0, math.floor(len(temp)/segment_len)-1)]
                    if label == 0:
                        labels = np.zeros(len(sequences), dtype=np.int64)
                    else:
                        labels = np.ones(len(sequences), dtype=np.int64)
                    
                All_labels.append(torch.tensor(labels))
                All_segments.append(torch.tensor(sequences, dtype = torch.double))

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
            if count % 5 == 0:
                removed_indices.append(i)
                All_labels_list.pop(i)  # Remove '1' from the list at index i
    # Remove corresponding elements from another_list based on indices
    adjusted_indices = sorted(removed_indices, reverse=True)  # Sort indices in reverse order to avoid index shift
    for i in adjusted_indices:
        del All_segments_list[i]
        
    dataset = TensorDataset(torch.stack(All_segments_list), torch.stack(All_labels_list))
    return dataset


Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Train'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Test'
Validation_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Validation'

Train_dataset = DataPreparation(Train_data_dir)
Test_dataset = DataPreparation(Test_data_dir)
Validation_dataset = DataPreparation(Validation_data_dir)

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
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=self.hidden_size1, batch_first=True, num_layers= self.num_layer, dropout=0.2, bidirectional=False)
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
        
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batchnorm3(self.conv3(x))))

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

def HyperparameterTuning (lr, batch_size, num_layer):

    num_epochs = 150
    input_size = 1  # Number of features in the input
    hidden_size1 = 32  # Number of hidden units in the LSTM layer
    hidden_size2 = 16  # Number of hidden units in the LSTM layer
    hidden_size3 = 8
    num_layers = 2  # Number of LSTM layers
    num_classes = 2  # Number of output classes
    D = 1 # 2 if bidirectional ow 1

    batch_size = int(batch_size)

    TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = False) ################## Shuffle is false to consider the time dependency in time series models!!!!
    ValidationLoader = DataLoader(Validation_dataset, batch_size = batch_size, shuffle = False)

    # Combine training and validation datasets for cross-validation splitting
    full_dataset = torch.utils.data.ConcatDataset([TrainLoader.dataset, ValidationLoader.dataset])

    kf = KFold(n_splits= 5, shuffle=True, random_state=42)  # Use random_state for reproducibility
    # Store accuracy scores for each fold
    val_accuracy_scores = []

    for train_index, val_index in kf.split(full_dataset):

        # Create data samplers for training and validation subsets
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_index)

        # Create data loaders for training and validation subsets
        train_subset_loader = DataLoader(full_dataset, batch_size=TrainLoader.batch_size, sampler=train_sampler)
        val_subset_loader = DataLoader(full_dataset, batch_size=ValidationLoader.batch_size, sampler=val_sampler)

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


best_hyperparams = {}
best_accuracy = 0
MeanAccuracies = []
# Define hyperparameters to tune
learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [64, 128, 256, 512]
Num_layers = [2, 3, 4, 5]
# Early stopping setup
early_stopping_counter = 0
for lr, batch_size, num_layer in product(learning_rates, batch_sizes, Num_layers):
    mean_accuracy = HyperparameterTuning(lr, batch_size, num_layer)
    MeanAccuracies.append(mean_accuracy)
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_hyperparams = {"lr": lr, "batch_size": batch_size, "Num_layer": num_layer}

        # Attempt to change the value of "batch_size" with allow_val_change=True
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
        wandb.log({"Best model accuracy:": best_accuracy})

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", best_hyperparams)
print("Best Validation Accuracy:", best_accuracy)
print('Mean Accuracy List:', MeanAccuracies)

# Save the best model or use it for further evaluation on the test set
wandb.run.save()




'''num_epochs = 130
input_size = 1  # Number of features in the input
hidden_size1 = 32  # Number of hidden units in the LSTM layer
hidden_size2 = 16  # Number of hidden units in the LSTM layer
hidden_size3 = 8
num_layer = 4  # Number of LSTM layers
num_classes = 2  # Number of output classes
D = 1

#######################  Train the model after hyperparameter tuning ###############################

#model = OneDCNN().to(device)  ######################## Uncomment it to use 1DCNN Model!
model = LSTM1DCNN(input_size, hidden_size1, hidden_size2, hidden_size3, num_layer, num_classes, D).to(device)
model = model.double()
# Log model architecture
wandb.watch(model)
#batch_size = best_hyperparams["batch_size"]
#learning_rate = best_hyperparams["lr"]
learning_rate = 0.0001
batch_size = 64
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

TrainLoader = DataLoader(Train_dataset, batch_size = batch_size, shuffle = False)
ValidationLoader = DataLoader(Validation_dataset, batch_size = batch_size, shuffle = False)
##### Merge two train and validation set to retrain the model and achieve the high training performance
first_elements = torch.cat((Train_dataset[:][0], Validation_dataset[:][0]))
second_elements = torch.cat((Train_dataset[:][1], Validation_dataset[:][1]))
TrainValDataset = TensorDataset(first_elements, second_elements)

TrainValLoader = DataLoader(TrainValDataset, batch_size=batch_size, shuffle=False)


# Attempt to change the value of "batch_size" with allow_val_change=True
#wandb.config.update({"batch_size": batch_size}, allow_val_change=True)
#wandb.config.update({"learning_rate": lr}, allow_val_change=True)
# Log hyperparameters
config = wandb.config
config.learning_rate = learning_rate
config.batch_size = batch_size
config.epochs = 130
config.optimizer = 'Adam'
config.segment_len = 500
config.Num_layer = num_layer

for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    model.train()
    for segments, labels in TrainValLoader:
        optimizer.zero_grad()
        outputs = model(segments)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        regularization_loss = 0
        for param in model.parameters():
            if param.requires_grad:
                regularization_loss += l1_strength * torch.norm(param, p=1) + l2_strength * torch.norm(param, p=2)

        loss = criterion(outputs, labels) + regularization_loss # Calculating loss with regularization parameter!

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
    test_loss = 0
    for segments, labels in TestLoader:
        model.eval()
        outputs = model(segments)
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
    

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ConfusionMatrix:", conf_mat)


wandb.run.save()'''
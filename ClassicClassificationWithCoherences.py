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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import torchvision.models as models
import seaborn as sns
import wandb
from itertools import product

wandb.init(project = "classificationwithcoherences")
np.random.seed(42)

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
    pca = PCA(n_components= 30)
    C3_Coherences = np.array(C3_Coherences)
    C3_Coherences = pca.fit_transform(C3_Coherences) ### Applying PCA on the input segments
    #merged_array = np.stack((C3_Coherences, F3_Coherences, O1_Coherences), axis=1)
    #merged_array = merged_array.squeeze(3)
    # Convert data to PyTorch tensors
    #merged_array = torch.Tensor(merged_array)
    #C3_labels = torch.Tensor(C3_labels).long()
    #dataset = TensorDataset(merged_array, C3_labels)
    return C3_labels, C3_Coherences

    
# Step 2: List and read the MATLAB files from the directory and its subdirectories, along with labels
Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Train'
Validation_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Validation'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Test'

Train_dataset = DataPreparation(Train_data_dir)
Validation_dataset = DataPreparation(Validation_data_dir)
Test_dataset = DataPreparation(Test_data_dir)


Train_labels, Train_segments = DataPreparation(Train_data_dir)
Test_labels, Test_segments = DataPreparation(Test_data_dir)
Validation_labels, Validation_segments = DataPreparation(Validation_data_dir)


# Initialize and train an SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')
svm_classifier.fit(Train_segments, Train_labels)

# Make predictions on the test set
predictions = svm_classifier.predict(Test_segments)

# Evaluate the classifier
accuracy = accuracy_score(Test_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
conf_mat = confusion_matrix(Test_labels, predictions)
precision = precision_score(Test_labels, predictions)
recall = recall_score(Test_labels, predictions)
f1 = f1_score(Test_labels, predictions)
wandb.log({"Confusion_Matrix": conf_mat, "test_accuracy": accuracy, "precision": precision, "recall": recall, "f1 score": f1})

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ConfusionMatrix:", conf_mat)

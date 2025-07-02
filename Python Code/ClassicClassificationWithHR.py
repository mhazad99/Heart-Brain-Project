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
from sklearn.model_selection import train_test_split
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


wandb.init(project = "classificationwithhr")

#torch.manual_seed(42)  # Set a specific seed for reproducibility

def DataPreparation(data_dir):
    All_labels = []
    All_segments = []
    # Specify the column you want to read from the CSV files
    desired_column = 'HeartRates'

    # Create an empty list to store the data from the specified column
    segment_len = 250
    pca = PCA(n_components= 30)
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
                    temp = [x for x in temp if (x > 30 and x < 180)]
                    sequences = [temp[i*segment_len:(i+1)*segment_len] for i in range(0, math.floor(len(temp)/segment_len)-1)]
                    sequence_array = np.array(sequences)
                    sequence_array = pca.fit_transform(sequence_array) ### Applying PCA on the input segments
                    if label == 0:
                        labels = np.zeros(len(sequence_array), dtype=np.int64)
                    else:
                        labels = np.ones(len(sequence_array), dtype=np.int64)

                All_labels.append(labels)
                All_segments.append(sequence_array)

    All_labels_list = [item for inner_list in All_labels for item in inner_list]
    All_segments_list = [item for inner_list in All_segments for item in inner_list]

    return All_labels_list, All_segments_list


Train_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Train'
Test_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Test'
Validation_data_dir = r'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Validation'

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
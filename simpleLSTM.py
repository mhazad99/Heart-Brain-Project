import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import random
import numpy as np

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MNIST dataset
train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
test_data = MNIST(root='data', train=False, transform=ToTensor())

# Set hyperparameters
input_size = 28
hidden_size = 128
output_size = 10
num_epochs = 10
learning_rate = 0.001
batch_size = 64

# Define the LSTM network
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        print(input.shape)
        lstm_out, _ = self.lstm(input)
        print(lstm_out.shape)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Create the LSTM network
model = LSTMNet(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for images, labels in train_loader:
        print(images.shape)
        images = images.view(-1, input_size, input_size).to(device)
        print(images.shape)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(dim=1) == labels).sum().item() / len(labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Acc: {epoch_acc/len(train_loader):.4f}')

# Evaluate the model
def evaluate(model, loader):
    model.eval()
    epoch_acc = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, input_size, input_size).to(device)
            labels = labels.to(device)

            predictions = model(images)
            acc = (predictions.argmax(dim=1) == labels).sum().item() / len(labels)

            epoch_acc += acc

    return epoch_acc / len(loader)

test_acc = evaluate(model, test_loader)
print(f'Test Accuracy: {test_acc:.4f}')

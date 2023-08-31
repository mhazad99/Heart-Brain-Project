import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

'''
def stack_images_from_folder(folder_path, transform):
    image_list = []
    labels = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Load and transform the image
            image = Image.open(file_path)
            
            # Extract the label based on the parent folder name
            parent_folder_name = os.path.basename(os.path.dirname(file_path))
            label = folder_dict[parent_folder_name]
            
            image_list.append(image)
            labels.append(label)
    
    stacked_images = torch.stack(image_list, dim=0)
    stacked_labels = torch.tensor(labels)
    
    return stacked_images, stacked_labels

# Define the root directory containing the folders
root_dir = "D:\MASc @ ETS\OneDrive - ETS\Desktop\Test\F3"

# Define the desired transformations for the images
transform = ToTensor()  # Convert the images to tensors

# Initialize an empty dictionary to store folder-label mappings
folder_dict = {}

# Iterate over the folders in the root directory and assign labels dynamically
label_counter = 0
for root, _, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(root, file)
        
        if os.path.dirname(file_path):
            folder_dict[file] = label_counter
            label_counter += 1
print(folder_dict)

# Initialize empty lists to store stacked images and labels

all_stacked_images = []
all_labels = []

# Iterate over the folders in the root directory
for folder_name in folder_dict.keys():
    folder_path = os.path.join(root_dir, folder_name)
    
    # Stack images and labels for each folder
    stacked_images, stacked_labels = stack_images_from_folder(folder_path, transform)
    
    all_stacked_images.append(stacked_images)
    all_labels.append(stacked_labels)

print(all_stacked_images)
print(all_labels)

# Concatenate stacked images and labels from all folders
#all_stacked_images = torch.cat(all_stacked_images, dim=0)
#all_labels = torch.cat(all_labels, dim=0)

# The 'all_stacked_images' tensor will have the shape (num_images, channels, height, width)
# The 'all_labels' tensor will have the shape (num_images,)
# You can use these tensors as input to your 3D CNN model
'''

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
num_epochs = 1
num_channels = 12
image_height = 250
image_width = 125
#input_size = image_height * image_width * num_channels
hidden_size = 128
num_layers = 2
output_size = 2
learning_rate = 0.001
sequence_length = 16

# Define the LSTM network
class LSTMNet(nn.Module):
    def __init__(self,  hidden_size, output_size, num_layers, stride=1, padding=1):
        super(LSTMNet, self).__init__()
        # prelayer, fix the input size
        self.padding = nn.ZeroPad2d((0, 3, 3, 3))
        # convolutional layer
        # 1st layer of CNN
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2end layer of CNN
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3rd layer of CNN
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 4th layer of CNN
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 5th layer of CNN
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=stride, padding=padding)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 6th layer of CNN
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=stride, padding=padding)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_size = 2*4*1024
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers= num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(self.input_size, output_size)
        self.max = nn.LogSoftmax()
        
    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0], x.shape[1], 1024, 4, 2).cuda()
        else:
            tmp = torch.zeros(x.shape[0], x.shape[1], 1024, 4, 2).cpu()
        for i in range(32):
            input = x[:, i]
            print('input shape:', input.shape)
            input = self.padding(input)

            input = self.relu1(self.conv1(input))
            input = self.pool1(input)

            input = self.relu2(self.conv2(input))
            input = self.pool2(input)

            input = self.relu3(self.conv3(input))
            input = self.pool3(input)

            input = self.relu4(self.conv4(input))
            input = self.pool4(input)

            input = self.relu5(self.conv5(input))
            input = self.pool5(input)
            
            input = self.relu6(self.conv6(input))
            tmp[:, i] = self.pool6(input)
            del input
        print(tmp.shape)
        input = tmp.reshape(x.shape[0], x.shape[1], 4 * 1024 * 2)
        del tmp
        #input = torch.cat((input[1:], input[2:]))
        print('new input:', input.shape)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        lstm_out, _ = self.lstm(input, (h0, c0))
        #print('LSTM output shape:', lstm_out.shape)
        output = self.fc(lstm_out[:, -1, :])
        output = self.max(output)
        print('output shape:',output.shape)
        return output

# Create the LSTM network
model = LSTMNet(hidden_size, output_size, num_layers).to(device)
summary(model, input_size=(32, 12, 250, 125))



class LSTM(nn.Module):
    '''
    Build the LSTM model applying a RNN over the 7 parallel convnets outputs

    param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
    param kernel: kernel size used for the convolutional layers
    param stride: stride apply during the convolutions
    param padding: padding used during the convolutions
    param max_kernel: kernel used for the maxpooling steps
    param n_classes: number of classes
    param n_units: number of units
    return x: output of the last layers after the log softmax
    '''
    def __init__(self, input_image=torch.zeros(1, 7, 3, 32, 32), kernel=(3,3), stride=1, padding=1,max_kernel=(2,2), n_classes=4, n_units=128):
        super(LSTM, self).__init__()

        n_window = input_image.shape[1]
        n_channel = input_image.shape[2]

        self.conv1 = nn.Conv2d(n_channel,32,kernel,stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32,64,kernel,stride=stride,padding=padding)
        self.conv6 = nn.Conv2d(64,64,kernel,stride=stride,padding=padding)
        self.conv7 = nn.Conv2d(64,128,kernel,stride=stride,padding=padding)

        # LSTM Layer
        self.rnn = nn.RNN(4*4*128, n_units, n_window)
        self.rnn_out = torch.zeros(2, 7, 128)

        self.pool = nn.MaxPool2d((n_window,1))
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(896, n_classes)
        self.max = nn.LogSoftmax()

    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cuda()
        else:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cpu()
        for i in range(7):
            img = x[:, i]
            img = F.relu(self.conv1(img))
            img = F.relu(self.conv2(img))
            img = F.relu(self.conv3(img))
            img = F.relu(self.conv4(img))
            img = self.pool1(img)
            img = F.relu(self.conv5(img))
            img = F.relu(self.conv6(img))
            img = self.pool1(img)
            img = F.relu(self.conv7(img))
            tmp[:, i] = self.pool1(img)
            del img
        print(tmp.shape)
        x = tmp.reshape(x.shape[0], x.shape[1], 4 * 128 * 4)
        print(x.shape)
        del tmp
        self.rnn_out, _ = self.rnn(x)
        print(self.rnn_out.shape)
        x = self.rnn_out.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.max(x)
        return x

model = LSTM().to(device)
summary(model, input_size=(7, 3, 32, 32))



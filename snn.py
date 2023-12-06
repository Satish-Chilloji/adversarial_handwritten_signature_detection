# Commented out IPython magic to ensure Python compatibility.
# %pip install tensorflow_addons

# from google.colab import drive
# drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/MyDrive/Project

!unzip 'train-20231205T034516Z-001.zip' -d '/content/train/'
!unzip 'test-20231205T034513Z-001.zip' -d '/content/test/'

# Commented out IPython magic to ensure Python compatibility.
# %pip install randimage
# %pip install torchvision

from matplotlib.pyplot import show
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision
import torchvision.transforms as transforms

#load dataset
training_folder = "/content/train/train"
testing_folder = "/content/test/test"
training_csv = "/content/gdrive/My Drive/Project/train_data.csv"
testing_csv =  "/content/gdrive/My Drive/Project/test_data.csv"

class DataSets():
    def __init__(self, dir_train, path_csv, transformation_fn = None):
        # Read and transform data for testing
        self.ds = pd.read_csv(path_csv)
        self.ds.columns =["img1","img2","lab"]
        self.directory_path = dir_train
        self.transformation_fn = transformation_fn

    def __getitem__(self,index):
        # getting the image path
        path1 = os.path.join(self.directory_path, self.ds.iat[index,0])
        path2 = os.path.join(self.directory_path, self.ds.iat[index,1])

        # Get Image data
        image1 = Image.open(path1)
        image2 = Image.open(path2)

        # Convert to single channed 'L' or multi-channel mode 'P'
        image1 = image1.convert("L")
        image2 = image2.convert("L")

        # Apply image transformations
        if self.transformation_fn is not None:
            image1 = self.transformation_fn(image1)
            image2 = self.transformation_fn(image2)

        return image1, image2, torch.from_numpy(np.array([int(self.ds.iat[index,2])], dtype=np.float32))

    def __len__(self):
        return len(self.ds)

# Get data for training
train_ds = DataSets(training_folder, training_csv, transformation_fn = transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))

# Siamese Network
class SiameseNeuralNetwork(nn.Module):
    def __init__(self):
        super(SiameseNeuralNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.layers_snn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size = 3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size = 3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc_layer_snn = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128,2))

    def forward_once(self, x):
        # Forward pass
        result = self.layers_snn(x)
        result = result.view(result.size()[0], -1)

        #  Fully Connected Layer
        result = self.fc_layer_snn(result)
        return result

    def forward(self, input1, input2):
        # Forward pass with Image 1
        result1 = self.forward_once(input1)

        # Forward pass with Image 2
        result2 = self.forward_once(input2)

        return result1, result2

class Contrastive_Loss(torch.nn.Module):
      def __init__(self, margin=2.0):
            super(Contrastive_Loss, self).__init__()
            self.margin = margin

      def forward(self, result1, result2, label):
            # Get Eucledian distance btween feature vectors
            euclidean_length = F.pairwise_distance(result1, result2)
            # Calculate Contrastive Loss with distance
            loss = torch.mean((1-label) * torch.pow(euclidean_length, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_length, min=0.0), 2))

            return loss

data_loader = DataLoader(train_ds, shuffle=True, num_workers=8, pin_memory=True, batch_size=32)

snn = SiameseNeuralNetwork().cuda()

# Loss
loss_module = Contrastive_Loss()

# Optimizer
opt = torch.optim.Adam(snn.parameters(), lr=1e-3, weight_decay=0.0005)

#train the model
def train(num_epochs):
    model_losses = []
    display_counter = []
    i = 0

    for epoch in range(1,num_epochs):
        for i, data in enumerate(data_loader,0):
            image1, image2 , label = data
            image1, image2 , label = image1.cuda(), image2.cuda() , label.cuda()

            # Following code line resets the gradients
            opt.zero_grad()

            # Send both images to our SNN
            output1, output2 = snn(image1, image2)

            contrastive_loss = loss_module(output1, output2, label)
            contrastive_loss.backward() # Gradient computation
            opt.step() # Single optimization step

        print("Epoch {}\n Training loss: {}\n".format(epoch, contrastive_loss.item()))

        i = i + 10 # Increment counter to display plot in steps of 10
        display_counter.append(i)
        model_losses.append(contrastive_loss.item())
    show(display_counter, model_losses)

    # Returned trained model
    return snn

#set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

snn_model = train(50)

torch.save(snn_model.state_dict(), "../best_model_50.pt")
print("Model has been Saved.")
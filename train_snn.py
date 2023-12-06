from snn import Contrastive_Loss, SiameseNeuralNetwork, DataSets
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
testing_folder = "/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest"
training_csv = "/content/gdrive/My Drive/Project-SignatureDetection/train_data.csv"
gan_testing_csv =  "/content/gdrive/My Drive/Project-SignatureDetection/gan_test_data.csv"

# Get data for training
train_ds = DataSets(training_folder, training_csv, transformation_fn = transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))
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

snn_model = train(30)

torch.save(snn_model.state_dict(), "../best_model_30.pt")
print("Model has been Saved.")

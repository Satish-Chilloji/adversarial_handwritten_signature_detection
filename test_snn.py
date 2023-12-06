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
import matplotlib.image as mpimg

#load dataset
testing_folder = "/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest"
gan_testing_csv =  "/content/gdrive/My Drive/Project-SignatureDetection/gan_test_data.csv"

# Get data for testing
test_ds = DataSets(testing_folder, gan_testing_csv, transformation_fn = transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))
data_loader_test = DataLoader(test_ds, shuffle=True, num_workers=8, pin_memory=True, batch_size=1)

saved_snn_model = SiameseNeuralNetwork().cuda()
saved_snn_model.load_state_dict(torch.load("/content/gdrive/My Drive/Project-SignatureDetection/best_model_50.pt"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Testing
i = 0
for i, data in enumerate(data_loader_test, 0):
  image1, image2, label = data
  concat = torch.cat((image1, image2), 0)
  result1, result2 = saved_snn_model(image1.to(device), image2.to(device))

  distance = F.pairwise_distance(result1, result2)

  if label == torch.FloatTensor([[0]]):
    label="Original Signature"
  else:
    label="Forged Signature Identified"

  print("Predicted Eucledian Distance Metric: ", distance.item())

  print("Comparison for an image")

  img1 = mpimg.imread(f'/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest/real/0{i + 1}_049_real.png')

  img2 = plt.imread(f'/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest/fake/0{i + 1}_049_fake.png')

  # Display the real image in the first subplot
  axes[0].imshow(img1)
  axes[0].set_title('Real Image')
  axes[0].axis('off')

  # Display the fake image in the second subplot
  axes[1].imshow(img2)
  axes[1].set_title('Fake Image')
  axes[1].axis('off')
  plt.tight_layout()

  i= i + 1

  if i == 10:
     break

plt.show()

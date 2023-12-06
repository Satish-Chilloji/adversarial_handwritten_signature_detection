# Adversarial Handwritten Signature Detection

Project code has been placed at `https://github.com/Satish-Chilloji/adversarial_handwritten_signature_detection.git`

`train.py` is used for training the CycleGAN
`test.py` is used for inference the CycleGAN
relevant data for GAN is present in folder `datasets`
CycleGan architecture is present in  `models` folder

Run below code in Google Colab to generate adversary images and signature detection using Siamese Network:

## Import Git repository
`! git clone https://github.com/Satish-Chilloji/adversarial_handwritten_signature_detection.git`

## Import dependencies
The SNN trained models checkpoints have been kept in google drive and we need to copy below folder to be copied to root google drive directory for the user running the code:
My Drive -> Project-SignatureDetection -> models -> (Contents of below drive)
`https://drive.google.com/drive/folders/1iAWbw6KgjSMxDdECdSbOrXNXCjP_9hA1`

Below are naming conventions for saved trained models:
`best_model_<epochs>.pt`
where <epochs> represent number of epochs used in training respective model.

## Change to folder Core
```
import os
os.chdir('adversarial_handwritten_signature_detection/')
```

## Requirement installation for GAN
`!pip install -r requirements.txt`

## Training GAN,
It takes few minutes.
`!python train.py --dataroot ./datasets/userspecific_signdata --name userspecific_signdata --n_epochs 10 --n_epochs_decay 10 --model cycle_gan --display_id -1`

## Take latest trained GAN checkpoint
`cp ./checkpoints/userspecific_signdata/latest_net_G_A.pth ./checkpoints/userspecific_signdata/latest_net_G.pth`

## Generate images on test data
`!python test.py --dataroot datasets/userspecific_signdata/testA --name userspecific_signdata --model test --no_dropout`

## Cycle GAN output
```
import matplotlib.pyplot as plt

img = plt.imread('../results/userspecific_signdata/test_latest/fake/01_049_fake.png')
plt.imshow(img)

img = plt.imread('../results/userspecific_signdata/test_latest/real/01_049_real.png')
plt.imshow(img)
```

## The below codes is now running for the Siamese Neural Network

## Importing the nesseccary Packages for running the Siamese network.
```
%pip install randimage
%pip install torchvision
```

##  Import the Libraries
```
from matplotlib.pyplot import show
from PIL import Image
from skimage import io
from snn import Contrastive_Loss, SiameseNeuralNetwork, DataSets
from google.colab import drive
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F

import os
import shutil

drive.mount('/content/gdrive')
```

## This below code would configure train and test data and csv paths
```
training_folder = "/content/train/train"
testing_folder = "/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest"
training_csv = "/content/gdrive/My Drive/Project-SignatureDetection/train_data.csv"
gan_testing_csv =  "/content/gdrive/My Drive/Project-SignatureDetection/gan_test_data.csv"
```

## Organizing the images created by CycleGAN, for SNN training.
```
def organize_images(source_folder, target_folder):
    # Create the target folders if they don't exist
    real_folder = os.path.join(target_folder, 'real')
    fake_folder = os.path.join(target_folder, 'fake')

    os.makedirs(real_folder, exist_ok=True)
    os.makedirs(fake_folder, exist_ok=True)

    # List all files in the source folder
    files = os.listdir(source_folder)

    # Move files to the appropriate folders
    for file in files:
        if file.endswith('_real.png'):
            shutil.move(os.path.join(source_folder, file), os.path.join(real_folder, file))
        elif file.endswith('_fake.png'):
            shutil.move(os.path.join(source_folder, file), os.path.join(fake_folder, file))

source_folder = "/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest/images"
target_folder = "/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest/"  # Change this to your desired target folder

organize_images(source_folder, target_folder)
```


## Loading the data for SNN inference
```
test_ds = DataSets(testing_folder, gan_testing_csv, transformation_fn = transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))

data_loader_test = DataLoader(test_ds, shuffle=True, num_workers=8, pin_memory=True, batch_size=1)

saved_snn_model = SiameseNeuralNetwork().cuda()
saved_snn_model.load_state_dict(torch.load("/content/gdrive/My Drive/Project-SignatureDetection/best_model_50.pt"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
```

## Testing the SNN model
```
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

  print("Comparison for last image")

  img1 = mpimg.imread(f'/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest/real/0{i + 1}_049_real.png')

  img2 = plt.imread(f'/content/adversarial_handwritten_signature_detection/results/userspecific_signdata/test_latest/fake/0{i + 1}_049_fake.png')
```

  ## Display the real image in the first subplot
  ```
  axes[0].imshow(img1)
  axes[0].set_title('Real Image')
  axes[0].axis('off')```

  # Display the fake image in the second subplot
  ```axes[1].imshow(img2)
  axes[1].set_title('Fake Image')
  axes[1].axis('off')

  plt.tight_layout()

  i= i + 1

  if i == 10:
     break

plt.show()

```

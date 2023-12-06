# Get data for testing
test_ds = DataSets(testing_folder, testing_csv, transformation_fn = transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))

data_loader_test = DataLoader(test_ds, shuffle=True, num_workers=8, pin_memory=True, batch_size=1)

saved_snn_model = SiameseNeuralNetwork().cuda()
saved_snn_model.load_state_dict(torch.load("../best_model.pt"))
# saved_snn_model.eval()

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
  print("Result: ",label)

  i= i + 1

  if i == 10:
     break
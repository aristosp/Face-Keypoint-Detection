# V3: Display predictions
# V2: Added custom model (found in utils.py file)
# V1: Learning rate and optimizer are defined beforehand + test set is used
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchsummary import summary
from random import sample, seed
import time
# Random seed based on time
seed(time.time())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory = '/P1_Facial_Keypoints/'
data = pd.read_csv('P1_Facial_Keypoints/data/training_frames_keypoints.csv')

# Split training set to training and validation subsets
train, validation = train_test_split(data, test_size=0.2)
train_dataset = FacesDataset(train.reset_index(drop=True), 'P1_Facial_Keypoints/data/training/')
validation_dataset = FacesDataset(validation.reset_index(drop=True), 'P1_Facial_Keypoints/data/training/')
train_loader = DataLoader(train_dataset, batch_size=32)
validation_loader = DataLoader(validation_dataset, batch_size=32)

# Create model and define hyper-parameters
model = get_model()
summary(model, input_size=(3, 224, 224))
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Lists to save per epoch losses
train_loss, validation_loss = [], []
# Set epoch number and early stopping epoch number
n_epochs = 50
early_stop = 10
min_loss = torch.tensor(float('inf'))

for epoch in tqdm(range(n_epochs), desc='Epoch', unit='Epoch', position=0, leave=True):
    avg_loss = train_per_epoch(train_loader, model, optimizer, criterion)
    # Save loss for later visualizations
    train_loss.append(avg_loss)
    # Begin evaluation on validation dataset
    val_loss = prediction_mode(validation_loader, model, 'Validation', criterion)
    # Append to list for later visualizations
    validation_loss.append(val_loss)
    print("\n")
    print("Epoch {}: Training loss {:.4f}, Validation Loss {:.4f}".format(epoch+1, avg_loss, val_loss))
    # Check how training progressed in regard to previous epoch in order to implement early stopping
    if val_loss < min_loss:
        min_loss = val_loss
        best_epoch = epoch
        early_stopping(model, "../Facial Keypoints/best_model.pth", 'save')
    elif epoch - best_epoch > early_stop:
        print("Early stopped training at epoch %d" % epoch)
        early_stopping(model, "../Facial Keypoints/best_model.pth", 'restore')
        break  # terminate the training loop

# Load test data to evaluate
test_csv = pd.read_csv('P1_Facial_Keypoints/data/test_frames_keypoints.csv')
test_dataset = FacesDataset(test_csv, 'P1_Facial_Keypoints/data/test/')
test_loader = DataLoader(test_dataset, batch_size=32)
test_loss = prediction_mode(test_loader, model, 'Test', criterion)
print("Test Loss: {:.4f}".format(test_loss))

# Plot loss over training
epochs = np.arange(epoch + 1)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, validation_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')

# Find key-points of a random image
indexes = sample(range(0, len(test_csv)), k=5)
figure, axs = plt.subplots(3, 5, figsize=(10, 8))
for i in range(len(indexes)):
    idx = indexes[i]
    im = test_dataset.load_img(idx, 'P1_Facial_Keypoints/data/test/')
    axs[0, i].imshow(im)
    axs[0, i].set_title('Original Image {}'.format(i + 1))
    # Display given keypoints to compare with predicted ones
    x, kp_og = test_dataset[idx]
    axs[1, i].imshow(im)
    axs[1, i].set_title('Image {} with Given Keypoints'.format(i + 1))
    axs[1, i].scatter(kp_og[:68] * 224, kp_og[68:] * 224, c='g', marker='.')
    # Display image with predicted keypoints
    axs[2, i].imshow(im)
    axs[2, i].set_title('Image {} with Predicted keypoints'.format(i + 1))
    # Use None to convert to tensor
    kp = model(x[None]).flatten().detach().cpu()
    # first 68 are X coordinates, last 68 are Y coordinates
    axs[2, i].scatter(kp[:68] * 224, kp[68:] * 224, c='r', marker='.')
    plt.grid(False)
plt.show()

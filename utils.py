import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
from random import seed
from copy import deepcopy


seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class FacesDataset(Dataset):
    def __init__(self, df, path):
        super(FacesDataset).__init__()
        self.df = df
        # Mean and std values to normalize input to VGG16
        # These values are from the ImageNet images and are usually used
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get and read an image
        img_path = self.path + self.df.iloc[idx, 0]
        # Normalize image
        img = cv2.imread(img_path) / 255
        # Save original keypoints
        kp = deepcopy(self.df.iloc[idx, 1:].tolist())
        # Keypoints are normalized according to image dimensions, so when we resize the image,
        # the keypoints remain the same
        # Keypoints along axis X, as defined in the data, i.e. values in even columns
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        # Keypoints along axis Y, as defined in the data, i.e. values in odd columns
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2

    def preprocess_input(self, img):
        """
        :param img: Input image to resize, permute and normalize
        :return: processed image
        """
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize(img).float()
        return img.to(device)

    def load_img(self, ix, path):
        # load image and rezise, only used to see a prediction example.
        img_path = path + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = cv2.resize(img, (224, 224))
        return img


def conv_layer(ni, no, kernel_size, stride=1):
    """
    Wrapper function containing the following sequence of layers:
    Conv2D -> ReLU -> Batch Normalization -> 2D MaxPool
    :param ni: number of input channels
    :param no: number of output channels
    :param kernel_size: kernel size
    :param stride: stride
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2))


def get_model():
    """
    Simple wrapper function creating the model.
    :return: network: the model object
    """
    network = nn.Sequential(
        conv_layer(3, 64, 3),
        conv_layer(64, 64, 3),
        nn.Dropout(0.35),
        conv_layer(64, 128, 3),
        conv_layer(128, 256, 3),
        nn.Dropout(0.35),
        conv_layer(256, 512, 3),
        conv_layer(512, 512, 3),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.Dropout(0.35),
        nn.Linear(256, 136),
        nn.Sigmoid()).to(device)
    # The following comments were the original implementation:

    # model = models.vgg16(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.avgpool = nn.Sequential(nn.Conv2d(512, 512, 3),
    #                               nn.MaxPool2d(2),
    #                               nn.Flatten())
    # model.classifier = nn.Sequential(
    #     nn.Linear(2048, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(256, 136),
    #     nn.Sigmoid()
    # )
    return network


def train_per_epoch(train_dl, model, optimizer, criterion):
    """
    Per epoch training process
    :param train_dl: training set dataloader
    :param model: model to use
    :param optimizer: optimizer to be used
    :param criterion: loss function
    :return:
    train_loss: average loss per epoch
    """
    model.train(True)
    train_loss = []
    for batch_no, data in tqdm(enumerate(train_dl), total=len(train_dl), desc='Training',
                           unit='Batch', position=0, leave=True):
        # Unpack inputs and labels
        image, keypoints = data[0].to(device), data[1].to(device)
        # Zero gradients for each batch
        optimizer.zero_grad()
        # Predictions for this batch
        predicted_kp = model(image)
        # Compute loss
        loss = criterion(predicted_kp, keypoints)
        # Append batch loss
        train_loss.append(loss.item())
        # Backpropagate loss
        loss.backward()
        # Change weights
        optimizer.step()
    # Calculate mean loss and accuracy
    train_loss = np.mean(train_loss)
    return train_loss


def prediction_mode(dl, model, desc, loss_function):
    """
    Function containing the evaluation process of the model.
    :param dl: Dataloader object
    :param model: Model to be used
    :param desc: Whether the function is used during validation or testing
    :param loss_function: user defined loss function
    :return:
    avg_v_loss: loss during evaluation for one epoch
    avg_v_acc: accuracy during evaluation for one epoch
    """
    model.eval()
    with torch.no_grad():
        validation_loss = []
        # Iterate through validation data
        for _, vdata in tqdm(enumerate(dl), total=len(dl), desc=desc,
                             unit='Batch', position=0, leave=True):
            v_input, v_target = vdata[0].to(device), vdata[1].to(device)
            v_pred = model(v_input)
            # Compute val loss
            v_loss = loss_function(v_pred, v_target).item()
            validation_loss.append(v_loss)
        # Compute the loss average for one epoch
        avg_v_loss = np.mean(validation_loss)
    return avg_v_loss


def early_stopping(model, filename, mode):
    """
    Function implementing early stopping techniques, using the mode variable.
    :param model: model to save
    :param filename: path and name of the file
    :param mode: whether to save the model or restore the best model from a path
    :return: NULL
    """
    if mode == 'save':
        torch.save(model.state_dict(), filename)
    elif mode == 'restore':
        model.load_state_dict(torch.load(filename))
    else:
        print("Not valid mode")

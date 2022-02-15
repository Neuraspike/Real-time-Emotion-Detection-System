# HOW TO RUN
# python train.py --model output/model.pth --plot output/plot.png

# import the necessary libraries
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from neuraspike import config as cfg
from neuraspike import EarlyStopping
from neuraspike import LRScheduler
from torchvision import transforms
from neuraspike import EmotionNet
from torchvision import datasets
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch.optim import SGD
import torch.nn as nn
import pandas as pd
import argparse
import torch
import math

# initialize the argument parser and establish the arguments required
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Path to save the trained model')
parser.add_argument('-p', '--plot', type=str, help='Path to save the loss/accuracy plot')
args = vars(parser.parse_args())

# configure the device to use for training the model, either gpu or cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Current training device: {device}")

# initialize a list of preprocessing steps to apply on each image during
# training/validation and testing
train_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    RandomHorizontalFlip(),
    RandomCrop((48, 48)),
    ToTensor()
])

test_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])

# load all the images within the specified folder and apply different augmentation
train_data = datasets.ImageFolder(cfg.trainDirectory, transform=train_transform)
test_data = datasets.ImageFolder(cfg.testDirectory, transform=test_transform)

# extract the class labels and the total number of classes
classes = train_data.classes
num_of_classes = len(classes)
print(f"[INFO] Class labels: {classes}")

# use train samples to generate train/validation set
num_train_samples = len(train_data)
train_size = math.floor(num_train_samples * cfg.TRAIN_SIZE)
val_size = math.ceil(num_train_samples * cfg.VAL_SIZE)
print(f"[INFO] Train samples: {train_size} ...\t Validation samples: {val_size}...")

# randomly split the training dataset into train and validation set
train_data, val_data = random_split(train_data, [train_size, val_size])

# modify the data transform applied towards the validation set
val_data.dataset.transforms = test_transform

# get the labels within the training set
train_classes = [label for _, label in train_data]

# count each labels within each classes
class_count = Counter(train_classes)
print(f"[INFO] Total sample: {class_count}")

# compute and determine the weights to be applied on each category
# depending on the number of samples available
class_weight = torch.Tensor([len(train_classes) / c
                             for c in pd.Series(class_count).sort_index().values])

# initialize a placeholder for each target image, and iterate via the train dataset,
# get the weights for each class and modify the default sample weight to its
# corresponding class weight already computed
sample_weight = [0] * len(train_data)
for idx, (image, label) in enumerate(train_data):
    weight = class_weight[label]
    sample_weight[idx] = weight

# define a sampler which randomly sample labels from the train dataset
sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_data),
                                replacement=True)

# load our own dataset and store each sample with their corresponding labels
train_dataloader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, sampler=sampler)
val_dataloader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

# initialize the model and send it to device
model = EmotionNet(num_of_channels=1, num_of_classes=num_of_classes)
model = model.to(device)

# initialize our optimizer and loss function
optimizer = SGD(params=model.parameters(), lr=cfg.LR)
criterion = nn.CrossEntropyLoss()

# initialize the learning rate scheduler and early stopping mechanism
lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

# calculate the steps per epoch for training and validation set
train_steps = len(train_dataloader.dataset) // cfg.BATCH_SIZE
val_steps = len(val_dataloader.dataset) // cfg.BATCH_SIZE

# initialize a dictionary to save the training history
history = {
    "train_acc": [],
    "train_loss": [],
    "val_acc": [],
    "val_loss": []
}

# iterate through the epochs
print(f"[INFO] Training the model...")
start_time = datetime.now()

for epoch in range(0, cfg.NUM_OF_EPOCHS):

    print(f"[INFO] epoch: {epoch + 1}/{cfg.NUM_OF_EPOCHS}")

    """
    Training the model
    """
    # set the model to training mode
    model.train()

    # initialize the total training and validation loss and
    # the total number of correct predictions in both steps
    total_train_loss = 0
    total_val_loss = 0
    train_correct = 0
    val_correct = 0

    # iterate through the training set
    for (data, target) in train_dataloader:
        # move the data into the device used for training,
        data, target = data.to(device), target.to(device)

        # perform a forward pass and calculate the training loss
        predictions = model(data)
        loss = criterion(predictions, target)

        # zero the gradients accumulated from the previous operation,
        # perform a backward pass, and then update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the training loss and keep track of the number of correct predictions
        total_train_loss += loss
        train_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    """
    Validating the model
    """
    model.eval()  # disable dropout and dropout layers

    # prevents pytorch from calculating the gradients, reducing
    # memory usage and speeding up the computation time (no back prop)
    with torch.set_grad_enabled(False):

        # iterate through the validation set
        for (data, target) in val_dataloader:
            # move the data into the device used for testing
            data, target = data.to(device), target.to(device)

            # perform a forward pass and calculate the training loss
            predictions = model(data)
            loss = criterion(predictions, target)

            # add the training loss and keep track of the number of correct predictions
            total_val_loss += loss
            val_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / val_steps

    # calculate the train and validation accuracy
    train_correct = train_correct / len(train_dataloader.dataset)
    val_correct = val_correct / len(val_dataloader.dataset)

    # print model training and validation records
    print(f"train loss: {avg_train_loss:.3f}  .. train accuracy: {train_correct:.3f}")
    print(f"val loss: {avg_val_loss:.3f}  .. val accuracy: {val_correct:.3f}", end='\n\n')

    # update the training and validation results
    history['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    history['train_acc'].append(train_correct)
    history['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    history['val_acc'].append(val_correct)

    # execute the learning rate scheduler and early stopping
    validation_loss = avg_val_loss.cpu().detach().numpy()
    lr_scheduler(validation_loss)
    early_stopping(validation_loss)

    # stop the training procedure due to no improvement while validating the model
    if early_stopping.early_stop_enabled:
        break

print(f"[INFO] Total training time: {datetime.now() - start_time}...")

# move model back to cpu and save the trained model to disk
if device == "cuda":
    model = model.to("cpu")
torch.save(model.state_dict(), args['model'])

# plot the training loss and accuracy overtime
plt.style.use("ggplot")
plt.figure()
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.ylabel('Loss/Accuracy')
plt.xlabel("#No of Epochs")
plt.title('Training Loss and Accuracy on FER2013')
plt.legend(loc='upper right')
plt.savefig(args['plot'])

# evaluate the model based on the test set
model = model.to(device)
with torch.set_grad_enabled(False):
    # set the evaluation mode
    model.eval()

    # initialize a list to keep track of our predictions
    predictions = []

    # iterate through the test set
    for (data, _) in test_dataloader:
        # move the data into the device used for testing
        data = data.to(device)

        # perform a forward pass and calculate the training loss
        output = model(data)
        output = output.argmax(axis=1).cpu().numpy()
        predictions.extend(output)

# evaluate the network
print("[INFO] evaluating network...")
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names=test_data.classes))

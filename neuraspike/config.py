# import the necessary packages
import os

# initialize the path to the root folder where the dataset resides and the
# path to the train and test directory
DATASET_FOLDER = f'dataset'
trainDirectory = os.path.join(DATASET_FOLDER, "train")
testDirectory = os.path.join(DATASET_FOLDER, "test")

# initialize the amount of samples to use for training and validation
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# specify the batch size, total number of epochs and the learning rate
BATCH_SIZE = 16
NUM_OF_EPOCHS = 50
LR = 1e-1

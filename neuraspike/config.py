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





















text = """

$ python3 train.py --model output/model.pth --plot output/plot.png

[INFO] Current training device: cuda
[INFO] Class labels: ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
[INFO] Train samples: 25838 ...	 Validation samples: 2871...
[INFO] Total sample: Counter({2: 6495, 3: 4493, 4: 4347, 0: 3973, 1: 3677, 5: 2853})
[INFO] Training the model...
[INFO] epoch: 1/50
train loss: 2.113  .. train accuracy: 0.174
val loss: 1.879  .. val accuracy: 0.152

[INFO] epoch: 2/50
train loss: 1.981  .. train accuracy: 0.212
val loss: 2.055  .. val accuracy: 0.179

[INFO] Early stopping: 1/10... 


[INFO] epoch: 3/50
train loss: 1.857  .. train accuracy: 0.264
val loss: 1.609  .. val accuracy: 0.336

[INFO] epoch: 4/50
train loss: 1.591  .. train accuracy: 0.361
val loss: 1.436  .. val accuracy: 0.423

[INFO] epoch: 5/50
train loss: 1.439  .. train accuracy: 0.426
val loss: 1.327  .. val accuracy: 0.483

[INFO] epoch: 6/50
train loss: 1.382  .. train accuracy: 0.449
val loss: 1.423  .. val accuracy: 0.441

[INFO] Early stopping: 1/10... 


[INFO] epoch: 7/50
train loss: 1.338  .. train accuracy: 0.473
val loss: 1.298  .. val accuracy: 0.498

[INFO] epoch: 8/50
train loss: 1.310  .. train accuracy: 0.485
val loss: 1.268  .. val accuracy: 0.508

[INFO] epoch: 9/50
train loss: 1.291  .. train accuracy: 0.495
val loss: 1.246  .. val accuracy: 0.512

[INFO] epoch: 10/50
train loss: 1.272  .. train accuracy: 0.505
val loss: 1.331  .. val accuracy: 0.490

[INFO] Early stopping: 1/10... 


[INFO] epoch: 11/50
train loss: 1.249  .. train accuracy: 0.519
val loss: 1.238  .. val accuracy: 0.525

[INFO] epoch: 12/50
train loss: 1.231  .. train accuracy: 0.521
val loss: 1.236  .. val accuracy: 0.520

[INFO] epoch: 13/50
train loss: 1.214  .. train accuracy: 0.529
val loss: 1.247  .. val accuracy: 0.515

[INFO] Early stopping: 1/10... 


[INFO] epoch: 14/50
train loss: 1.204  .. train accuracy: 0.530
val loss: 1.241  .. val accuracy: 0.531

[INFO] Early stopping: 2/10... 


[INFO] epoch: 15/50
train loss: 1.188  .. train accuracy: 0.542
val loss: 1.187  .. val accuracy: 0.548

[INFO] epoch: 16/50
train loss: 1.171  .. train accuracy: 0.547
val loss: 1.197  .. val accuracy: 0.545

[INFO] Early stopping: 1/10... 


[INFO] epoch: 17/50
train loss: 1.168  .. train accuracy: 0.551
val loss: 1.159  .. val accuracy: 0.545

[INFO] epoch: 18/50
train loss: 1.148  .. train accuracy: 0.554
val loss: 1.175  .. val accuracy: 0.553

[INFO] Early stopping: 1/10... 


[INFO] epoch: 19/50
train loss: 1.127  .. train accuracy: 0.569
val loss: 1.180  .. val accuracy: 0.553

[INFO] Early stopping: 2/10... 


[INFO] epoch: 20/50
train loss: 1.120  .. train accuracy: 0.571
val loss: 1.202  .. val accuracy: 0.550

[INFO] Early stopping: 3/10... 


[INFO] epoch: 21/50
train loss: 1.120  .. train accuracy: 0.574
val loss: 1.176  .. val accuracy: 0.557

[INFO] Early stopping: 4/10... 


[INFO] epoch: 22/50
train loss: 1.100  .. train accuracy: 0.582
val loss: 1.188  .. val accuracy: 0.548

[INFO] Early stopping: 5/10... 


[INFO] epoch: 23/50
train loss: 1.102  .. train accuracy: 0.581
val loss: 1.170  .. val accuracy: 0.572

Epoch    23: reducing learning rate of group 0 to 5.0000e-02.
[INFO] Early stopping: 6/10... 


[INFO] epoch: 24/50
train loss: 1.005  .. train accuracy: 0.617
val loss: 1.120  .. val accuracy: 0.568

[INFO] epoch: 25/50
train loss: 0.975  .. train accuracy: 0.631
val loss: 1.104  .. val accuracy: 0.576

[INFO] epoch: 26/50
train loss: 0.965  .. train accuracy: 0.633
val loss: 1.119  .. val accuracy: 0.570

[INFO] Early stopping: 1/10... 


[INFO] epoch: 27/50
train loss: 0.954  .. train accuracy: 0.642
val loss: 1.136  .. val accuracy: 0.594

[INFO] Early stopping: 2/10... 


[INFO] epoch: 28/50
train loss: 0.949  .. train accuracy: 0.645
val loss: 1.127  .. val accuracy: 0.592

[INFO] Early stopping: 3/10... 


[INFO] epoch: 29/50
train loss: 0.938  .. train accuracy: 0.651
val loss: 1.132  .. val accuracy: 0.588

[INFO] Early stopping: 4/10... 


[INFO] epoch: 30/50
train loss: 0.917  .. train accuracy: 0.656
val loss: 1.161  .. val accuracy: 0.572

[INFO] Early stopping: 5/10... 


[INFO] epoch: 31/50
train loss: 0.913  .. train accuracy: 0.657
val loss: 1.163  .. val accuracy: 0.581

Epoch    31: reducing learning rate of group 0 to 2.5000e-02.
[INFO] Early stopping: 6/10... 


[INFO] epoch: 32/50
train loss: 0.855  .. train accuracy: 0.681
val loss: 1.119  .. val accuracy: 0.597

[INFO] Early stopping: 7/10... 


[INFO] epoch: 33/50
train loss: 0.840  .. train accuracy: 0.685
val loss: 1.142  .. val accuracy: 0.597

[INFO] Early stopping: 8/10... 


[INFO] epoch: 34/50
train loss: 0.826  .. train accuracy: 0.689
val loss: 1.129  .. val accuracy: 0.589

[INFO] Early stopping: 9/10... 


[INFO] epoch: 35/50
train loss: 0.816  .. train accuracy: 0.693
val loss: 1.165  .. val accuracy: 0.591

[INFO] Early stopping: 10/10... 


[INFO] Early stopping enabled
[INFO] Time taken to complete training the model: 0:14:00.241209...
[INFO] evaluating network...
              precision    recall  f1-score   support

       angry       0.52      0.60      0.56      1069
        fear       0.44      0.36      0.39      1024
       happy       0.86      0.75      0.80      1774
     neutral       0.54      0.53      0.53      1233
         sad       0.43      0.55      0.48      1247
    surprise       0.76      0.73      0.74       831

    accuracy                           0.59      7178
   macro avg       0.59      0.58      0.59      7178
weighted avg       0.61      0.59      0.60      7178

"""

# print(text)
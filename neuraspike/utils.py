# import the necessary libraries
from torch.optim import lr_scheduler
import cv2


class LRScheduler:
    """
    Check if the validation loss does not decrease for a given number of epochs
    (patience), then decrease the learning rate by a given 'factor'
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        :returns:  new_lr = old_lr * factor
        """

        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min",
                                                           patience=self.patience,
                                                           factor=self.factor,
                                                           min_lr=self.min_lr,
                                                           verbose=True)

    def __call__(self, validation_loss):
        self.lr_scheduler.step(validation_loss)


class EarlyStopping:
    """
    Early stopping breaks the training procedure when the loss does not improve
    over a certain number of iterations
    """

    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: number of epochs to wait stopping the training procedure
        :param min_delta: the minimum difference between (previous and the new loss)
                           to consider the network is improving.
        """

        self.early_stop_enabled = False
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def __call__(self, validation_loss):

        # update the validation loss if the condition doesn't hold
        if self.best_loss is None:
            self.best_loss = validation_loss

        # check if the training procedure should be stopped
        elif (self.best_loss - validation_loss) < self.min_delta:
            self.counter += 1
            print(f"[INFO] Early stopping: {self.counter}/{self.patience}... \n\n")

            if self.counter >= self.patience:
                self.early_stop_enabled = True
                print(f"[INFO] Early stopping enabled")

        # reset the early stopping counter
        elif (self.best_loss - validation_loss) > self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # check if the width and height is specified
    if width is None and height is None:
        return image

    # initialize the dimension of the image and grab the
    # width and height of the image
    dimension = None
    (h, w) = image.shape[:2]

    # calculate the ratio of the height and
    # construct the new dimension
    if height is not None:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))

    # resize the image
    resized_image = cv2.resize(image, dimension, interpolation=inter)

    return resized_image

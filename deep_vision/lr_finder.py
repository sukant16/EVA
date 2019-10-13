from matplotlib import pyplot as plt
import keras.backend as K
import numpy as np


class LR_Finder(Callback):

    def __init__(self, start_lr=1e-5, end_lr=10, step_size=None, beta=.98):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.step_size = step_size
        self.beta = beta
        self.lr_mult = (end_lr/start_lr)**(1/step_size)

    def on_train_begin(self, logs=None):
        self.best_loss = 1e9
        self.avg_loss = 0
        self.losses, self.smoothed_losses, self.lrs, self.iterations = ([], [],
                                                                        [], [])
        self.iteration = 0
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        self.iteration += 1

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta**self.iteration)

        # Check if the loss is not exploding
        if self.iteration > 1 and smoothed_loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if smoothed_loss < self.best_loss or self.iteration == 1:
            self.best_loss = smoothed_loss

        lr = self.start_lr * (self.lr_mult**self.iteration)

        self.losses.append(loss)
        self.smoothed_losses.append(smoothed_loss)
        self.lrs.append(lr)
        self.iterations.append(self.iteration)
        K.set_value(self.model.optimizer.lr, lr)

    def plot_lr(self):
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')
        plt.plot(self.iterations, self.lrs)

    def plot_lr_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log',
                     y_lim=None):
        plt.ylabel('Loss')
        plt.xlabel('Learning rate (log scale)')
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end],
                 self.losses[n_skip_beginning:-n_skip_end])
        plt.ylim(y_lim)
        plt.xscale('log')

    def plot_smoothed_lr_loss(self, n_skip_beginning=10, n_skip_end=5,
                              x_scale='log', y_lim=None):
        plt.ylabel('Smoothed Losses')
        plt.xlabel('Learning rate (log scale)')
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end],
                 self.smoothed_losses[n_skip_beginning:-n_skip_end])
        plt.ylim(y_lim)
        plt.xscale(x_scale)

    def plot_iter_loss(self, n_skip_beginning=10, n_skip_end=5):
        plt.ylabel('Losses')
        plt.xlabel('Iterations')
        plt.plot(self.iterations[n_skip_beginning:-n_skip_end],
                 self.losses[n_skip_beginning:-n_skip_end])

    def get_derivatives(self, smooth_loss=True, lr_begin=0.001, lr_end=1):
        '''
        returns: tuple of array of derivative of loss w.r.t lr,
                 lr array and loss array in the specified range of lr
        parameters:
          smooth: whether to use smooth loss
          lr_begin & lr_end: these learning rates specify the range in which to
                        calculate derivative of the loss w.r.t learning rate
        '''
        lr_complete_vector = np.array(self.lrs)
        if lr_begin is not None and lr_end is not None:
            indices = np.where((lr_complete_vector > lr_begin) & (lr_complete_vector < lr_end))
            lr_vector = np.array(lr_complete_vector[indices])
            if smooth_loss:
                loss_vector = np.array(self.smoothed_losses)[indices]
            else:
                loss_vector = np.arrat(self.losses)[indices]
        elif lr_begin is not None and lr_end is None:
            indices = np.where(lr_complete_vector > lr_begin)
            lr_vector = np.array(lr_complete_vector[indices])
            if smooth_loss:
                loss_vector = np.array(self.smoothed_losses)[indices]
            else:
                loss_vector = np.arrat(self.losses)[indices]
        else:
            indices = np.where(lr_complete_vector < lr_end)
            lr_vector = np.array(lr_complete_vector[indices])
            if smooth_loss:
                loss_vector = np.array(self.smoothed_losses)[indices]
            else:
                loss_vector = np.arrat(self.losses)[indices]
        der_vector = np.gradient(lr_vector, loss_vector)
        return der_vector, lr_vector, loss_vector

    def get_best_lr(self, der_vector, lr_vector):
        '''
        returns: learing rate at which loss change is maximum
        parameters:
            1. der_vector: array of the derivatives of loss w.r.t lr
            2. lr_vector: array of lr
        '''
        idx = np.argmax(der_vector)
        return lr_vector[idx]

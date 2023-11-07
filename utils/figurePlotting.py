import os
import matplotlib.pyplot as plt

class figurePlotting:
    def __init__(self, dataset):
        super(figurePlotting, self).__init__()
        self.dataset = dataset
    
    def plot_accu_figure(self, status, filename, tol_epochs, accu_arr, modelName):
        plt.figure()
        if status == 'train':
            plt.title("Training accuracy for " + self.dataset + " on " + modelName)
        elif status == 'test':
            plt.title("Testing accuracy for " + self.dataset + " on " + modelName)
        plt.xlabel("Epochs") # x label
        plt.ylabel("Accuracy(%)") # y label
        plt.plot(tol_epochs, accu_arr, color='blue')
        plt.savefig(filename, format = "png")
        plt.close()

    def plot_loss_figure(self, status, filename, tol_epochs, loss_arr, modelName):
        plt.figure()
        if status == 'train':
            plt.title("Training loss for " + self.dataset + " on " + modelName)
        elif status == 'test':
            plt.title("Testing loss for " + self.dataset + " on " + modelName)
        plt.xlabel("Epochs") # x label
        plt.ylabel("Loss Value (Cross Entropy)") # y label
        plt.plot(tol_epochs, loss_arr, color='red')
        plt.savefig(filename, format = "png")
        plt.close()
    
    def plotFigure(self, status, types, weights_to, tol_epochs, accu_arr, loss_arr, modelName):
        if types == "initialize train":
            train_accufig = status + '_accu.png'
            train_lossfig = status + '_loss.png'
        elif types == "re-train":
            train_accufig = 're-' + status + '_accu.png'
            train_lossfig = 're-' + status + '_loss.png'
        # plot the loss and accuracy figure for training
        save_train_accu_dir = os.path.join(weights_to, train_accufig)
        save_train_loss_dir = os.path.join(weights_to, train_lossfig)
        self.plot_accu_figure(status, save_train_accu_dir, tol_epochs, accu_arr, modelName)
        self.plot_loss_figure(status, save_train_loss_dir, tol_epochs, loss_arr, modelName)

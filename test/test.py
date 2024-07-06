import matplotlib.pyplot as plt
import numpy as np
import time

class MatplotlibProgress:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Turn on interactive mode

    def update(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.plot()

    def plot(self):
        self.ax.clear()
        self.ax.plot(self.epochs, self.train_losses, label='Training Loss')
        self.ax.plot(self.epochs, self.val_losses, label='Validation Loss')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Example usage with dummy training loop
def train_model():
    mp_progress = MatplotlibProgress()

    for epoch in range(1, 11):
        # Simulate training and validation loss
        train_loss = np.exp(-epoch / 10) + np.random.rand() * 0.1
        val_loss = np.exp(-epoch / 10) + np.random.rand() * 0.1

        # Update progress
        mp_progress.update(epoch, train_loss, val_loss)

        # Simulate time delay for training
        time.sleep(1)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot

if __name__ == "__main__":
    train_model()

import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Where to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss, avg_val_loss):
        if val_loss < self.best_loss - (self.min_delta*avg_val_loss):
            self.best_loss = val_loss
            self.counter = 0
            #torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

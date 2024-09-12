import torch.utils.data as data
import torch

class MyDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

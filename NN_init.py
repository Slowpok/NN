import torch

size_of_array = 50
learning_rate = 0.001
batch_size = 10000
finename = r"Материалы датасет4.csv"
num_classes = 2

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")


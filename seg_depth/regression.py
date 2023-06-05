import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset

class DummyData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self._degree = degree
        self.linear = nn.Linear(self._degree, 1)

    def forward(self, x):
        return self.linear(self._polynomial_features(x))

    def _polynomial_features(self, x):
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, self._degree + 1)], 1)

def train_step(model, data, optimizer, criterion):
    running_loss = 0.0

    for i, (x, y) in enumerate(data):

        x_ = Variable(x, requires_grad=True)
        y_ = Variable(y.unsqueeze(1)) # unsqueeze to match dimensions with y_pred later

        def closure():
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_)

            # Compute loss
            loss = criterion(y_pred, y_)

            # Backward pass
            loss.backward()

            return loss

        # Update weights
        optimizer.step(closure)

        # Update the running loss
        loss = closure()
        running_loss += loss.item()
    return running_loss
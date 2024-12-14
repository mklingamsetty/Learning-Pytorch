'''
    Now that we have learned how to use PyTorch to make a LinearRegression model, 
    let's try to make a model from scratch, plot it, and incorporate everything
    we have learned so far.
'''

import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(1.0, requires_grad=True)
        self.bias = torch.randn(0.0, requires_grad=True)
    
    def forward(self, x) -> torch.Tensor:
        return self.weight * x + self.bias

def trainAndTest(model, x, y, y_pred, x_train, x_test, y_train, y_test):
    # Define our loss function
    loss = nn.L1Loss()

    # Define our optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Train our model
    # 1. Loop through the data
    epochs = 1000
    for epoch in range(epochs):
        

def plotGraph(x, y, y_pred):


def main():
    torch.manual_seed(42)

    model = LinearRegression()
    
    # Create our dataset
    weight = 6.35
    bias = 3.9
    x = torch.arange(1, 50, 1.5).unsqueeze(1)
    y = weight * x + bias

    # Split our dataset into training and testing
    training_size = int(0.8 * len(x))
    x_train, x_test = x[:training_size], x[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    # Make predictions
    y_pred = model(x)

    # Train and test our model
    trainAndTest(model, x, y, y_pred, x_train, x_test, y_train, y_test)


    # Start making predictions
    y_pred = model(x)


if __name__ == "__main__":
    main()
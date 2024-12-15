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
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

def trainAndTest(model, y_pred, x_train, x_test, y_train, y_test):
    # Define our loss function
    loss = nn.L1Loss()

    # Define our optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

    ''' Training Loop'''
    # 1. Loop through the data
    epochs = 1000
    for epoch in range(epochs):
        
        # 2. Forward pass
        y_pred = model(x_train)
        
        #3 Calculate the loss
        train_loss = loss(y_pred, y_train)
        
        #4 Zero the gradients
        optimizer.zero_grad()
        
        #5 Backward pass
        train_loss.backward()
        
        #6 Optimizer step (gradient descent)
        optimizer.step()
        
        ''' Testing Loop '''
        
        # 2. Forward pass
        with torch.inference_mode():
            y_pred = model(x_test)
            
        # 3. Calculate the loss
        test_loss = loss(y_pred, y_test)
        print(f'Epoch {epoch} Training Loss: {train_loss.item()} Testing Loss: {test_loss.item()}' if epoch % 100 == 0 else '', end='\r')
        
    plotGraph(x_train, y_train, x_test, y_test, y_pred)
        

def plotGraph(x_train, y_train, x_test, y_test, y_pred):
    plt.scatter(x_train, y_train, label='Training Data', color='blue')
    plt.scatter(x_test, y_test, label='Testing Data', color='green')
    if y_pred is not None:
        plt.scatter(x_test, y_pred, label='Model Prediction', color='red')
    plt.show()


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
    
    plotGraph(x_train, y_train, x_test, y_test, None)

    # Make predictions
    with torch.inference_mode():
        y_pred = model(x_test)
    
    plotGraph(x_train, y_train, x_test, y_test, y_pred)

    # Train and test our model
    trainAndTest(model, y_pred, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
from libraries import *
from initalizeData import *

class CNNModel(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):

        '''
        What are these parameters?
        - input_shape: The number of channels in the input data.
            -> If an image has 3 channels (RGB), then the input_shape is 3.
            -> If an image has 1 channel (grayscale), then the input_shape is 1.

        - hidden_units: The number of neurons in the hidden layer.
            -> This is the layer that is between the input and output layers.
            -> The more hidden units, the more accurate the model, but the longer it takes to train. 
             
        - output_shape: The number of classes in the output layer.
            -> Basically the number of items you want to classify.
                -> we will pass in the Hyperparameter: NUMBER_OF_LABELS from initializeData.py
        '''

        super().__init__()
        self.convolutional_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape, 
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1),
            nn.ReLU(),
            # nn.Conv2d(
            #     in_channels=hidden_units, 
            #     out_channels=hidden_units, 
            #     kernel_size=3, 
            #     stride=1, 
            #     padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convolutional_layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1),
            nn.ReLU(),
            # nn.Conv2d(
            #     in_channels=hidden_units, 
            #     out_channels=hidden_units, 
            #     kernel_size=3, 
            #     stride=1, 
            #     padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units*16*16, 
                out_features=output_shape)
        )
        
    def forward(self, x):
        x = self.convolutional_layer_1(x)
        x = self.convolutional_layer_2(x)
        x = self.classifier(x)
        return x
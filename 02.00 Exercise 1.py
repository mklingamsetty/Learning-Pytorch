'''
Develop a Binary Classification model on your own
'''

import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

def calculate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=128)
        self.layer_2 = nn.Linear(in_features=128, out_features=512)
        self.layer_3 = nn.Linear(in_features=512, out_features=1024)
        self.layer_4 = nn.Linear(in_features=1024, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))

def main():
    torch.manual_seed(42)
    x, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
    # x -> 2D List

    plt.scatter(x[:, 0], # First index of x
            x[:, 1], # Second index of x
            c=y, 
            cmap=plt.cm.RdYlBu)
    plt.show()
    x = torch.from_numpy(x).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = BinaryClassificationModel()

    # determine loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1)
    
    # train the model
    epochs = 1000
    for epoch in range(epochs):

        ''' Training Loop '''

        model.train()
        
        y_logits = model(x_train).squeeze(dim=1)
        y_preds = torch.round(torch.sigmoid(y_logits))

        training_loss = loss_fn(y_logits, y_train)
        acc = calculate_accuracy(y_train, y_preds)

        optimizer.zero_grad()

        training_loss.backward()

        optimizer.step()

        ''' Testing Loop '''
        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze(dim=1)
            test_preds = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = calculate_accuracy(y_test, test_preds)

        if epoch % 100 == 0:
            print("Training Loss: ", training_loss, " ; Training Accuracy: ", acc)
            print("Testing Loss: ", test_loss, " ; Test Accuracy: ", test_acc)




if __name__ == "__main__":
    main()
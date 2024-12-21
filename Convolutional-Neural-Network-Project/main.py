'''
- The data we're going to be using is a subset of the Food101 dataset.
- Food101 is popular computer vision benchmark as it contains 1000 images 
    of 101 different kinds of foods, totaling 101,000 images 
    (75,750 train and 25,250 test).
'''
from libraries import *
from initalizeData import *
from cnnModel import *
from helperFunctions import *

# More Hyperparameters and Constants
CLASSES = train_dataframe.keys()
CLASS_DICT = train_data.class_to_idx

def train_loop(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader,
    loss_function:torch.nn.Module, optimizer:torch.optim.Optimizer, 
    accuracy_function:calculate_accuracy):
    
    model_accuracy, model_loss = 0, 0
    model.train()
    for batch, (x_train, y_train) in enumerate(train_dataloader):
        # 1. Forward Pass through Neural Network
        train_prediction = model(x_train)

        # 2. Loss Calculation
        loss = loss_function(train_prediction, y_train)
        model_loss += loss.item()

        # 3. Zero the gradients
        optimizer.zero_grad()

        # 4. Backward Pass through Neural Network
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()

        # 6. Calculate Accuracy
        predicted_class = torch.argmax(torch.softmax(train_prediction, dim=1), dim=1)
        model_accuracy += (predicted_class == y_train).sum().item()/len(train_prediction)

    # 7. Scale accuracy and loss to match dataloader
    model_loss = model_loss / len(train_dataloader)
    model_accuracy = model_accuracy / len(train_dataloader)
    return model_accuracy, model_loss

def test_loop(model:torch.nn.Module, test_dataloader:torch.utils.data.DataLoader,
    loss_function:torch.nn.Module, accuracy_function:calculate_accuracy):
    
    model_accuracy, model_loss = 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (x_test, y_test) in enumerate(test_dataloader):
            # 1. Forward Pass through Neural Network
            test_prediction = model(x_test)

            # 2. Loss Calculation
            loss = loss_function(test_prediction, y_test)
            model_loss += loss.item()

            # 3. Calculate Accuracy
            predicted_class = torch.argmax(torch.softmax(test_prediction, dim=1), dim=1)
            model_accuracy += (predicted_class == y_test).sum().item()/len(test_prediction)
    
    model_loss = model_loss / len(test_dataloader)
    model_accuracy = model_accuracy / len(test_dataloader)
    return model_accuracy, model_loss

def learn(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, 
    test_dataloader:torch.utils.data.DataLoader, loss_function:torch.nn.Module, 
    optimizer:torch.optim.Optimizer, accuracy_function:calculate_accuracy):

    # 1. Record Results in a Dictionary
    results = {
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": []
    }

    # 2. Train the Model for EPOCHS
    for epoch in tqdm(range(EPOCHS)):
        train_accuracy, train_loss = train_loop(model, train_dataloader, loss_function, optimizer, accuracy_function)
        test_accuracy, test_loss = test_loop(model, test_dataloader, loss_function, accuracy_function)

        # 3. Print out results for each epoch
        print(f"\nEpoch: {epoch+1}/{EPOCHS}")
        print(f"Train Accuracy: {train_accuracy:.2f}%; Train Loss: {train_loss:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%; Test Loss: {test_loss:.2f}")
        print("\n")

        # 4. Append Results to Dictionary
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_accuracy"].append(train_accuracy.item() if isinstance(train_accuracy, torch.Tensor) else train_accuracy)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_accuracy"].append(test_accuracy.item() if isinstance(test_accuracy, torch.Tensor) else test_accuracy)


    return results

# Initalize the model
def main():
    convolutional_neural_network = CNNModel(
        input_shape=3, 
        hidden_units=10, 
        output_shape=NUMBER_OF_LABELS)
    
    # Loss Function and Optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        convolutional_neural_network.parameters(), 
        lr=LEARNING_RATE)
    
    start_time = timer()
    # Train the Model
    results = learn(
        model=convolutional_neural_network, 
        train_dataloader=train_loader,
        test_dataloader=test_loader, 
        loss_function=lossFunction, 
        optimizer=optimizer, 
        accuracy_function=calculate_accuracy)
    
    # Plot the Loss Curves
    plot_loss_curves(results)

    end_time = timer()
    print_train_time(start_time, end_time)
    

if __name__ == "__main__":
    main()
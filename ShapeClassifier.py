import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5) # Applies 6 5x5 kernels
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max-pooling 
        self.conv2 = nn.Conv2d(6, 16, 5) # Applies 16 5x5 kernels
       
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 10 * 10, 120) # Input dense layer
        self.fc2 = nn.Linear(120, 84) # Hidden dense layer
        self.fc3 = nn.Linear(84, 5) # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Passes in the data from the train_loader in order to train the network and evaluate loss
def train_model(model, train_loader, optimizer, criterion):
    for epoch in range(1):
        print("Starting Epoch: {}".format(epoch+1)) 
        for i, data in enumerate(train_loader, 0):
            model.train()
            inputs, labels = data # data is a list of [inputs, labels]
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


# Validates the data from the test_loader in order to evaluate its accuracy
def validate_model(model, val_loader):
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            output = model(inputs)
            for image_id, i in enumerate(output):
                if torch.argmax(i) == labels[image_id]:
                    correct += 1  # Increment total number of images validated correctly
                count += 1  # Increment total number of images validated
    acc = round(correct/count, 2)  # Calculates accuracy
    return acc


def main():
    print("Shape Classifier...")
    print("...")
    train_path = "./shapes/training/"
    val_path = "./shapes/validation/"
    test_path = "./shapes/testing/"
    PATH = "state_dict_model.pt"

    transform = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1), transforms.Resize((54,54)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_data = ImageFolder(root=train_path, transform=transform)
    val_data = ImageFolder(root=val_path, transform=transform)
    test_data = ImageFolder(root=test_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


    print("Number of training samples: ", len(train_data))
    print("Number of validation samples: ", len(val_data))
    print("Number of testing samples: ", len(test_data))

    print("Detected Classes are: ", train_data.class_to_idx)

    shapes = ["circle", "pentagon", "square", "star", "triangle"] # Array to store shape classiciation 

    model = Net()  # Create object of the neural network model
    model.load_state_dict(torch.load(PATH)) # Loads model if previously trained and saved
    model.eval()

    # Set the models parameters default learning rate to 0.001
    criterion = nn.CrossEntropyLoss() # Cross Entropy Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, optimizer, criterion)  # Trains the network using the training dataset
    acc = validate_model(model, test_loader)  # Validates data and returns accuracy percentage

    print(acc)
    print("Done!")

    torch.save(model.state_dict(), PATH) # Save the model

    while True:
        path = input("Please enter a filepath:\n")  # Hand drawn image (digitally or on paper) loaded using file path 
        # Example:
        # ./shapes/testing2/square/IMG-0991.jpg
        # Breaks loop if user input is "exit"
        if path == "exit":
            print("Exiting...")
            break
        else:
            image = Image.open(path)  # Reads in image at the user's input path
            image = image.resize((54,54)) # Resizes the image to 54x54
            image = ImageOps.grayscale(image) # Changes to grayscale if RGB
            image = ToTensor()(image).unsqueeze(0)  # Converts image to tensor
            image = Variable(image)
            image_tensor = torch.argmax(model(image))
            output = str(image_tensor)  # Type cast to string
            output = output[-2:-1]  # Substring to obtain classification number
            output = int(output) # Type cast to int
            print("This shape is a " + shapes[output] + ".")


if __name__ == '__main__':
    main()
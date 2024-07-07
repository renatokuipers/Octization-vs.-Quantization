import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define the Quantization Layer
class QuantizeLayer(nn.Module):
    def __init__(self, num_bits=8):
        super(QuantizeLayer, self).__init__()
        self.num_bits = num_bits

    def forward(self, x):
        scale = 2 ** self.num_bits - 1
        return torch.round(x * scale) / scale

# Define the Octal Quantization Layer
class OctalQuantizeLayer(nn.Module):
    def __init__(self, num_bits=8):
        super(OctalQuantizeLayer, self).__init__()
        self.num_bits = num_bits
        self.scale = 2 ** num_bits - 1

    def forward(self, x):
        # Scale input to [0, scale]
        x = x * self.scale
        
        # Soft quantization to octal
        octal = torch.zeros_like(x)
        for i in range(8):  # 8 octal digits for 24-bit representation
            digit = torch.clamp(x / (8 ** i), 0, 7)
            octal += digit * (10 ** i)
            x = x - digit * (8 ** i)
        
        # Normalize back to [0, 1] range
        return octal / (10 ** 8 - 1)

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self, quantization_layer):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.quantize = quantization_layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.quantize(x)
        x = self.fc2(x)
        return x

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def train_model(model, criterion, optimizer, trainloader, epochs=20):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    return train_losses

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == '__main__':
    # Load data
    trainloader, testloader = load_data()

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate Standard Quantization Model
    quant_model = CNNModel(QuantizeLayer(num_bits=8)).to(device)
    quant_optimizer = optim.Adam(quant_model.parameters(), lr=0.001)
    print("\n##################################")
    print("Training Standard Quantization Model")
    start_time = time.time()
    quant_train_losses = train_model(quant_model, criterion, quant_optimizer, trainloader)
    quant_inference_time = time.time() - start_time
    quant_accuracy = evaluate_model(quant_model, testloader)
    quant_model_size = sum(p.numel() for p in quant_model.parameters() if p.requires_grad)
    print("##################################\n")

    # Train and evaluate Octization Model
    oct_model = CNNModel(OctalQuantizeLayer(num_bits=8)).to(device)
    oct_optimizer = optim.Adam(oct_model.parameters(), lr=0.001)
    print("\n##################################")
    print("Training Octization Model")
    start_time = time.time()
    oct_train_losses = train_model(oct_model, criterion, oct_optimizer, trainloader)
    oct_inference_time = time.time() - start_time
    oct_accuracy = evaluate_model(oct_model, testloader)
    oct_model_size = sum(p.numel() for p in oct_model.parameters() if p.requires_grad)
    print("##################################\n")

    # Print Results
    print("\n##################################")
    print(f"Standard Quantization CNN - Test Accuracy: {quant_accuracy:.2f}%, Inference Time: {quant_inference_time:.2f}s, Model Size: {quant_model_size}")
    print(f"Octization CNN - Test Accuracy: {oct_accuracy:.2f}%, Inference Time: {oct_inference_time:.2f}s, Model Size: {oct_model_size}")
    print("##################################\n")

    # Plot training losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(quant_train_losses, label='Quantization')
    plt.title('Standard Quantization CNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(oct_train_losses, label='Octization')
    plt.title('Octization CNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

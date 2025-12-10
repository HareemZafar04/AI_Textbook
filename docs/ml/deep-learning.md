---
sidebar_label: Deep Learning
---

# Deep Learning

Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to learn complex patterns from large amounts of data. It has revolutionized fields such as computer vision, natural language processing, and speech recognition.

## Overview of Deep Learning

Deep learning is inspired by the human brain's neural networks. It uses artificial neural networks with multiple layers between the input and output layers. The "depth" of these networks allows them to learn hierarchical representations of data, with each layer learning more abstract features than the previous one.

### Key Characteristics:
- **Hierarchical Feature Learning**: Learns features automatically without manual feature engineering
- **End-to-End Learning**: Can learn from raw input to final output without intermediate manual steps
- **Large-Scale Data Handling**: Excels with large datasets
- **Versatility**: Applicable to various types of data (images, text, audio, time series)

## Neural Network Fundamentals

### Basic Neural Network Components
```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        # Backward propagation
        m = X.shape[0]  # number of samples
        
        # Calculate gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs, learning_rate):
        costs = []
        for i in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate cost (mean squared error)
            cost = np.mean((output - y) ** 2)
            costs.append(cost)
            
            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2, learning_rate)
            
            if i % 100 == 0:
                print(f"Epoch {i}, Cost: {cost:.6f}")
        
        return costs

# Example: Learning XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
costs = nn.train(X, y, epochs=1000, learning_rate=1.0)

# Test the network
predictions = nn.forward(X)
print("\nPredictions after training:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")
```

## Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation_functions():
    x = np.linspace(-5, 5, 100)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Tanh
    tanh = np.tanh(x)
    
    # ReLU
    relu = np.maximum(0, x)
    
    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, x * 0.01)
    
    # ELU
    alpha = 1.0
    elu = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(x, sigmoid, label='Sigmoid')
    plt.title('Sigmoid')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(x, tanh, label='Tanh', color='orange')
    plt.title('Tanh')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(x, relu, label='ReLU', color='green')
    plt.title('ReLU')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(x, leaky_relu, label='Leaky ReLU', color='red')
    plt.title('Leaky ReLU')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(x, elu, label='ELU', color='purple')
    plt.title('ELU')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_activation_functions()
```

## Deep Learning with PyTorch

### Basic Neural Network in PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # Add dropout for regularization
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer (no activation or dropout)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example: Training a model for binary classification
def train_pytorch_model():
    # Generate sample data
    n_samples = 1000
    input_size = 20
    X = torch.randn(n_samples, input_size)
    y = (X.sum(dim=1) > 0).float().unsqueeze(1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = DeepNeuralNetwork(input_size, [64, 32, 16], 1)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model

# Train the model
# model = train_pytorch_model()
```

## Convolutional Neural Networks (CNNs)

CNNs are specialized for processing grid-like data such as images.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        # So the flattened size is 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Convolutional block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Convolutional block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Convolutional block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Example usage (without actual training data)
# cnn_model = CNN(num_classes=10)
# sample_input = torch.randn(4, 3, 32, 32)  # 4 images, 3 channels, 32x32
# output = cnn_model(sample_input)
# print(f"Output shape: {output.shape}")
```

## Recurrent Neural Networks (RNNs)

RNNs are designed for sequential data where the order matters.

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Example usage
# input_size = 10  # Size of each input in the sequence
# hidden_size = 20  # Size of hidden state
# num_layers = 2  # Number of RNN layers
# num_classes = 5  # Number of output classes
# sequence_length = 15  # Length of input sequence
# batch_size = 8  # Number of samples in a batch

# rnn_model = RNNModel(input_size, hidden_size, num_layers, num_classes)
# lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# sample_input = torch.randn(batch_size, sequence_length, input_size)
# rnn_output = rnn_model(sample_input)
# lstm_output = lstm_model(sample_input)

# print(f"RNN Output shape: {rnn_output.shape}")
# print(f"LSTM Output shape: {lstm_output.shape}")
```

## Deep Learning for NLP: Transformers

Transformers have revolutionized NLP with self-attention mechanisms.

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "Embed size not divisible by num heads"
        
        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)  # (N, heads, query_len, key_len)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )  # (N, query_len, embed_size)
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # Add skip connection and normalize
        x = self.dropout(self.norm1(attention + query))
        
        # Feed forward
        forward = self.feed_forward(x)
        
        # Add skip connection and normalize
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

# This is a simplified implementation of the transformer architecture
# A complete implementation would include additional components like decoder
```

## Training Techniques and Best Practices

### 1. Batch Normalization
```python
import torch
import torch.nn as nn

class NetworkWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NetworkWithBatchNorm, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 2. Dropout for Regularization
```python
class NetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(NetworkWithDropout, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            
            # Add dropout after each hidden layer except the last one
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 3. Learning Rate Scheduling
```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def create_scheduler_example():
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Step scheduler: reduce LR by factor of 0.1 every 30 epochs
    step_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Plateau scheduler: reduce LR when validation loss stops improving
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    return optimizer, step_scheduler, plateau_scheduler
```

## Transfer Learning

Using pre-trained models and fine-tuning them for specific tasks.

```python
import torchvision.models as models
import torch.nn as nn

def create_transfer_learning_model(num_classes, pretrained=True):
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=pretrained)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer to adapt to our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def create_fine_tuning_model(num_classes, pretrained=True):
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=pretrained)
    
    # Keep some layers frozen, fine-tune others
    for param in model.parameters():
        param.requires_grad = False
    
    # Fine-tune the last few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Adjust the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
```

## Deep Learning Applications

### 1. Computer Vision
Deep learning models like CNNs have achieved state-of-the-art results in image classification, object detection, and segmentation.

```python
def simple_image_classifier():
    """
    A simple example of how CNNs are used in computer vision.
    This would require actual image data and preprocessing.
    """
    import torch
    import torch.nn as nn
    
    # Define a simple CNN for image classification
    class ImageClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super(ImageClassifier, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Assuming input is 32x32
            self.fc2 = nn.Linear(512, num_classes)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            
            x = x.view(-1, 64 * 8 * 8)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    return ImageClassifier()

# img_classifier = simple_image_classifier()
```

### 2. Natural Language Processing
Transformers and RNNs are widely used in NLP for tasks like translation, summarization, and question answering.

### 3. Generative Models
Deep learning is used to create generative models like GANs and VAEs.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

## Challenges in Deep Learning

1. **Data Requirements**: Deep learning models often require large amounts of data
2. **Computational Resources**: Training deep models requires significant computational power
3. **Overfitting**: Models can memorize training data instead of learning generalizable patterns
4. **Interpretability**: Deep models are often considered "black boxes"
5. **Hyperparameter Tuning**: Finding optimal hyperparameters can be time-consuming
6. **Vanishing/Exploding Gradients**: Problems with training very deep networks

## Advanced Topics

### 1. Attention Mechanisms
Attention allows models to focus on relevant parts of the input when making predictions.

### 2. Autoencoders
Neural networks used for unsupervised learning of efficient codings.

### 3. Normalizing Flows
Advanced generative models for complex probability distributions.

Deep learning continues to evolve rapidly, with new architectures and techniques being developed regularly. The field has achieved remarkable success in solving complex problems across various domains and continues to be an active area of research and application.
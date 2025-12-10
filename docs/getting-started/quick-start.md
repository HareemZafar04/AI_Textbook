---
sidebar_label: Quick Start
---

# Quick Start

Get started with AI by working through this hands-on introduction that covers the basics and demonstrates core concepts.

## Your First AI Program

Let's start with a simple example that demonstrates the core principles of machine learning:

```python
# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])  # Input features
y = np.array([2, 4, 6, 8, 10])          # Target values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
prediction = model.predict([[6]])
print(f"Prediction for input 6: {prediction[0]}")  # Should be close to 12
```

## Understanding the Components

1. **Data**: The foundation of all AI systems
2. **Model**: The algorithm that learns patterns from data
3. **Training**: The process of teaching the model from examples
4. **Prediction**: Using the trained model to make new predictions

## Key AI Concepts

### 1. Supervised Learning
Learning from labeled examples to make predictions on new data.

### 2. Unsupervised Learning
Discovering patterns in data without labeled examples.

### 3. Reinforcement Learning
Learning through interaction with an environment to maximize rewards.

## Simple Neural Network Example

Here's a basic neural network using TensorFlow:

```python
import tensorflow as tf

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
```

## Running Your First Model

Try running the examples above in your Python environment. Experiment with different values and see how the model behaves.

## Next Steps

- Explore the [Foundations of AI](../foundations/history.md) to learn about the history and core concepts
- Dive deeper into [Machine Learning](../ml/introduction.md) for more detailed algorithms
- Check out our [blog](/blog) for the latest AI developments
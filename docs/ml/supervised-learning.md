---
sidebar_label: Supervised Learning
---

# Supervised Learning

Supervised learning is a type of machine learning where the model is trained on a labeled dataset, meaning the input data is paired with the correct output. The goal is to learn a mapping from inputs to outputs that can be used to predict the output for new, unseen inputs.

## Overview of Supervised Learning

In supervised learning, the algorithm learns from a training dataset that contains both input features and the corresponding target values. The model adjusts its parameters to minimize the difference between its predictions and the true values.

### Key Components:
- **Input variables (X)**: Features or independent variables
- **Output variable (y)**: Target or dependent variable
- **Training dataset**: Examples with inputs and correct outputs
- **Model**: Function that maps inputs to outputs
- **Loss function**: Measures the difference between predictions and true values

## Types of Supervised Learning Problems

### 1. Classification
Predicting discrete class labels.

#### Binary Classification
Predicting one of two possible classes.
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0, 
                           n_informative=4, n_clusters_per_class=1, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))
```

#### Multiclass Classification
Predicting one of multiple possible classes.
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### 2. Regression
Predicting continuous numerical values.
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate a regression dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.6)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
```

## Common Supervised Learning Algorithms

### 1. Linear Regression
Used for regression tasks, models the relationship between variables using a straight line.
```python
from sklearn.linear_model import LinearRegression
import numpy as np

def linear_regression_example():
    # Simple linear regression: y = mx + b
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.5  # y = 2x + 1 + noise
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Slope: {model.coef_[0]:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"Equation: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")

linear_regression_example()
```

### 2. Logistic Regression
Despite its name, used for binary classification problems.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def logistic_regression_example():
    # Generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Get class probabilities
    probabilities = model.predict_proba(X[:5])
    predictions = model.predict(X[:5])
    
    print("First 5 predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Predicted={pred}, Probability=[{prob[0]:.3f}, {prob[1]:.3f}]")

logistic_regression_example()
```

### 3. Decision Trees
Tree-like model of decisions and their possible consequences.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.tree import export_text

def decision_tree_example():
    # Generate classification data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Print the decision tree
    tree_rules = export_text(model, feature_names=[f'feature_{i}' for i in range(X.shape[1])])
    print(tree_rules)
    
    # Feature importance
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i}: {importance:.3f}")

decision_tree_example()
```

### 4. Random Forest
Ensemble method using multiple decision trees.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def random_forest_example():
    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(X[:5])
    probabilities = model.predict_proba(X[:5])
    
    print("Random Forest Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Predicted={pred}, Probability=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Feature importance
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i}: {importance:.3f}")

random_forest_example()
```

### 5. Support Vector Machines (SVM)
Effective for both classification and regression, but especially powerful for classification.
```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification

def svm_example():
    # Generate classification data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(X[:5])
    print("SVM Predictions on first 5 samples:", predictions)

svm_example()
```

## Key Concepts in Supervised Learning

### 1. Training, Validation, and Test Sets
```python
from sklearn.model_selection import train_test_split

# Split data into train, validation, and test sets
def split_data(X, y, test_size=0.2, val_size=0.2):
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Second split: separate train and validation sets
    train_size = 1 - val_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1-train_size, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Example usage with iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### 2. Cross-Validation
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def cross_validation_example():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

cross_validation_example()
```

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def hyperparameter_tuning_example():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Define the model
    model = RandomForestClassifier(random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Use the best model
    best_model = grid_search.best_estimator_
    return best_model

best_model = hyperparameter_tuning_example()
```

## Model Evaluation Metrics

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def classification_metrics_example():
    # Generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

classification_metrics_example()
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def regression_metrics_example():
    # Generate regression data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Root MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

regression_metrics_example()
```

## Common Challenges in Supervised Learning

### 1. Overfitting and Underfitting
- **Overfitting**: Model learns training data too well, including noise
- **Underfitting**: Model fails to capture the underlying pattern

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def overfitting_underfitting_example():
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = 1.5 * X.ravel() + np.sin(1.5 * np.pi * X.ravel()) + np.random.normal(scale=0.1, size=X.shape[0])
    
    # Create polynomial models of different degrees
    degrees = [1, 4, 15]
    
    plt.figure(figsize=(16, 5))
    for i, degree in enumerate(degrees):
        # Create pipeline with polynomial features and linear regression
        poly_reg = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Fit the model
        poly_reg.fit(X, y)
        
        # Make predictions
        X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        y_plot = poly_reg.predict(X_plot)
        
        # Plot
        plt.subplot(1, 3, i+1)
        plt.scatter(X, y, alpha=0.5)
        plt.plot(X_plot, y_plot, color='red')
        plt.title(f'Polynomial Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()

overfitting_underfitting_example()
```

### 2. Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification

def preprocessing_example():
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
    
    # StandardScaler: removes mean and scales to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Original data - Mean: {X.mean(axis=0)}, Std: {X.std(axis=0)}")
    print(f"Scaled data - Mean: {X_scaled.mean(axis=0)}, Std: {X_scaled.std(axis=0)}")
    
    # MinMaxScaler: scales features to a range (typically 0-1)
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    
    print(f"MinMax scaled data - Min: {X_minmax.min(axis=0)}, Max: {X_minmax.max(axis=0)}")

preprocessing_example()
```

Supervised learning algorithms form the basis of many practical machine learning applications, from email spam detection to medical diagnosis. Choosing the right algorithm, properly preprocessing data, and evaluating model performance are crucial for success in supervised learning tasks.
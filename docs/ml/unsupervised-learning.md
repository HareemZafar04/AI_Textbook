---
sidebar_label: Unsupervised Learning
---

# Unsupervised Learning

Unsupervised learning is a type of machine learning where the model is trained on unlabeled data, meaning the input data does not have corresponding output labels. The goal is to discover hidden patterns, structures, or relationships in the data without explicit guidance.

## Overview of Unsupervised Learning

Unlike supervised learning, unsupervised learning algorithms work with input data without target variables. The model must find patterns and structure in the data on its own. This makes unsupervised learning particularly useful for exploration and understanding complex datasets.

### Key characteristics:
- No labeled output data
- Discovery of hidden patterns
- Often used for data preprocessing and exploration
- Evaluation can be challenging as there are no "correct" answers

## Types of Unsupervised Learning Problems

### 1. Clustering
Grouping similar instances together based on their features.

#### K-Means Clustering
One of the most popular clustering algorithms that partitions data into K distinct clusters.
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Plot the clustering results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(f"Cluster centers:\n{centers}")
```

#### Hierarchical Clustering
Creates a tree of clusters by iteratively merging or splitting clusters.
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=50, centers=4, cluster_std=1.0, random_state=42)

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Create dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Perform Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
y_pred = hierarchical.fit_predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('Hierarchical Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

#### DBSCAN (Density-Based Spatial Clustering)
Groups together points that are closely packed together.
```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Create data with irregular clusters
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
# Add some noise
rng = np.random.RandomState(42)
X = np.vstack([X, rng.uniform(low=-6, high=6, size=(20, 2))])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Count points in each cluster and noise points
unique_labels, counts = np.unique(y_pred, return_counts=True)
print(f"Cluster labels: {unique_labels}")
print(f"Number of points per cluster: {dict(zip(unique_labels, counts))}")
```

### 2. Association Rule Learning
Discovering interesting relationships between variables in large databases.

#### Apriori Algorithm
Commonly used for market basket analysis.
```python
# Using mlxtend library for apriori algorithm
# Note: This requires installing mlxtend: pip install mlxtend
try:
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    import pandas as pd
    
    # Create sample transaction data
    transactions = [
        ['Milk', 'Eggs', 'Bread', 'Cheese'],
        ['Eggs', 'Bread'],
        ['Milk', 'Bread'],
        ['Eggs', 'Bread', 'Butter'],
        ['Milk', 'Eggs', 'Bread', 'Butter'],
        ['Milk', 'Eggs', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Eggs'],
        ['Bread', 'Butter', 'Cheese'],
        ['Milk', 'Bread', 'Cheese']
    ]
    
    # Convert to transaction matrix
    unique_items = set(item for transaction in transactions for item in transaction)
    df = pd.DataFrame(
        [[item in transaction for item in unique_items] for transaction in transactions],
        columns=unique_items
    )
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    
    print("Frequent Itemsets:")
    print(frequent_itemsets)
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
except ImportError:
    print("mlxtend library not available. Install with: pip install mlxtend")
```

### 3. Dimensionality Reduction
Reducing the number of features while preserving important information.

#### Principal Component Analysis (PCA)
Linear dimensionality reduction technique.
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Iris Dataset')
plt.colorbar(scatter)
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
```

#### t-SNE (t-distributed Stochastic Neighbor Embedding)
Non-linear dimensionality reduction technique especially good for visualizing high-dimensional data.
```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.title('t-SNE Visualization of Digits Dataset')
plt.colorbar(scatter)
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()
```

#### UMAP (Uniform Manifold Approximation and Projection)
Modern technique for dimensionality reduction and visualization.
```python
# Note: This requires installing umap-learn: pip install umap-learn
try:
    import umap
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt

    # Load the digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    X_umap = reducer.fit_transform(X)

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.title('UMAP Visualization of Digits Dataset')
    plt.colorbar(scatter)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()
except ImportError:
    print("umap-learn library not available. Install with: pip install umap-learn")
```

## Feature Scaling and Preprocessing

Many unsupervised algorithms are sensitive to the scale of the features.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

# Generate sample data with different scales
np.random.seed(42)
X = np.column_stack([
    np.random.normal(0, 10, 100),  # Feature with large variance
    np.random.normal(0, 1, 100)   # Feature with small variance
])

print(f"Before scaling - Feature 1: mean={X[:, 0].mean():.2f}, std={X[:, 0].std():.2f}")
print(f"Before scaling - Feature 2: mean={X[:, 1].mean():.2f}, std={X[:, 1].std():.2f}")

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter scaling - Feature 1: mean={X_scaled[:, 0].mean():.2f}, std={X_scaled[:, 0].std():.2f}")
print(f"After scaling - Feature 2: mean={X_scaled[:, 1].mean():.2f}, std={X_scaled[:, 1].std():.2f}")

# Compare clustering results with and without scaling
kmeans_before = KMeans(n_clusters=3, random_state=42)
labels_before = kmeans_before.fit_predict(X)

kmeans_after = KMeans(n_clusters=3, random_state=42)
labels_after = kmeans_after.fit_predict(X_scaled)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels_before, cmap='viridis')
ax1.set_title('K-Means without Scaling')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_after, cmap='viridis')
ax2.set_title('K-Means with Scaling')
ax2.set_xlabel('Feature 1 (scaled)')
ax2.set_ylabel('Feature 2 (scaled)')

plt.tight_layout()
plt.show()
```

## Anomaly Detection

Identifying unusual data points that do not conform to expected patterns.

### Isolation Forest
Effective for anomaly detection in high-dimensional datasets.
```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Create normal data
X_normal, _ = make_blobs(n_samples=300, centers=1, n_features=2, 
                         random_state=42, cluster_std=1.0)

# Add some anomalies
X_anomaly = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_anomaly])

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X)  # -1 for anomalies, 1 for normal

# Plot results
plt.figure(figsize=(10, 6))
normal_points = anomaly_labels == 1
anomaly_points = anomaly_labels == -1

plt.scatter(X[normal_points, 0], X[normal_points, 1], 
           c='blue', label='Normal', alpha=0.7)
plt.scatter(X[anomaly_points, 0], X[anomaly_points, 1], 
           c='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection with Isolation Forest')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Number of anomalies detected: {sum(anomaly_points)}")
print(f"Expected anomalies: 20")
```

### Local Outlier Factor (LOF)
Identifies anomalies based on local density.
```python
from sklearn.neighbors import LocalOutlierFactor

# Using the same data as above
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(X)

# Plot results
plt.figure(figsize=(10, 6))
normal_points = lof_labels == 1
anomaly_points = lof_labels == -1

plt.scatter(X[normal_points, 0], X[normal_points, 1], 
           c='blue', label='Normal', alpha=0.7)
plt.scatter(X[anomaly_points, 0], X[anomaly_points, 1], 
           c='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection with Local Outlier Factor')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

## Model Evaluation in Unsupervised Learning

Since there are no true labels, evaluation is more subjective:

### 1. Silhouette Analysis
Measures how well-separated clusters are.
```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# Try different numbers of clusters
n_clusters_range = range(2, 8)
silhouette_scores = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For {n_clusters} clusters, silhouette score is: {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
```

### 2. Elbow Method
For determining the optimal number of clusters.
```python
from sklearn.metrics import pairwise_distances

def calculate_wcss(X, max_k):
    """Calculate Within-Cluster Sum of Squares for different k values"""
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

# Calculate WCSS for different k values
wcss = calculate_wcss(X, max_k=10)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()
```

## Unsupervised Learning Applications

### 1. Customer Segmentation
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create a sample customer dataset
np.random.seed(42)
n_customers = 300

data = {
    'Annual_Spending': np.random.normal(5000, 1500, n_customers),
    'Frequency_of_Purchase': np.random.poisson(10, n_customers),
    'Avg_Order_Value': np.random.normal(75, 25, n_customers)
}

customer_df = pd.DataFrame(data)

# Preprocess the data
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_segments = kmeans.fit_predict(customer_scaled)

# Add segments to the dataframe
customer_df['Segment'] = customer_segments

# Visualize customer segments
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(customer_df['Annual_Spending'], customer_df['Frequency_of_Purchase'], 
           c=customer_segments, cmap='viridis', alpha=0.7)
plt.title('Customer Segments: Spending vs Frequency')
plt.xlabel('Annual Spending')
plt.ylabel('Frequency of Purchase')

plt.subplot(1, 2, 2)
plt.scatter(customer_df['Avg_Order_Value'], customer_df['Annual_Spending'], 
           c=customer_segments, cmap='viridis', alpha=0.7)
plt.title('Customer Segments: Order Value vs Spending')
plt.xlabel('Avg Order Value')
plt.ylabel('Annual Spending')

plt.tight_layout()
plt.show()

# Analyze each segment
segment_analysis = customer_df.groupby('Segment').agg({
    'Annual_Spending': ['mean', 'std'],
    'Frequency_of_Purchase': ['mean', 'std'],
    'Avg_Order_Value': ['mean', 'std']
}).round(2)

print("Customer Segment Analysis:")
print(segment_analysis)
```

### 2. Document Clustering
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample documents
documents = [
    "The stock market showed significant growth today",
    "New smartphone released with advanced camera features",
    "Scientists discovered new species in the rainforest",
    "Economic indicators suggest upcoming recession",
    "Apple announces new iPhone model",
    "Conservation efforts protect endangered animals",
    "Stock prices fluctuate with market trends",
    "Technology companies report quarterly earnings",
    "Biologists study ecosystem diversity",
    "Financial markets react to policy changes"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(documents)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
document_clusters = kmeans.fit_predict(X)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Visualize document clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=document_clusters, cmap='viridis', alpha=0.7)

# Annotate points with document indices
for i, doc in enumerate(documents):
    plt.annotate(str(i), (X_pca[i, 0], X_pca[i, 1]), fontsize=8, ha='center')

plt.title('Document Clustering with TF-IDF')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter)
plt.show()

# Show which documents belong to each cluster
for cluster_id in range(3):
    cluster_docs = [documents[i] for i, label in enumerate(document_clusters) if label == cluster_id]
    print(f"\nCluster {cluster_id}:")
    for doc in cluster_docs:
        print(f"  - {doc}")
```

## Challenges in Unsupervised Learning

1. **No Ground Truth**: Difficult to evaluate model performance objectively
2. **Parameter Selection**: Often requires domain knowledge to choose optimal parameters
3. **Interpretation**: Results may not always be intuitive or actionable
4. **Scalability**: Some unsupervised methods don't scale well with large datasets

Unsupervised learning provides powerful tools for exploring and understanding data, identifying patterns, and preprocessing data for other machine learning tasks. The choice of algorithm depends on the specific problem and data characteristics.
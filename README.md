# Unsupervised Learning: Clustering Penguin Data  

## Table of Contents  
- [Project Overview](#project-overview)  
- [Data Source](#data-source)  
- [Tools & Libraries](#tools--libraries)  
- [Data Preparation](#data-preparation)  
- [Clustering Analysis](#clustering-analysis)  
- [Findings](#findings)  
- [Limitations](#limitations)  

## Project Overview  
---  
This project applies **K-Means clustering** to analyze penguin data and identify hidden groupings. Given the lack of labeled species data, the goal is to cluster penguins based on physical attributes and derive meaningful insights.  

## Data Source  
- Dataset: `penguins.csv`  
- Data collected by **Dr. Kristen Gorman** and the **Palmer Station, Antarctica LTER**  
- Features:  
  - **culmen_length_mm** (Beak length)  
  - **culmen_depth_mm** (Beak depth)  
  - **flipper_length_mm** (Flipper size)  
  - **body_mass_g** (Body mass in grams)  
  - **sex** (Penguin gender)  

## Tools & Libraries  
- **Python**: Data analysis & visualization  
- **Libraries**:  
  - `pandas` for data handling  
  - `matplotlib` for visualization  
  - `scikit-learn` for clustering  
  - `StandardScaler` for data normalization  

## Data Preparation  
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()

# One-hot encode categorical data
penguin_dummies = pd.get_dummies(penguins_df)

# Standardize numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(penguin_dummies)
```
## Clustering Analysis
### Finding the optimal number of clusters
```python
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertias.append(model.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(ks, inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()
```
![elbow_pengiuns](https://github.com/user-attachments/assets/e8d3021c-82c5-496b-86ec-6891bc377780)

## Applying KMeans Clustering
```python
# Applying K-Means with optimal clusters
kmeans = KMeans(n_clusters=4, random_state=42).fit(scaled_data)
penguin_dummies["Cluster"] = kmeans.labels_

# Visualizing clusters
plt.figure(figsize=(8, 5))
plt.scatter(penguin_dummies["Cluster"], penguin_dummies["culmen_length_mm"], c=kmeans.labels_, cmap="viridis")
plt.xlabel("Cluster")
plt.ylabel("Culmen Length (mm)")
plt.title("Penguin Clusters Based on Culmen Length")
plt.colorbar(label="Cluster Label")
plt.grid(True)
plt.show()
```
![elbow_pengiuns](https://github.com/user-attachments/assets/f6179f8a-1c98-4e25-ba57-39196347c23d)

## Findings
- 4 clusters were identified, corresponding to distinct physical traits.
- Beak and flipper dimensions played a key role in clustering.
- Some overlap exists, possibly due to natural variation or missing species labels.
## Limitations
- No species labels were provided, making validation difficult.
- Clustering is purely data-driven and may not map directly to real species.
- Dataset size and missing values may affect accuracy.


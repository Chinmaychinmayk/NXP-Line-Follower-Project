import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier

# Sample LiDAR data (replace with your actual data)
lidar_data = np.array([5, 5, 5, 6, 7, 7, 8, 9, 10, 10, 10, 9, 8, 7, 7, 6, 5, 5, 5])

# Preprocess Data with Median and Gaussian Filters
smoothed_data = median_filter(lidar_data, size=5)
smoothed_data = gaussian_filter(smoothed_data, sigma=1.0)

# Adaptive thresholds
mean_distance = np.mean(smoothed_data)
std_distance = np.std(smoothed_data)
discontinuity_threshold = 0.1 * mean_distance
slope_threshold = 0.05 * std_distance

# Detect Obstacles
edges = np.where(np.abs(np.diff(smoothed_data)) > discontinuity_threshold)[0]

# Cluster Points using DBSCAN
db = DBSCAN(eps=0.3, min_samples=2).fit(smoothed_data.reshape(-1, 1))
labels = db.labels_

# Extract features for classification
def extract_features(data, labels):
    features = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = data[labels == label]
        features.append([
            np.mean(cluster),
            np.std(cluster),
            len(cluster)
        ])
    return np.array(features)

features = extract_features(smoothed_data, labels)

# Train a Random Forest Classifier (Dummy training data used here)
# In practice, use real labeled data
train_features = np.array([[5, 0, 5], [9, 1, 4], [7, 0.5, 4]])  # Example training data
train_labels = np.array([0, 1, 0])  # 0 for non-obstacle, 1 for obstacle
model = RandomForestClassifier()
model.fit(train_features, train_labels)

# Predict obstacles and ramps
predictions = model.predict(features)

# Detect Ramps
ramp_points = []
for i in range(len(smoothed_data) - 1):
    slope = (smoothed_data[i + 1] - smoothed_data[i])
    if abs(slope) > slope_threshold:
        ramp_points.append(i)

# Combine Consecutive Ramp Points
ramps = []
current_ramp = []
for i in ramp_points:
    if current_ramp and i != current_ramp[-1] + 1:
        ramps.append(current_ramp)
        current_ramp = []
    current_ramp.append(i)
if current_ramp:
    ramps.append(current_ramp)

# Filter Out Noise from Ramps based on height difference
filtered_ramps = [ramp for ramp in ramps if np.abs(smoothed_data[ramp[-1]] - smoothed_data[ramp[0]]) > mean_distance * 0.1]

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(lidar_data, label='Original Data')
plt.plot(smoothed_data, label='Smoothed Data')
plt.scatter(edges, smoothed_data[edges], color='red', label='Edges')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for label in set(labels):
    if label == -1:
        continue
    plt.scatter(np.where(labels == label), smoothed_data[labels == label], label=f'Cluster {label}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for ramp in filtered_ramps:
    plt.plot(range(ramp[0], ramp[-1] + 1), smoothed_data[ramp[0]:ramp[-1] + 1], label='Ramp')
plt.legend()
plt.show()

import cv2
import os
import numpy as np
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define image preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize to fixed size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return gray

# Define feature extraction using HOG
def extract_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# Load dataset
def load_dataset(folder1, folder2):
    images = []
    labels = []
    
    for img_path in glob.glob(os.path.join(folder1, "*.jpg")):
        img = preprocess_image(img_path)
        features = extract_features(img)
        images.append(features)
        labels.append(0)  # Class 0
    
    for img_path in glob.glob(os.path.join(folder2, "*.jpg")):
        img = preprocess_image(img_path)
        features = extract_features(img)
        images.append(features)
        labels.append(1)  # Class 1
    
    return np.array(images), np.array(labels)

# Set dataset paths
folder1 = "path_to_class1_images"
folder2 = "path_to_class2_images"

# Load images and extract features
X, y = load_dataset(folder1, folder2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

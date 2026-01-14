import numpy as np
from matplotlib.image import imread, imsave
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# Load all data
image_dir = "images"
mask_dir = "masks"

images = []
masks = []
for img_fn, mask_fn in zip(glob('images/*.png')[:100], glob('masks/*.png')[:100]):
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
    
    images.append(img)
    masks.append(mask)

images = np.array(images)  # Shape: (100, 240, 240, 3)
masks = np.array(masks)    # Shape: (100, 240, 240)

masks = np.round(masks/128).astype(int)

# Convert to binary: 0=background, 1=disease
binary_masks = np.where(masks == 2, 1, masks)
binary_images = images

# Create disease levels for stratification
disease_levels = []
for mask in binary_masks:
    leaf_area = np.sum(mask != 0)
    if leaf_area > 0:
        disease_level = np.sum(mask == 1) / leaf_area
    else:
        disease_level = 0
    disease_levels.append(disease_level)

# Bin disease levels for stratification
disease_bins = np.digitize(disease_levels, bins=[0.01, 0.25, 0.5, 0.75])

# First split: 70% train, 30% temp
train_idx, temp_idx = train_test_split(
    np.arange(len(binary_images)), 
    test_size=0.3, 
    stratify=disease_bins,
    random_state=42
)

X_train = binary_images[train_idx]
y_train = binary_masks[train_idx]
X_temp = binary_images[temp_idx]
y_temp = binary_masks[temp_idx]

# Second split: 50/50 for val and test from temp
# Get disease levels for temp set only
temp_disease_levels = []
for mask in y_temp:
    leaf_area = np.sum(mask != 0)
    if leaf_area > 0:
        disease_level = np.sum(mask == 1) / leaf_area
    else:
        disease_level = 0
    temp_disease_levels.append(disease_level)

temp_disease_bins = np.digitize(temp_disease_levels, bins=[0.01, 0.25, 0.5, 0.75])

val_idx, test_idx = train_test_split(
    np.arange(len(X_temp)),
    test_size=0.5,
    stratify=temp_disease_bins,
    random_state=42
)

X_val = X_temp[val_idx]
y_val = y_temp[val_idx]
X_test = X_temp[test_idx]
y_test = y_temp[test_idx]

# Simple normalization to [0, 1]
X_train_norm = X_train / 255.0
X_val_norm = X_val / 255.0
X_test_norm = X_test / 255.0

# Skip HSV conversion entirely
def extract_simple_features(image):
    """Extract only RGB features"""
    h, w = image.shape[:2]
    features = image.reshape(-1, 3)  # Just RGB
    return features

# Use a small subset for speed
train_features = []
train_labels = []

# Sample 2000 random pixels from first 10 images
for i in range(min(10, len(X_train))):
    img = X_train[i]
    mask = y_train[i]
    
    # Get random pixels
    h, w = img.shape[:2]
    n_samples = 200
    rows = np.random.randint(0, h, n_samples)
    cols = np.random.randint(0, w, n_samples)
    
    for r, c in zip(rows, cols):
        # Only use if not background
        if np.mean(img[r, c]) > 20:
            train_features.append(img[r, c])
            train_labels.append(mask[r, c])

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# Train
clf_simple = LogisticRegression(max_iter=500, random_state=42)
clf_simple.fit(train_features, train_labels)

def extract_patch_features(image, patch_size=16):
    """Extract patches and their features"""
    h, w = image.shape[:2]
    features_list = []
    positions = []
    
    # Slide window
    for i in range(0, h - patch_size, patch_size//2):
        for j in range(0, w - patch_size, patch_size//2):
            patch = image[i:i+patch_size, j:j+patch_size]
            
            # Only process if not all background
            if np.mean(patch) > 0.05:
                # Simple features
                mean_rgb = patch.mean(axis=(0,1))
                std_rgb = patch.std(axis=(0,1))
                
                # Color ratios
                if mean_rgb[1] > 0:
                    r_g_ratio = mean_rgb[0] / mean_rgb[1]
                else:
                    r_g_ratio = 0
                
                features = np.concatenate([mean_rgb, std_rgb, [r_g_ratio]])
                features_list.append(features)
                positions.append((i, j))
    
    return np.array(features_list), positions

# Prepare training data
train_patches = []
train_patch_labels = []

for img, mask in zip(X_train_norm[:15], y_train[:15]):
    features, positions = extract_patch_features(img, patch_size=16)
    
    # Get label for each patch (majority vote in center region)
    for (i, j), feat in zip(positions, features):
        center_i, center_j = i + 8, j + 8
        if center_i < mask.shape[0] and center_j < mask.shape[1]:
            # Label patch as diseased if any disease in it
            patch_mask = mask[i:i+16, j:j+16]
            label = 1 if np.sum(patch_mask) > 10 else 0  # Threshold
            train_patches.append(feat)
            train_patch_labels.append(label)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(train_patches, train_patch_labels)

# Build model
def build_simple_cnn(input_shape=(240, 240, 3)):
    model = models.Sequential([
        # Downsample for speed
        layers.Resizing(128, 128),
        
        # Conv block 1
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        
        # Conv block 2
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        
        # Upsample
        layers.UpSampling2D(2),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        
        layers.UpSampling2D(2),
        layers.Conv2D(1, 3, padding='same', activation='sigmoid'),
        
        # Resize back to original
        layers.Resizing(240, 240)
    ])
    
    return model

# Build and compile
model = build_simple_cnn()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

# Train (very few epochs for demo)
history = model.fit(X_train_norm, y_train,
                    validation_data=(X_val_norm, y_val),
                    epochs=5,
                    batch_size=4,  # Small batch due to memory
                    verbose=1)

# Save models
with open('leaf_disease_app/models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(clf_simple, f)

with open('leaf_disease_app/models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)

model.save('leaf_disease_app/models/cnn_model.h5')
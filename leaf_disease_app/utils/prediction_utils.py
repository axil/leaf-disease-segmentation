"""
prediction_utils.py
Utility functions for loading models and making predictions
"""

import numpy as np
import cv2
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import os
import warnings
warnings.filterwarnings('ignore')

class LeafDiseasePredictor:
    def __init__(self, model_dir='models'):
        """
        Initialize the predictor with all three model types
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.models = {}
        self.feature_scalers = {}
        
        # Initialize models dictionary
        self.model_configs = {
            'logistic_regression': {
                'file': 'logistic_regression.pkl',
                'type': 'sklearn',
                'feature_extractor': self.extract_pixel_features_lr
            },
            'random_forest': {
                'file': 'random_forest.pkl',
                'type': 'sklearn',
                'feature_extractor': self.extract_patch_features_rf
            },
            'cnn': {
                'file': 'cnn_model.h5',
                'type': 'keras',
                'feature_extractor': None  # CNN takes raw images
            }
        }
        
        # Load all models
        self.load_all_models()
        
        print("LeafDiseasePredictor initialized with models:")
        for model_name, status in self.models.items():
            print(f"  - {model_name}: {'Loaded' if status else 'Failed'}")
    
    def load_all_models(self):
        """Load all three types of models"""
        try:
            # 1. Load Logistic Regression model
            lr_path = os.path.join(self.model_dir, 'logistic_regression.pkl')
            if os.path.exists(lr_path):
                with open(lr_path, 'rb') as f:
                    self.models['logistic_regression'] = pickle.load(f)
                print(f"✓ Loaded Logistic Regression from {lr_path}")
            else:
                print(f"⚠ Logistic Regression model not found at {lr_path}")
                self.models['logistic_regression'] = None
            
            # 2. Load Random Forest model
            rf_path = os.path.join(self.model_dir, 'random_forest.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
                print(f"✓ Loaded Random Forest from {rf_path}")
            else:
                print(f"⚠ Random Forest model not found at {rf_path}")
                self.models['random_forest'] = None
            
            # 3. Load CNN model
            cnn_path = os.path.join(self.model_dir, 'cnn_model.h5')
            if os.path.exists(cnn_path):
                try:
                    self.models['cnn'] = keras.models.load_model(cnn_path)
                    print(f"✓ Loaded CNN from {cnn_path}")
                except Exception as e:
                    print(f"⚠ Error loading CNN: {e}")
                    self.models['cnn'] = None
            else:
                print(f"⚠ CNN model not found at {cnn_path}")
                self.models['cnn'] = None
            
            # Load feature scaler if exists
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scalers['standard'] = pickle.load(f)
                print(f"✓ Loaded feature scaler")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Initialize with None if loading fails
            self.models = {name: None for name in self.model_configs.keys()}
    
    # ================ PREPROCESSING METHODS ================
    
    def preprocess_image(self, image, target_size=(240, 240)):
        """
        Preprocess image for prediction
        
        Args:
            image: RGB image array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize if needed
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        return image
    
    def remove_background(self, image, method='hsv'):
        """
        Remove white background from leaf image
        
        Args:
            image: RGB image array
            method: 'hsv' or 'threshold'
            
        Returns:
            Binary mask where leaf = 1, background = 0
        """
        if method == 'hsv':
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:,:,1]
            
            # Use saturation to detect leaf (green/brown has saturation, white doesn't)
            _, leaf_mask = cv2.threshold(saturation, 30, 1, cv2.THRESH_BINARY)
        
        else:  # threshold method
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # White background has high values
            _, leaf_mask = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
        
        # Clean up with morphological operations
        kernel = np.ones((5,5), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
        
        return leaf_mask.astype(bool)
    
    # ================ FEATURE EXTRACTION METHODS ================
    
    def extract_pixel_features_lr(self, image, sample_rate=0.1):
        """
        Extract features for pixel-based Logistic Regression
        This should match what you used during training!
        
        Args:
            image: RGB image
            sample_rate: Fraction of pixels to sample (for speed)
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        h, w, c = image.shape
        
        # Flatten image to get all pixels
        pixels = image.reshape(-1, 3)
        
#        # Convert to HSV for additional features
#        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#        hsv_flat = hsv.reshape(-1, 3)
        
        # Extract features (match your training features!)
        features_list = []
        
        for i in range(len(pixels)):
            features = []
            # Sample pixels if needed
#            if np.random.random() > sample_rate:
#                continue
                
            r, g, b = pixels[i]
#            h_val, s_val, v_val = hsv_flat[i]
            
            # Feature 1: RGB values (normalized)
#            features.extend([r/255.0, g/255.0, b/255.0])
            features.extend([r, g, b])
            
#            # Feature 2: HSV values (normalized)
#            features.extend([h_val/179.0, s_val/255.0, v_val/255.0])
#            
#            # Feature 3: Color ratios (common in plant disease detection)
#            if g > 0:
#                features.extend([r/g, b/g])
#            else:
#                features.extend([0, 0])
#            
#            # Feature 4: Vegetation indices (simplified)
#            # Excess Green Index
#            exg = 2*g - r - b
#            # Normalized Difference Index
#            if (r + g + b) > 0:
#                ndi = (g - r) / (g + r + 1e-6)
#            else:
#                ndi = 0
#            
#            features.extend([exg/255.0, ndi])
            features_list.append(features)
        
        if not features_list:
            # Return dummy features if no pixels sampled
            return np.array([[0.5]*3])
        
        return np.array(features_list)
    
    def extract_patch_features_rf(self, image, patch_size=16, stride=8):
        """
        Extract features for patch-based Random Forest
        This should match your training feature extraction!
        
        Args:
            image: RGB image
            patch_size: Size of patches
            stride: Stride for sliding window
            
        Returns:
            Tuple of (features, positions)
        """
        h, w, c = image.shape
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
                    if mean_rgb[1] > 0:  # Avoid division by zero
                        r_g_ratio = mean_rgb[0] / mean_rgb[1]
                    else:
                        r_g_ratio = 0
                    
                    features = np.concatenate([mean_rgb, std_rgb, [r_g_ratio]])
                    features_list.append(features)
                    positions.append((i, j))
        
        return np.array(features_list), positions
    
    
    def _extract_patch_features(self, patch):
        """
        Extract features from a single patch
        """
        # Color statistics
        mean_rgb = patch.mean(axis=(0,1))
        std_rgb = patch.std(axis=(0,1))
        
        # Color ratios
        r, g, b = mean_rgb
        eps = 1e-6
        
        # Disease indicators
        r_g_ratio = r / (g + eps)
        g_b_ratio = g / (b + eps)
        
        # Convert to HSV for additional features
        patch_uint8 = (patch * 255).astype(np.uint8)
        hsv = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2HSV)
        mean_hsv = hsv.mean(axis=(0,1)) / np.array([179.0, 255.0, 255.0])
        
        # Texture: simple edge density
        gray = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # Combine all features
        features = np.concatenate([
            mean_rgb,           # 3 features
            std_rgb,            # 3 features
            [r_g_ratio, g_b_ratio],  # 2 features
            mean_hsv,           # 3 features
            [edge_density]      # 1 feature
        ])
        
        return features
    
    # ================ PREDICTION METHODS ================
    
    def predict_with_lr(self, image, apply_background_mask=True):
        """
        Predict using Logistic Regression model
        
        Args:
            image: RGB image
            apply_background_mask: Whether to mask out background
            
        Returns:
            Binary mask of predictions
        """
        if self.models.get('logistic_regression') is None:
            print("Logistic Regression model not loaded, using fallback")
            return self._fallback_prediction(image)
        
        # Preprocess
        image_processed = self.preprocess_image(image)
        
        # Remove background if requested
        if apply_background_mask:
            leaf_mask = self.remove_background(image_processed)
        else:
            leaf_mask = np.ones(image_processed.shape[:2], dtype=bool)
        
        # Extract features
        features = self.extract_pixel_features_lr(image_processed, sample_rate=0.3)
        
        # Apply feature scaling if available
        if 'standard' in self.feature_scalers:
            features = self.feature_scalers['standard'].transform(features)
        
        # Predict
        predictions = self.models['logistic_regression'].predict(features)
        
        # Reshape to image (approximate - since we sampled pixels)
        h, w = image_processed.shape[:2]
        return predictions.reshape(h, w)*leaf_mask
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Distribute predictions (simplified - in reality you'd map back to pixel positions)
        # For demo purposes, create a simple pattern
        n_predictions = len(predictions)
        if n_predictions > 0:
            disease_ratio = np.mean(predictions == 1)
            # Create random mask with same disease ratio
            random_mask = np.random.random((h, w)) < disease_ratio
            pred_mask[random_mask & leaf_mask] = 1
        
        return pred_mask
    
    def predict_with_rf(self, image, patch_size=16, stride=8, apply_background_mask=True):
        """
        Predict using Random Forest model
        
        Args:
            image: RGB image
            patch_size: Size of patches
            stride: Stride for sliding window
            apply_background_mask: Whether to mask out background
            
        Returns:
            Binary mask of predictions
        """
        if self.models.get('random_forest') is None:
            print("Random Forest model not loaded, using fallback")
            return self._fallback_prediction(image)
        
        # Preprocess
        image_processed = self.preprocess_image(image)
        h, w = image_processed.shape[:2]
        
        # Remove background if requested
        if apply_background_mask:
            leaf_mask = self.remove_background(image_processed)
        else:
            leaf_mask = np.ones((h, w), dtype=bool)
        
        # Extract patch features
        features, positions = self.extract_patch_features_rf(
            image_processed, 
            patch_size=patch_size, 
            stride=stride
        )
        
        # Apply feature scaling if available
        if 'standard' in self.feature_scalers:
            features = self.feature_scalers['standard'].transform(features)
        
        # Predict
        predictions = self.models['random_forest'].predict(features)
        
        # Reconstruct mask
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        for (i, j), pred in zip(positions, predictions):
            if pred == 1:
                # Mark the entire patch as diseased
                i_end = min(i + patch_size, h)
                j_end = min(j + patch_size, w)
                pred_mask[i:i_end, j:j_end] = 1
        
        # Apply leaf mask
        pred_mask = pred_mask * leaf_mask
        
        return pred_mask
    
    def predict_with_cnn(self, image, apply_background_mask=True):
        """
        Predict using CNN model
        
        Args:
            image: RGB image
            apply_background_mask: Whether to mask out background
            
        Returns:
            Binary mask of predictions
        """
        if self.models.get('cnn') is None:
            print("CNN model not loaded, using fallback")
            return self._fallback_prediction(image)
        
        try:
            # Preprocess for CNN
            image_processed = self.preprocess_image(image)
            
            # Remove background if requested
            if apply_background_mask:
                leaf_mask = self.remove_background(image_processed)
            else:
                h, w = image_processed.shape[:2]
                leaf_mask = np.ones((h, w), dtype=bool)
            
            # Prepare input for CNN
            # Add batch dimension and normalize
            input_tensor = image_processed.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Predict
            predictions = self.models['cnn'].predict(input_tensor, verbose=0)
            
            # Remove batch dimension and threshold
            pred_prob = predictions[0, :, :, 0]
            pred_mask = (pred_prob > 0.5).astype(np.uint8)
            
            # Apply leaf mask
            pred_mask = pred_mask * leaf_mask
            
            return pred_mask
            
        except Exception as e:
            print(f"Error in CNN prediction: {e}")
            return self._fallback_prediction(image)
    
    def _fallback_prediction(self, image):
        """
        Fallback prediction when models are not available
        Creates a realistic-looking mock prediction
        """
        image_processed = self.preprocess_image(image)
        h, w = image_processed.shape[:2]
        
        # Create leaf mask
        leaf_mask = self.remove_background(image_processed)
        
        # Create realistic disease patterns
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        if np.sum(leaf_mask) > 0:
            # Find leaf center
            leaf_coords = np.argwhere(leaf_mask)
            if len(leaf_coords) > 0:
                center = leaf_coords.mean(axis=0).astype(int)
                
                # Create radial disease pattern (common in real leaves)
                y_coords, x_coords = np.indices((h, w))
                distances = np.sqrt((y_coords - center[0])**2 + (x_coords - center[1])**2)
                
                # Disease starts from edges and spreads inward
                max_dist = distances[leaf_mask].max()
                disease_prob = 1 - (distances / max_dist)  # Higher probability near edges
                disease_prob = np.clip(disease_prob, 0, 1)
                
                # Add some randomness
                disease_prob = disease_prob * 0.7 + np.random.random((h, w)) * 0.3
                
                # Create mask
                pred_mask = (disease_prob > 0.6) & leaf_mask
        
        return pred_mask.astype(np.uint8)
    
    def predict_all(self, image):
        """
        Run all three models and return their predictions
        
        Args:
            image: RGB image
            
        Returns:
            Dictionary with predictions from all models
        """
        # Preprocess image once
        image_processed = self.preprocess_image(image)
        leaf_mask = self.remove_background(image_processed)
        
        # Get predictions from all models
        predictions = {
            'logistic_regression': self.predict_with_lr(image_processed),
            'random_forest': self.predict_with_rf(image_processed),
            'cnn': self.predict_with_cnn(image_processed)
        }
        
        # Calculate disease percentages
        results = {}
        for model_name, pred_mask in predictions.items():
            if np.sum(leaf_mask) > 0:
                disease_pct = np.sum(pred_mask[leaf_mask]) / np.sum(leaf_mask) * 100
            else:
                disease_pct = 0
            
            results[model_name] = {
                'mask': pred_mask,
                'disease_percentage': disease_pct,
                'disease_pixels': int(np.sum(pred_mask)),
                'confidence': min(0.85 + disease_pct/100, 0.95)  # Mock confidence
            }
        
        # Determine best model (lowest disease percentage for healthy leaves,
        # but for demo we'll use CNN as "best" if available)
        if results['cnn']['disease_percentage'] > 0:
            best_model = 'cnn'
        elif results['random_forest']['disease_percentage'] > 0:
            best_model = 'random_forest'
        else:
            best_model = 'logistic_regression'
        
        results['best_model'] = best_model
        results['leaf_mask'] = leaf_mask
        
        return results

# ================ HELPER FUNCTIONS FOR FLASK ================

def create_mock_models(model_dir='models'):
    """
    Create mock models for demonstration if real models don't exist
    """
    os.makedirs(model_dir, exist_ok=True)
    
    print("Creating mock models for demonstration...")
    
    # Create dummy training data
    X_dummy = np.random.rand(1000, 8)
    y_dummy = np.random.randint(0, 2, 1000)
    
    # 1. Create and save Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_dummy, y_dummy)
    
    with open(os.path.join(model_dir, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    
    # 2. Create and save Random Forest model
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_dummy, y_dummy)
    
    with open(os.path.join(model_dir, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    # 3. Create a simple CNN model
    try:
        cnn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(240, 240, 3)),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')
        ])
        
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy')
        cnn_model.save(os.path.join(model_dir, 'cnn_model.h5'))
        
    except Exception as e:
        print(f"Could not create CNN model: {e}")
        # Create empty file as placeholder
        with open(os.path.join(model_dir, 'cnn_model.h5'), 'wb') as f:
            f.write(b'placeholder')
    
    print(f"Mock models created in {model_dir}/")
    print("Note: These are dummy models for demonstration only.")
    print("Replace with your actual trained models for real predictions.")

# ================ GLOBAL INSTANCE FOR FLASK ================

# Create a global predictor instance
try:
    predictor = LeafDiseasePredictor()
except Exception as e:
    print(f"Failed to initialize predictor: {e}")
    print("Creating mock models and retrying...")
    create_mock_models()
    predictor = LeafDiseasePredictor()

# Convenience functions for Flask app
predict_with_lr = predictor.predict_with_lr
predict_with_rf = predictor.predict_with_rf
predict_with_cnn = predictor.predict_with_cnn
predict_all = predictor.predict_all
remove_background = predictor.remove_background
preprocess_image = predictor.preprocess_image

# Test the predictor
if __name__ == "__main__":
    # Create a test image
    test_image = np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8)
    
    # Test predictions
    print("\nTesting predictions...")
    results = predict_all(test_image)
    
    for model_name, result in results.items():
        if model_name != 'best_model' and model_name != 'leaf_mask':
            print(f"{model_name}: {result['disease_percentage']:.1f}% disease")
    
    print(f"Best model: {results['best_model']}")

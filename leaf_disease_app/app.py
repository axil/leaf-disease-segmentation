
"""
app.py - Leaf Disease Segmentation Flask Application
Complete with real model predictions, visualization, and API endpoints
"""

import os
import json
import base64
import io
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Import our prediction utilities
try:
    from utils.prediction_utils import (
        predictor, 
        predict_with_lr, 
        predict_with_rf, 
        predict_with_cnn,
        predict_all,
        remove_background,
        preprocess_image
    )
    MODELS_LOADED = True
    print("✓ Prediction utilities loaded successfully")
except Exception as e:
    print(f"⚠ Could not load prediction utilities: {e}")
    print("Using mock predictions instead")
    MODELS_LOADED = False
    # Define mock functions
    def predict_with_lr(image):
        return np.random.random(image.shape[:2]) > 0.7
    
    def predict_with_rf(image):
        return np.random.random(image.shape[:2]) > 0.8
    
    def predict_with_cnn(image):
        return np.random.random(image.shape[:2]) > 0.9
    
    def predict_all(image):
        h, w = image.shape[:2]
        leaf_mask = np.ones((h, w), dtype=bool)
        
        # Create mock predictions with different patterns
        lr_mask = create_mock_prediction(image, pattern='radial')
        rf_mask = create_mock_prediction(image, pattern='patchy')
        cnn_mask = create_mock_prediction(image, pattern='edge')
        
        results = {
            'logistic_regression': {
                'mask': lr_mask,
                'disease_percentage': float(np.mean(lr_mask) * 100),
                'disease_pixels': int(np.sum(lr_mask)),
                'confidence': 0.82
            },
            'random_forest': {
                'mask': rf_mask,
                'disease_percentage': float(np.mean(rf_mask) * 100),
                'disease_pixels': int(np.sum(rf_mask)),
                'confidence': 0.85
            },
            'cnn': {
                'mask': cnn_mask,
                'disease_percentage': float(np.mean(cnn_mask) * 100),
                'disease_pixels': int(np.sum(cnn_mask)),
                'confidence': 0.88
            },
            'best_model': 'cnn',
            'leaf_mask': leaf_mask
        }
        return results
    
    def create_mock_prediction(image, pattern='radial'):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create different patterns for different models
        if pattern == 'radial':
            # Radial pattern from center
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = (dist < min(h, w) * 0.4) & (np.random.random((h, w)) > 0.5)
        
        elif pattern == 'patchy':
            # Patchy disease pattern
            for _ in range(20):
                patch_size = np.random.randint(10, 40)
                y = np.random.randint(0, h - patch_size)
                x = np.random.randint(0, w - patch_size)
                mask[y:y+patch_size, x:x+patch_size] = 1
        
        elif pattern == 'edge':
            # Disease starting from edges
            y, x = np.ogrid[:h, :w]
            edge_dist = np.minimum(np.minimum(y, h - y), np.minimum(x, w - x))
            mask = (edge_dist < min(h, w) * 0.3) & (np.random.random((h, w)) > 0.6)
        
        return mask

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'css', 'js'}
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production!

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    """Save uploaded file and return its path"""
    filename = secure_filename(file.filename)
    # Add timestamp to avoid collisions
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{name}_{timestamp}{ext}"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    return unique_filename, filepath

def read_image_file(filepath):
    """Read image file and convert to RGB numpy array"""
    # Try PIL first
    try:
        image = Image.open(filepath)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        # Fallback to OpenCV
        print(f"PIL failed, using OpenCV: {e}")
        image_np = cv2.imread(filepath)
        if image_np is None:
            raise ValueError(f"Could not read image: {filepath}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    return image_np

def create_visualization(original_image, predictions, model_names, disease_percentages, leaf_mask=None):
    """
    Create comparison visualization of all model predictions
    
    Args:
        original_image: Original RGB image
        predictions: List of prediction masks
        model_names: List of model names
        disease_percentages: List of disease percentages
        leaf_mask: Optional leaf mask for overlay
    
    Returns:
        base64 encoded image string
    """
    n_models = len(model_names)
    
    # Calculate grid size
    if n_models <= 2:
        n_cols = n_models + 1
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_models + 2) // 3 + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Original image
    ax = axes[0, 0] if n_rows > 0 else axes[0]
    ax.imshow(original_image)
    ax.set_title('Original Image', fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Leaf mask overlay if available
    if leaf_mask is not None and np.any(leaf_mask):
        ax_leaf = axes[0, 1] if n_cols > 1 else axes[1]
        overlay = original_image.copy()
        overlay[~leaf_mask] = overlay[~leaf_mask] * 0.3  # Darken background
        ax_leaf.imshow(overlay)
        ax_leaf.set_title(f'Leaf Area\n({np.mean(leaf_mask)*100:.1f}% of image)', 
                         fontsize=11, pad=10)
        ax_leaf.axis('off')
        start_idx = 2
    else:
        start_idx = 1
    
    # Model predictions
    for i, (pred_mask, model_name, disease_pct) in enumerate(zip(predictions, model_names, disease_percentages)):
        row = (i + start_idx) // n_cols
        col = (i + start_idx) % n_cols
        
        if row < n_rows and col < n_cols:
            ax = axes[row, col]
            
            # Create overlay: red for disease, original for healthy
            overlay = original_image.copy()
            if pred_mask is not None and np.any(pred_mask):
                # Make diseased areas red
                overlay[pred_mask == 1] = [255, 100, 100]
            
            ax.imshow(overlay)
            
            # Determine risk level
            if disease_pct > 30:
                risk_color = 'red'
                risk_text = 'HIGH'
            elif disease_pct > 10:
                risk_color = 'orange'
                risk_text = 'MEDIUM'
            else:
                risk_color = 'green'
                risk_text = 'LOW'
            
            title = f'{model_name}\n{disease_pct:.1f}% disease\n{risk_text} RISK'
            ax.set_title(title, fontsize=11, color=risk_color, pad=10)
            ax.axis('off')
    
    # Add legend in last subplot
    legend_row = n_rows - 1
    legend_col = n_cols - 1
    if legend_row >= 0 and legend_col >= 0:
        ax_legend = axes[legend_row, legend_col]
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Diseased Area'),
            Patch(facecolor=[0, 0.7, 0], alpha=0.5, label='Healthy Leaf'),
            Patch(facecolor=[0.3, 0.3, 0.3], alpha=0.5, label='Background')
        ]
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=10)
        ax_legend.axis('off')
        ax_legend.set_title('Legend', fontsize=11, pad=10)
    
    # Remove any empty subplots
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if row < n_rows and col < n_cols:
            if axes[row, col].has_data() == False and axes[row, col].get_title() == '':
                axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def calculate_disease_statistics(pred_mask, leaf_mask=None):
    """Calculate comprehensive disease statistics"""
    if leaf_mask is None:
        leaf_mask = np.ones_like(pred_mask, dtype=bool)
    
    # Ensure masks are boolean
    pred_mask_bool = pred_mask.astype(bool)
    leaf_mask_bool = leaf_mask.astype(bool)
    
    # Calculate statistics
    total_leaf_pixels = np.sum(leaf_mask_bool)
    
    if total_leaf_pixels == 0:
        return {
            'disease_percentage': 0.0,
            'disease_pixels': 0,
            'healthy_pixels': 0,
            'coverage': 0.0
        }
    
    disease_pixels = np.sum(pred_mask_bool & leaf_mask_bool)
    disease_percentage = (disease_pixels / total_leaf_pixels) * 100
    
    # Calculate disease clusters (connected components)
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(pred_mask_bool & leaf_mask_bool)
    
    # Get sizes of connected components
    component_sizes = []
    if num_features > 0:
        for i in range(1, num_features + 1):
            component_sizes.append(np.sum(labeled_array == i))
    
    return {
        'disease_percentage': float(disease_percentage),
        'disease_pixels': int(disease_pixels),
        'healthy_pixels': int(total_leaf_pixels - disease_pixels),
        'coverage': float(disease_percentage / 100),
        'num_clusters': num_features,
        'avg_cluster_size': float(np.mean(component_sizes)) if component_sizes else 0.0,
        'max_cluster_size': float(np.max(component_sizes)) if component_sizes else 0.0
    }

# ================ ROUTES ================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', models_loaded=MODELS_LOADED)

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': MODELS_LOADED,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'models': list(predictor.models.keys()) if MODELS_LOADED else []
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle single file upload and prediction"""
    start_time = datetime.now()
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file part in request'
        }), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    try:
        # Save uploaded file
        filename, filepath = save_uploaded_file(file)
        
        # Read and process image
        image = read_image_file(filepath)
        original_shape = image.shape
        
        # Remove background to get leaf mask
        leaf_mask = remove_background(image) if MODELS_LOADED else np.ones(image.shape[:2], dtype=bool)
        
        # Get predictions from all models
        if MODELS_LOADED:
            results = predict_all(image)
        else:
            results = predict_all(image)  # Use mock predictions
        
        # Extract predictions and statistics
        model_names = ['Logistic Regression', 'Random Forest', 'CNN']
        model_keys = ['logistic_regression', 'random_forest', 'cnn']
        
        predictions = []
        model_predictions = []
        disease_percentages = []
        
        for display_name, key in zip(model_names, model_keys):
            if key in results and key != 'best_model' and key != 'leaf_mask':
                pred_mask = results[key]['mask']
                stats = calculate_disease_statistics(pred_mask, leaf_mask)
                
                predictions.append(pred_mask)
                disease_percentages.append(stats['disease_percentage'])
                
                model_predictions.append({
                    'model': display_name,
                    'model_key': key,
                    'disease_percentage': stats['disease_percentage'],
                    'disease_pixels': stats['disease_pixels'],
                    'healthy_pixels': stats['healthy_pixels'],
                    'num_clusters': stats['num_clusters'],
                    'avg_cluster_size': stats['avg_cluster_size'],
                    'confidence': results[key].get('confidence', 0.85),
                    'coverage': stats['coverage']
                })
        
        # Determine best model (lowest disease percentage for demo)
        # In reality, you might choose based on validation accuracy
        if disease_percentages:
            # For demo: choose model with median disease percentage
            # (avoiding extremes)
            median_idx = np.argsort(disease_percentages)[len(disease_percentages) // 2]
            best_model = model_names[median_idx]
            best_model_key = model_keys[median_idx]
        else:
            best_model = 'CNN'
            best_model_key = 'cnn'
        
        # Create visualization
        viz_base64 = create_visualization(
            image, predictions, model_names, disease_percentages, leaf_mask
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response data
        response_data = {
            'success': True,
            'filename': filename,
            'original_filename': secure_filename(file.filename),
            'file_size': os.path.getsize(filepath),
            'image_size': {
                'height': int(original_shape[0]),
                'width': int(original_shape[1]),
                'channels': int(original_shape[2]) if len(original_shape) > 2 else 1
            },
            'leaf_coverage': float(np.mean(leaf_mask) * 100),
            'leaf_pixels': int(np.sum(leaf_mask)),
            'best_model': best_model,
            'best_model_key': best_model_key,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'model_predictions': model_predictions,
            'visualization': viz_base64,
            'models_used': model_names,
            'models_loaded': MODELS_LOADED
        }
        
        # Save results to file for history
        result_id = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        results_file = os.path.join('static/results', f'{result_id}.json')
        
        # Add result_id to response
        response_data['result_id'] = result_id
        
        # Save full results (including masks as base64 for smaller size)
        save_data = response_data.copy()
        
        # Convert masks to base64 for storage
        for i, pred_mask in enumerate(predictions):
            if pred_mask is not None:
                # Convert mask to PNG and encode as base64
                mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                buffered = io.BytesIO()
                mask_img.save(buffered, format="PNG")
                mask_base64 = base64.b64encode(buffered.getvalue()).decode()
                save_data['model_predictions'][i]['mask_base64'] = mask_base64
        
        # Remove visualization from saved data (already saved separately)
        if 'visualization' in save_data:
            viz_filename = f"{result_id}_viz.png"
            viz_path = os.path.join('static/visualizations', viz_filename)
            
            # Save visualization as image file
            viz_data = base64.b64decode(save_data['visualization'])
            with open(viz_path, 'wb') as f:
                f.write(viz_data)
            
            # Update response with URL to visualization
            response_data['visualization_url'] = url_for('static', 
                filename=f'visualizations/{viz_filename}', _external=True)
            
            # Remove base64 from saved data to reduce size
            del save_data['visualization']
        
        # Save results as JSON
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle multiple file uploads"""
    if 'files[]' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files provided'
        }), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({
            'success': False,
            'error': 'No files selected'
        }), 400
    
    results = []
    errors = []
    
    for file in files:
        try:
            # Check file type
            if not allowed_file(file.filename):
                errors.append({
                    'filename': file.filename,
                    'error': 'File type not allowed'
                })
                continue
            
            # Save file
            filename, filepath = save_uploaded_file(file)
            
            # Read image
            image = read_image_file(filepath)
            
            # Get predictions (use CNN as default for batch)
            if MODELS_LOADED:
                pred_result = predict_with_cnn(image)
                leaf_mask = remove_background(image)
            else:
                # Mock prediction for batch
                pred_result = create_mock_prediction(image, pattern='patchy')
                leaf_mask = np.ones(image.shape[:2], dtype=bool)
            
            # Calculate statistics
            stats = calculate_disease_statistics(pred_result, leaf_mask)
            
            # Determine risk level
            disease_pct = stats['disease_percentage']
            if disease_pct > 30:
                status = 'HIGH RISK'
            elif disease_pct > 10:
                status = 'MEDIUM RISK'
            else:
                status = 'LOW RISK'
            
            results.append({
                'filename': filename,
                'original_filename': secure_filename(file.filename),
                'disease_percentage': disease_pct,
                'disease_pixels': stats['disease_pixels'],
                'healthy_pixels': stats['healthy_pixels'],
                'status': status,
                'num_clusters': stats['num_clusters'],
                'model': 'CNN' if MODELS_LOADED else 'Mock',
                'processed_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            errors.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    # Calculate summary statistics
    if results:
        disease_pcts = [r['disease_percentage'] for r in results]
        summary = {
            'total_files': len(results),
            'successful': len(results),
            'failed': len(errors),
            'avg_disease': float(np.mean(disease_pcts)),
            'max_disease': float(np.max(disease_pcts)),
            'min_disease': float(np.min(disease_pcts)),
            'high_risk_count': sum(1 for r in results if r['status'] == 'HIGH RISK'),
            'medium_risk_count': sum(1 for r in results if r['status'] == 'MEDIUM RISK'),
            'low_risk_count': sum(1 for r in results if r['status'] == 'LOW RISK')
        }
    else:
        summary = {
            'total_files': 0,
            'successful': 0,
            'failed': len(errors)
        }
    
    return jsonify({
        'success': True,
        'summary': summary,
        'results': results,
        'errors': errors,
        'total_processed': len(results) + len(errors)
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    start_time = datetime.now()
    
    # Check for image data
    if 'image' not in request.files and 'image_base64' not in request.json:
        return jsonify({
            'success': False,
            'error': 'No image data provided. Use "image" file or "image_base64" in JSON'
        }), 400
    
    try:
        # Get model preference
        model_type = request.json.get('model', 'cnn') if request.json else 'cnn'
        model_type = model_type.lower()
        
        # Get image
        if 'image' in request.files:
            # From file upload
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            _, filepath = save_uploaded_file(file)
            image = read_image_file(filepath)
            
        else:
            # From base64
            image_data = request.json['image_base64']
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        
        # Preprocess
        image_processed = preprocess_image(image)
        
        # Select model
        if model_type == 'lr':
            pred_mask = predict_with_lr(image_processed)
            model_name = 'Logistic Regression'
        elif model_type == 'rf':
            pred_mask = predict_with_rf(image_processed)
            model_name = 'Random Forest'
        else:  # cnn or default
            pred_mask = predict_with_cnn(image_processed)
            model_name = 'CNN'
        
        # Remove background
        leaf_mask = remove_background(image_processed) if MODELS_LOADED else np.ones(image_processed.shape[:2], dtype=bool)
        
        # Calculate statistics
        stats = calculate_disease_statistics(pred_mask, leaf_mask)
        
        # Convert mask to base64
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'model_key': model_type,
            'disease_percentage': stats['disease_percentage'],
            'disease_pixels': stats['disease_pixels'],
            'healthy_pixels': stats['healthy_pixels'],
            'leaf_coverage': float(np.mean(leaf_mask) * 100),
            'mask_base64': mask_base64,
            'mask_shape': pred_mask.shape,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/history')
def get_history():
    """Get prediction history"""
    try:
        history_files = []
        results_dir = 'static/results'
        
        if os.path.exists(results_dir):
            for filename in sorted(os.listdir(results_dir), reverse=True):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Extract essential info
                        history_files.append({
                            'id': filename.replace('.json', ''),
                            'filename': data.get('filename', 'Unknown'),
                            'timestamp': data.get('timestamp', 'Unknown'),
                            'best_model': data.get('best_model', 'Unknown'),
                            'disease_percentage': max(
                                [p.get('disease_percentage', 0) for p in data.get('model_predictions', [])]
                            ) if data.get('model_predictions') else 0,
                            'processing_time': data.get('processing_time', 0)
                        })
                    except:
                        continue
        
        return jsonify({
            'success': True,
            'total_results': len(history_files),
            'results': history_files[:50]  # Limit to 50 most recent
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/result/<result_id>')
def get_result(result_id):
    """Get specific result by ID"""
    result_file = os.path.join('static/results', f'{result_id}.json')
    
    if not os.path.exists(result_file):
        return jsonify({
            'success': False,
            'error': 'Result not found'
        }), 404
    
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        # Add visualization URL if exists
        viz_filename = f"{result_id}_viz.png"
        viz_path = os.path.join('static/visualizations', viz_filename)
        
        if os.path.exists(viz_path):
            result_data['visualization_url'] = url_for('static', 
                filename=f'visualizations/{viz_filename}', _external=True)
        
        return jsonify({
            'success': True,
            'result': result_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<result_id>')
def download_result(result_id):
    """Download result as JSON file"""
    result_file = os.path.join('static/results', f'{result_id}.json')
    
    if not os.path.exists(result_file):
        return jsonify({
            'success': False,
            'error': 'Result not found'
        }), 404
    
    return send_file(
        result_file,
        as_attachment=True,
        download_name=f'{result_id}.json',
        mimetype='application/json'
    )

@app.route('/stats')
def get_stats():
    """Get application statistics"""
    try:
        total_uploads = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                           if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))])
        
        total_results = len([f for f in os.listdir('static/results') 
                           if f.endswith('.json')])
        
        total_viz = len([f for f in os.listdir('static/visualizations') 
                        if f.endswith('.png')])
        
        # Calculate disk usage
        def get_folder_size(folder):
            total = 0
            for dirpath, dirnames, filenames in os.walk(folder):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
            return total
        
        uploads_size = get_folder_size(app.config['UPLOAD_FOLDER'])
        results_size = get_folder_size('static/results')
        viz_size = get_folder_size('static/visualizations')
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_uploads': total_uploads,
                'total_results': total_results,
                'total_visualizations': total_viz,
                'disk_usage': {
                    'uploads': f"{uploads_size / (1024*1024):.2f} MB",
                    'results': f"{results_size / 1024:.2f} KB",
                    'visualizations': f"{viz_size / (1024*1024):.2f} MB",
                    'total': f"{(uploads_size + results_size + viz_size) / (1024*1024):.2f} MB"
                },
                'models_loaded': MODELS_LOADED,
                'models_available': list(predictor.models.keys()) if MODELS_LOADED else []
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all history (admin function)"""
    try:
        # Clear uploads
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Clear results
        results_dir = 'static/results'
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        # Clear visualizations
        viz_dir = 'static/visualizations'
        if os.path.exists(viz_dir):
            for filename in os.listdir(viz_dir):
                file_path = os.path.join(viz_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        return jsonify({
            'success': True,
            'message': 'History cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/models/status')
def models_status():
    """Check status of all models"""
    if not MODELS_LOADED:
        return jsonify({
            'success': False,
            'message': 'Models not loaded'
        })
    
    model_status = {}
    for name, model in predictor.models.items():
        model_status[name] = {
            'loaded': model is not None,
            'type': str(type(model)) if model else 'Not loaded'
        }
    
    return jsonify({
        'success': True,
        'models_loaded': MODELS_LOADED,
        'model_status': model_status
    })

# ================ ERROR HANDLERS ================

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ================ MAIN ================

if __name__ == '__main__':
    print("=" * 60)
    print("Leaf Disease Segmentation Flask Application")
    print("=" * 60)
    print(f"Models Loaded: {'YES' if MODELS_LOADED else 'NO (using mock predictions)'}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Allowed Extensions: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    print("\nEndpoints:")
    print("  GET  /                    - Main interface")
    print("  POST /upload              - Upload single image")
    print("  POST /batch_predict       - Upload multiple images")
    print("  POST /api/predict         - API endpoint (JSON)")
    print("  GET  /history             - View prediction history")
    print("  GET  /result/<id>         - Get specific result")
    print("  GET  /stats               - Application statistics")
    print("  GET  /api/health          - Health check")
    print("  GET  /models/status       - Model status")
    print("\nStarting server on http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

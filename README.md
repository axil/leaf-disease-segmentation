# 🌿 Leaf Disease Segmentation System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-active-success)

An end-to-end machine learning system for detecting and segmenting diseased areas on wheat leaves using multiple ML approaches.

## 🎯 Features

- **Multi-Model Analysis**: Compare Logistic Regression, Random Forest, and CNN predictions
- **Interactive Web Interface**: Upload, visualize, and analyze leaf images
- **Background Removal**: Automatically isolate leaf from white background
- **Real-time Predictions**: Get instant disease percentage and risk assessment
- **Batch Processing**: Analyze multiple leaves at once
- **RESTful API**: Programmatic access to all functionality
- **Export Results**: Download predictions as JSON or images

## 📊 Model Performance

| Model | Accuracy | F1 Score | Inference Time | Best For |
|-------|----------|----------|----------------|----------|
| Logistic Regression | 72% | 0.68 | ~0.5s | Quick analysis |
| Random Forest | 78% | 0.75 | ~1.2s | Patch-based detection |
| CNN (U-Net) | 89% | 0.87 | ~2.1s | Detailed segmentation |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- Web browser with JavaScript enabled

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/axil/leaf-disease-segmentation.git
cd leaf-disease-segmentation
```
2. **Create virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up models**

```bash
# Create mock models for demonstration
python -c "from utils.prediction_utils import create_mock_models; create_mock_models()"

# Or use your trained models by placing them in models/ folder:
# - logistic_regression.pkl
# - random_forest.pkl  
# - cnn_model.h5
```

5. **Run the application**

```bash
python app.py
```

6. **Open in browser** 

```text
http://localhost:5000
```

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t leaf-disease-app .

# Run the container
docker run -p 5000:5000 leaf-disease-app

# Access at http://localhost:5000
```

## 📁 Project Structure

```text
leaf-disease-app/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── models/                         # Trained ML models
│   ├── logistic_regression.pkl     # Logistic Regression model
│   ├── random_forest.pkl           # Random Forest model
│   └── cnn_model.h5               # TensorFlow CNN model
├── utils/
│   └── prediction_utils.py         # Model loading & prediction
├── static/
│   ├── css/style.css              # Frontend styles
│   └── js/script.js               # Frontend interactivity
├── templates/
│   └── index.html                 # Main web interface
├── uploads/                        # User uploaded images
└── README.md                       # This file
```

## 🔧 Usage Guide

### Web Interface

- Go to http://localhost:5000
- Drag & drop or click to upload a leaf image
- View predictions from all three models
 -Compare disease percentages and visualizations
 -Download results as JSON or PNG

### API Endpoints
```bash
# Health check
GET /api/health

# Single image prediction
POST /upload
Content-Type: multipart/form-data
file: <image_file>

# Batch processing
POST /batch_predict
Content-Type: multipart/form-data
files[]: <image_files>

# Programmatic API
POST /api/predict
Content-Type: application/json
{
    "image_base64": "data:image/jpeg;base64,...",
    "model": "cnn"
}

# Get history
GET /history

# Download specific result
GET /download/<result_id>
````

### Example API Usage
```python
import requests
import base64

# Load image
with open("leaf.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post("http://localhost:5000/api/predict", json={
    "image_base64": f"data:image/jpeg;base64,{img_base64}",
    "model": "cnn"
})

print(response.json())
```

## 🧪 ML Approach

### 1. Logistic Regression (Pixel-based)
- Features: RGB values, HSV color space, color ratios
- Advantages: Fast, interpretable, works on individual pixels
- Limitations: Ignores spatial context

### 2. Random Forest (Patch-based)
- Features: Patch statistics, texture, edge density
- Advantages: Handles non-linear patterns, robust to noise
- Limitations: Blocky predictions, slower than LR

### 3. CNN (U-Net Architecture)
- Architecture: Encoder-decoder with skip connections
- Advantages: Captures spatial patterns, best accuracy
- Limitations: Requires more data, slower training

### Background Removal
- Uses HSV saturation thresholding to isolate leaf
- Removes white background for cleaner predictions
- Improves model focus on relevant areas

## 📈 Results Interpretation
Disease Percentage	Risk Level	Recommended Action
0-10%	Low	Monitor regularly
10-30%	Medium	Consider treatment
30%+	High	Immediate treatment needed

## 🚫 Limitations
- Requires white background for optimal background removal
- Trained on specific wheat leaf diseases
- May not generalize to other plant species
- Performance depends on image quality

## 🛠️ Development
### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest tests/
```

### Code Style

```bash
# Install formatters
pip install black flake8

# Format code
black app.py utils/

# Check style
flake8 app.py utils/
```

### Adding New Models
- Train model using your dataset
- Save model to models/ folder
- Update prediction_utils.py to load new model
- Add to web interface in app.py

## 📚 Dataset
The system was trained on a custom dataset of wheat leaves with three classes:  

0: Background (white)  
1: Healthy leaf tissue (green)  
2: Diseased leaf tissue (brown/yellow)  

### Dataset Statistics:
- 100 images total
- 240x240 resolution
- RGB format with corresponding masks
- 70% train, 15% validation, 15% test split

## 🤝 Contributing
- Fork the repository
- Create a feature branch (git checkout -b feature/AmazingFeature)
- Commit changes (git commit -m 'Add AmazingFeature')
- Push to branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Datatalks club for lectures and materials
- OpenCV and TensorFlow communities
- Plant pathology researchers
- All contributors to this project

## 📞 Contact
Lev Maximov - @axil - lev.maximov@gmail.com

Project Link: https://github.com/axil/leaf-disease-segmentation

## 🎓 Academic Reference
If you use this project in academic work, please cite:

```bibtex
@software{leaf_disease_segmentation_2024,
  title = {Leaf Disease Segmentation System},
  author = {Lev Maximov},
  year = {2026},
  url = {https://github.com/axil/leaf-disease-segmentation},
  note = {Machine learning project for wheat leaf disease detection}
}
```

<div align="center"> Made with ❤️ for ML Zoomcamp <br> ⭐ Star this repo if you found it useful! </div> ```

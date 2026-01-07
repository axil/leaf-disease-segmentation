##  Running standalone

# 1. Navigate to your project folder
cd leaf_disease_app

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create mock model files
python create_mock_models.py

# 5. Run Flask app
python app.py

# 6. Open browser: http://localhost:5000



##  Running with Docker 

# 1. Build Docker image
docker build -t leaf-disease-app .

# 2. Run container
docker run -p 5000:5000 leaf-disease-app

# 3. Open browser: http://localhost:5000
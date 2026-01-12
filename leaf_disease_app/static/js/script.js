// static/js/script.js

// Global variables
let currentResults = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    setupDragAndDrop();
    setupFileInput();
    setupBatchProcessing();
    setupSampleButton();
    setupExportButtons();
    
    // Check for existing results in URL
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('result_id')) {
        loadPreviousResult(urlParams.get('result_id'));
    }
});

function setupDragAndDrop() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        });
    });
    
    // Remove highlight when leaving
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        });
    });
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processSingleFile(files[0]);
        }
    }
}

function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', function(e) {
        if (this.files.length > 0) {
            processSingleFile(this.files[0]);
        }
    });
}

function setupBatchProcessing() {
    const batchInput = document.getElementById('batchInput');
    const batchBtn = document.getElementById('batchBtn');
    
    batchInput.addEventListener('change', function() {
        batchBtn.disabled = this.files.length === 0;
        batchBtn.textContent = `Process ${this.files.length} Files`;
    });
    
    batchBtn.addEventListener('click', processBatchFiles);
}

function setupSampleButton() {
    document.getElementById('sampleBtn').addEventListener('click', async function() {

      try {
          // Fetch the sample image from your images folder
          const response = await fetch('/static/images/0629-00.png');
          
          if (!response.ok) {
              throw new Error(`Failed to load sample image: ${response.status}`);
          }
          
          // Convert response to blob
          const blob = await response.blob();
          
          // Create a File object from the blob
          const file = new File([blob], 'sample_leaf.png', { type: 'image/png' });
          
          // Process the file as if user uploaded it
          await processSingleFile(file);
          
      } catch (error) {
          console.error('Error loading sample image:', error);
          alert('Could not load sample image. Please upload your own image.');
      }
  
  });
}

function setupExportButtons() {
    document.getElementById('downloadJsonBtn').addEventListener('click', downloadJSON);
    document.getElementById('downloadImageBtn').addEventListener('click', downloadImage);
//    document.getElementById('shareResultsBtn').addEventListener('click', shareResults);
}

async function processSingleFile(file) {
    if (!validateFile(file)) {
        return;
    }
    
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        currentResults = data;
        displayResults(data);
        
        // Update URL with result ID (if available)
        if (data.result_id) {
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('result_id', data.result_id);
            window.history.pushState({}, '', newUrl);
        }
        
    } catch (error) {
        showError('Error processing image: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function processBatchFiles() {
    const files = document.getElementById('batchInput').files;
    if (files.length === 0) return;
    
    showLoading(true);
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        if (!validateFile(files[i])) {
            continue;
        }
        formData.append('files[]', files[i]);
    }
    
    try {
        const response = await fetch('/batch_predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayBatchResults(data);
        
    } catch (error) {
        showError('Batch processing failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function validateFile(file) {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload an image file (JPG, PNG, BMP, TIFF)');
        return false;
    }
    
    if (file.size > maxSize) {
        alert('File is too large. Maximum size is 16MB');
        return false;
    }
    
    return true;
}

function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').classList.remove('d-none');
    
    // Update basic info
    const resultsInfo = document.getElementById('resultsInfo');
    resultsInfo.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <p><strong>Filename:</strong> ${data.filename}</p>
                <p><strong>Image Size:</strong> ${data.image_size[1]} × ${data.image_size[0]} pixels</p>
                <p><strong>Leaf Coverage:</strong> ${data.leaf_coverage.toFixed(1)}%</p>
            </div>
            <div class="col-md-6">
                <p><strong>Best Model:</strong> ${data.best_model}</p>
                <p><strong>Processing Time:</strong> ${data.processing_time || 'N/A'} seconds</p>
                <p><strong>Timestamp:</strong> ${new Date().toLocaleString()}</p>
            </div>
        </div>
    `;
    
    // Update visualization
    const vizImg = document.getElementById('visualization');
    if (data.visualization) {
        vizImg.src = `data:image/png;base64,${data.visualization}`;
        vizImg.style.display = 'block';
    } else {
        vizImg.style.display = 'none';
    }
    
    // Display model results
    const modelResults = document.getElementById('modelResults');
    modelResults.innerHTML = '';
    
    data.model_predictions.forEach((pred, index) => {
        const riskLevel = getRiskLevel(pred.disease_percentage);
        
        const card = `
            <div class="col-md-4 mb-4">
                <div class="card model-card h-100">
                    <div class="card-header ${getModelColorClass(pred.model)}">
                        <h5 class="card-title mb-0">
                            <i class="${getModelIcon(pred.model)} me-2"></i>
                            ${pred.model}
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="disease-percentage mb-3">
                            <h3 class="${getRiskColorClass(riskLevel)}">
                                ${pred.disease_percentage.toFixed(1)}%
                            </h3>
                            <p class="text-muted">Disease Coverage</p>
                        </div>
                        
                        <div class="mb-3">
                            <strong>Disease Pixels:</strong> 
                            <span class="badge bg-secondary">${pred.disease_pixels.toLocaleString()}</span>
                        </div>
                        
                        <div class="progress mb-3" style="height: 20px;">
                            <div class="progress-bar ${getRiskColorClass(riskLevel, 'bg')}" 
                                 role="progressbar" 
                                 style="width: ${Math.min(pred.disease_percentage, 100)}%">
                                ${pred.disease_percentage.toFixed(1)}%
                            </div>
                        </div>
                        
                        <div class="risk-assessment">
                            <span class="badge ${getRiskColorClass(riskLevel, 'badge')}">
                                ${riskLevel} RISK
                            </span>
                        </div>
                    </div>
                    <div class="card-footer text-muted">
                        Confidence: ${(pred.confidence || 0.85).toFixed(2)}
                    </div>
                </div>
            </div>
        `;
        
        modelResults.innerHTML += card;
    });
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

function displayBatchResults(data) {
    const container = document.getElementById('batchResults');
    
    let html = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-clipboard-list me-2"></i>
                    Batch Results (${data.total_files} files)
                </h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Disease %</th>
                                <th>Status</th>
                                <th>Model</th>
                            </tr>
                        </thead>
                        <tbody>
    `;
    
    data.results.forEach(result => {
        const riskLevel = getRiskLevel(result.disease_percentage);
        
        html += `
            <tr>
                <td>${result.filename}</td>
                <td>
                    <strong>${result.disease_percentage.toFixed(1)}%</strong>
                    <div class="progress" style="height: 5px;">
                        <div class="progress-bar ${getRiskColorClass(riskLevel, 'bg')}" 
                             style="width: ${Math.min(result.disease_percentage, 100)}%">
                        </div>
                    </div>
                </td>
                <td>
                    <span class="badge ${getRiskColorClass(riskLevel, 'badge')}">
                        ${result.status}
                    </span>
                </td>
                <td>${result.model || 'CNN'}</td>
            </tr>
        `;
    });
    
    html += `
                        </tbody>
                    </table>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-4">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h6 class="card-title">Average Disease</h6>
                                <h2 class="${getRiskColorClass(getRiskLevel(data.summary.avg_disease))}">
                                    ${data.summary.avg_disease.toFixed(1)}%
                                </h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h6 class="card-title">Maximum Disease</h6>
                                <h2 class="text-danger">
                                    ${data.summary.max_disease.toFixed(1)}%
                                </h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h6 class="card-title">High Risk Files</h6>
                                <h2 class="text-warning">
                                    ${data.summary.high_risk_count}
                                </h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    
    // Show container
    container.classList.remove('d-none');
}

// Helper functions
function getRiskLevel(percentage) {
    if (percentage > 30) return 'HIGH';
    if (percentage > 10) return 'MEDIUM';
    return 'LOW';
}

function getRiskColorClass(riskLevel, type = 'text') {
    const prefix = type === 'badge' ? 'badge-' : type === 'bg' ? 'bg-' : 'text-';
    
    switch (riskLevel) {
        case 'HIGH': return prefix + 'danger';
        case 'MEDIUM': return prefix + 'warning';
        case 'LOW': return prefix + 'success';
        default: return prefix + 'secondary';
    }
}

function getModelColorClass(modelName) {
    if (modelName.includes('Logistic')) return 'bg-danger text-white';
    if (modelName.includes('Random')) return 'bg-warning text-dark';
    if (modelName.includes('CNN')) return 'bg-info text-white';
    return 'bg-secondary text-white';
}

function getModelIcon(modelName) {
    if (modelName.includes('Logistic')) return 'fas fa-chart-line';
    if (modelName.includes('Random')) return 'fas fa-tree';
    if (modelName.includes('CNN')) return 'fas fa-brain';
    return 'fas fa-cube';
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    const uploadArea = document.getElementById('dropArea');
    
    if (show) {
        spinner.classList.remove('d-none');
        uploadArea.style.opacity = '0.5';
        uploadArea.style.pointerEvents = 'none';
    } else {
        spinner.classList.add('d-none');
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
    }
}

function showError(message) {
    // Create or show error alert
    let errorAlert = document.getElementById('errorAlert');
    
    if (!errorAlert) {
        errorAlert = document.createElement('div');
        errorAlert.id = 'errorAlert';
        errorAlert.className = 'alert alert-danger alert-dismissible fade show';
        errorAlert.role = 'alert';
        errorAlert.innerHTML = `
            <span id="errorMessage"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.querySelector('.main-container').prepend(errorAlert);
    }
    
    document.getElementById('errorMessage').textContent = message;
    errorAlert.classList.remove('d-none');
}

function downloadJSON() {
    if (!currentResults) return;
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `leaf_analysis_${Date.now()}.json`;
    link.click();
    
    URL.revokeObjectURL(link.href);
}

function downloadImage() {
    if (!currentResults || !currentResults.visualization) return;
    
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${currentResults.visualization}`;
    link.download = `leaf_analysis_${Date.now()}.png`;
    link.click();
}

function shareResults() {
    if (!currentResults) return;
    
    // Create a shareable summary
    const summary = `
Leaf Disease Analysis Results:
- Best Model: ${currentResults.best_model}
- Disease Percentage: ${Math.max(...currentResults.model_predictions.map(p => p.disease_percentage)).toFixed(1)}%
- Risk Level: ${getRiskLevel(Math.max(...currentResults.model_predictions.map(p => p.disease_percentage)))}
        
Generated at: ${new Date().toLocaleString()}
    `;
    
    // Copy to clipboard
    navigator.clipboard.writeText(summary)
        .then(() => alert('Results copied to clipboard!'))
        .catch(() => prompt('Copy the following text:', summary));
}

function loadPreviousResult(resultId) {
    // In a real app, you would fetch from server
    console.log('Loading previous result:', resultId);
    // You could implement this to load saved results
}

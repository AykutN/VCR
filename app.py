#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask Web Application for Deepfake Voice Detection

This application provides a web interface for detecting deepfake voices.
It supports rule-based, ML-based, and hybrid detection methods.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from batch_test import detect_deepfake
from ml_detector import detect_with_ml
from hybrid_detector import detect_hybrid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect deepfake in uploaded audio file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: wav, mp3, flac, ogg, m4a'}), 400
    
    # Get detection method
    method = request.form.get('method', 'hybrid')  # hybrid, rule, ml
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # Perform detection based on method
        if method == 'rule':
            result = detect_deepfake(
                filepath,
                real_dir='data/real',
                threshold=0.34
            )
            response = {
                'is_fake': result.get('is_fake'),
                'score': result.get('score'),
                'confidence': result.get('confidence'),
                'method': 'rule-based',
                'details': result.get('feature_analysis', {})
            }
        
        elif method == 'ml':
            result = detect_with_ml(filepath, models_dir='models')
            response = {
                'is_fake': result.get('is_fake'),
                'score': result.get('combined_score'),
                'confidence': result.get('confidence'),
                'method': 'ml-based',
                'details': {
                    'lr_score': result.get('lr_score'),
                    'svm_score': result.get('svm_score')
                }
            }
        
        else:  # hybrid
            result = detect_hybrid(
                filepath,
                real_dir='data/real',
                models_dir='models',
                rule_threshold=0.34,
                ml_weight=0.5,
                rule_weight=0.5
            )
            response = {
                'is_fake': result.get('is_fake'),
                'score': result.get('hybrid_score'),
                'confidence': result.get('confidence'),
                'method': 'hybrid',
                'details': {
                    'rule_score': result.get('rule_score'),
                    'ml_score': result.get('ml_score'),
                    'weights': result.get('weights')
                }
            }
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        # Clean up on error
        try:
            os.remove(filepath)
        except:
            pass
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    import sys
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print("="*70)
    print("Starting Deepfake Voice Detection Web Interface")
    print("="*70)
    print(f"Server will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("="*70)
    app.run(debug=True, host='127.0.0.1', port=port)


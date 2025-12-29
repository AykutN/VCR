document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('audioFile');
    const fileLabel = document.getElementById('fileLabel');
    const detectBtn = document.getElementById('detectBtn');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    const resultDetails = document.getElementById('resultDetails');

    // Update file label when file is selected
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            fileLabel.textContent = file.name;
        } else {
            fileLabel.textContent = 'Click to upload or drag and drop';
        }
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an audio file');
            return;
        }

        const formData = new FormData(form);
        const method = form.querySelector('input[name="method"]:checked').value;

        // Show loading state
        detectBtn.disabled = true;
        loader.style.display = 'inline';
        detectBtn.querySelector('.btn-text').style.display = 'none';
        resultSection.style.display = 'none';

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            detectBtn.disabled = false;
            loader.style.display = 'none';
            detectBtn.querySelector('.btn-text').style.display = 'inline';
        }
    });

    function displayResult(data) {
        resultSection.style.display = 'block';
        
        // Set icon and text based on result
        if (data.is_fake === true) {
            resultIcon.textContent = '🚨';
            resultText.textContent = 'FAKE DETECTED';
            resultText.className = 'result-text result-fake';
        } else if (data.is_fake === false) {
            resultIcon.textContent = '✅';
            resultText.textContent = 'REAL VOICE';
            resultText.className = 'result-text result-real';
        } else {
            resultIcon.textContent = '❓';
            resultText.textContent = 'UNCERTAIN';
            resultText.className = 'result-text';
        }

        // Build details HTML
        let detailsHTML = '';

        // Main score
        detailsHTML += `
            <div class="detail-item">
                <span class="detail-label">Detection Score:</span>
                <span class="detail-value">${(data.score * 100).toFixed(2)}%</span>
            </div>
            <div class="score-bar">
                <div class="score-fill" style="width: ${data.score * 100}%">
                    ${(data.score * 100).toFixed(1)}%
                </div>
            </div>
        `;

        // Confidence
        detailsHTML += `
            <div class="detail-item">
                <span class="detail-label">Confidence:</span>
                <span class="detail-value">${(data.confidence * 100).toFixed(2)}%</span>
            </div>
        `;

        // Method-specific details
        if (data.method === 'hybrid' && data.details) {
            detailsHTML += `
                <div class="detail-item">
                    <span class="detail-label">Rule-Based Score:</span>
                    <span class="detail-value">${(data.details.rule_score * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">ML-Based Score:</span>
                    <span class="detail-value">${(data.details.ml_score * 100).toFixed(2)}%</span>
                </div>
            `;
        } else if (data.method === 'ml' && data.details) {
            detailsHTML += `
                <div class="detail-item">
                    <span class="detail-label">Logistic Regression:</span>
                    <span class="detail-value">${(data.details.lr_score * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">SVM Score:</span>
                    <span class="detail-value">${(data.details.svm_score * 100).toFixed(2)}%</span>
                </div>
            `;
        }

        detailsHTML += `
            <div class="detail-item">
                <span class="detail-label">Method Used:</span>
                <span class="detail-value">${data.method.toUpperCase()}</span>
            </div>
        `;

        resultDetails.innerHTML = detailsHTML;

        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});


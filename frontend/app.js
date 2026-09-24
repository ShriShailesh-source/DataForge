const uploadInput = document.getElementById('dataset-file');
const analyzeButton = document.getElementById('analyze-button');
const statusEl = document.getElementById('status');
const overviewEl = document.getElementById('overview');
const qualityEl = document.getElementById('quality-summary');
const reportEl = document.getElementById('report-output');

async function analyzeFile() {
  const file = uploadInput.files[0];
  if (!file) {
    statusEl.textContent = 'Please choose a CSV or Excel file first.';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  statusEl.textContent = 'Uploading dataset...';

  try {
    const uploadResponse = await fetch('http://127.0.0.1:8000/upload', {
      method: 'POST',
      body: formData,
    });
    const uploadData = await uploadResponse.json();
    if (!uploadResponse.ok) {
      throw new Error(uploadData.detail || 'Upload failed');
    }

    statusEl.textContent = 'Dataset uploaded. Running analysis...';

    const analyzeResponse = await fetch(`http://127.0.0.1:8000/analyze?file_name=${encodeURIComponent(uploadData.file_name)}`);
    const analyzeData = await analyzeResponse.json();
    if (!analyzeResponse.ok) {
      throw new Error(analyzeData.detail || 'Analysis failed');
    }

    overviewEl.textContent = JSON.stringify(analyzeData.report.dataset_overview, null, 2);
    qualityEl.textContent = JSON.stringify(analyzeData.quality, null, 2);
    reportEl.textContent = JSON.stringify({
      key_observations: analyzeData.report.key_observations,
      transformation_log: analyzeData.report.cleaning_transformation_log,
      charts: analyzeData.eda.charts,
    }, null, 2);
    statusEl.textContent = 'Analysis complete.';
  } catch (error) {
    statusEl.textContent = `Error: ${error.message}`;
  }
}

analyzeButton.addEventListener('click', analyzeFile);

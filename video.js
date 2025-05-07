const uploadBtn = document.getElementById('uploadBtn');
const videoInput = document.getElementById('videoInput');
const resultDiv = document.getElementById('result');

uploadBtn.addEventListener('click', async () => {
  const file = videoInput.files[0];
  if (!file) {
    resultDiv.textContent = 'Please upload a video.';
    return;
  }

  resultDiv.textContent = 'Analyzing video... This may take a few minutes.';

  const formData = new FormData();
  formData.append('video', file);

  try {
    const response = await fetch('https://your-api-url/video-detection', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    resultDiv.textContent = data.isManipulated
      ? '🚨 This video is MANIPULATED!'
      : '✅ This video is ORIGINAL!';
  } catch (error) {
    resultDiv.textContent = 'Error: Unable to check the video.';
  }
});

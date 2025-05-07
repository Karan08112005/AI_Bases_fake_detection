const uploadBtn = document.getElementById('uploadBtn');
const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');

uploadBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) {
    resultDiv.textContent = 'Please upload an image.';
    return;
  }

  resultDiv.textContent = 'Checking image...';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('https://your-api-url/image-detection', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    resultDiv.textContent = data.isMorphed
      ? '🚨 This image is MORPHED!'
      : '✅ This image is ORIGINAL!';
  } catch (error) {
    resultDiv.textContent = 'Error: Unable to check the image.';
  }
});

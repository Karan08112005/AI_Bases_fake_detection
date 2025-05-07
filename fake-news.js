document.getElementById('newsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
  
    const textarea = e.target.querySelector('textarea');
    const resultDiv = document.getElementById('result');
  
    resultDiv.textContent = 'Checking...';
  
    try {
      const response = await fetch('https://your-api-url/fake-news',{
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: textarea.value }),
      });
  
      const data = await response.json();
  
      resultDiv.textContent = data.isFake
        ? '🚨 This news is FAKE!'
        : '✅ This news is REAL!';
    } catch (error) {
      resultDiv.textContent = 'Error: Unable to check the news.';
    }
  });
  
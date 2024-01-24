
const form = document.getElementById('news-form');


form.addEventListener('submit', async (event) => {

  event.preventDefault();


  const input = document.getElementById('news-text').value;
  console.log(input)

  try {
    console.log(JSON.stringify({text: input}))
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({text: input}),
    });


    if (response.ok) {
 
        const jsonResponse = await response.json();
        console.log(jsonResponse)
        const prediction = jsonResponse.prediction;
        console.log(prediction)

      const resultDiv = document.getElementById('prediction-result');
      resultDiv.innerText = prediction < 0.5 ? 'The tweet is not suicidal': 'The tweet is suicidal.';
    } else {
      console.error('Request failed:', response.status);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
});

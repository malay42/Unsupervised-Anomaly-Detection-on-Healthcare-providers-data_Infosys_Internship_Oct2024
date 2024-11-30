document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const allowedAmount = document.getElementById('allowed-amount').value;
    const paymentAmount = document.getElementById('payment-amount').value;
    const beneficiaries = document.getElementById('beneficiaries').value;
    const gender = document.getElementById('gender').value;
    const resultElement = document.getElementById('prediction-result');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                allowed_amount: parseFloat(allowedAmount),
                payment_amount: parseFloat(paymentAmount),
                beneficiaries: parseInt(beneficiaries),
                gender: gender
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            const resultHTML = `
                <div class="${data.prediction.is_anomaly ? 'anomaly' : 'normal'}">
                    <p>Anomaly Score: ${data.prediction.score.toFixed(4)}</p>
                    <p>Threshold: ${data.prediction.threshold.toFixed(4)}</p>
                    <p>${data.prediction.is_anomaly ? 
                        '⚠️ This data point is ANOMALOUS!' : 
                        '✅ This data point is NOT anomalous.'}</p>
                </div>
            `;
            resultElement.innerHTML = resultHTML;
        } else {
            resultElement.innerHTML = `<div class="error">Error: ${data.message}</div>`;
        }
    } catch (error) {
        resultElement.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}); 
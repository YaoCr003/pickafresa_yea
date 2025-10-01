function renderTemperatureChart(labels, data) {
    const ctx = document.getElementById("tempChart").getContext("2d");

    const colors = data.map(light => {
        if (light >= 70) {
            return "green";  
        } else {
            return "blue"; 
        }
    });

    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Light %",
                data: data,
                borderColor: "#f5fd05ff",               
                backgroundColor: "rgba(232, 249, 0, 1)",  
                tension: 0.3,
                pointBackgroundColor: colors,      
                pointRadius: 5,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "°C"
                    }
                }
            }
        }
    });
}


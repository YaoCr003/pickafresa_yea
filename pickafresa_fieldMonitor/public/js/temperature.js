function renderTemperatureChart(labels, data) {
    const ctx = document.getElementById("tempChart").getContext("2d");

    // Generamos colores dinámicos
    const colors = data.map(temp => {
        if (temp < 20) {
            return "blue"; 
        } else if (temp < 30) {
            return "green"; 
        } else {
            return "orange"; 
        }
    });

    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Temperatura (°C)",
                data: data,
                borderColor: "#e46617ff",               
                backgroundColor: "rgba(255, 117, 31, 1)",  
                tension: 0.3,
                pointBackgroundColor: colors,      
                pointRadius: 5,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
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

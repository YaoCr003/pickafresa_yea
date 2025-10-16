function renderTemperatureChart(labels, data) {
    const ctx = document.getElementById("tempChart").getContext("2d");

    const colors = data.map(temp => {
        if (temp < 15) {
            return "blue"; 
        } else if (temp < 26) {
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
                label: "Temperature (°C)",
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

//Droplet effect
for (let i = 0; i < 20; i++) {
    const drop = document.createElement("div");
    drop.classList.add("drop");
    drop.style.left = Math.random() * 100 + "vw";
    drop.style.animationDuration = (Math.random() * 3 + 2) + "s";
    drop.style.opacity = Math.random();
    document.body.appendChild(drop);
}


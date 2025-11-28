const ctx = document.getElementById("lineChart").getContext("2d");

const data = {
    labels: labels,
    datasets: [
        {
            label: "🌡️ Temperature (°C)",
            data: temperature,
            borderColor: "rgba(230, 57, 70, 1)",
            backgroundColor: "rgba(230, 57, 70, 0.2)",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pontBackgroundColor: "rgba(230, 57, 70, 1)",
            fill: false
        },
        {
            label: "💧 Ambient Humidity (%)",
            data: ambientHumidity,
            borderColor: "rgba(72, 149, 239, 1)",
            backgroundColor: "rgba(72, 149, 239, 0.2)",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: "rgba(72, 149, 239, 1)",
            fill: false
        },
        {
            label: "🌱 Substrate moisture (%)",
            data: substrateMoisture,
            borderColor: "rgba(38, 166, 91, 1)",
            backgroundColor: "rgba(38, 166, 91, 0.2)",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: "rgba(38, 166, 91, 1)",
            fill: false
        },
        {
            label: "☀️ Light (%)",
            data: percentageLight,
            borderColor: "rgba(255, 193, 7, 1)",
            backgroundColor: "rgba(255, 193, 7, 0.2)",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: "rgba(255, 193, 7, 1)",
            fill: false
        }
    ]
};

const options = {
    responsive: true,
    maintainAspectRatio: false,   
    plugins: {
        legend: {
            position: "top",
            labels: {
                font: { size: 14, family: "Poppins" }
            }
        },
        tooltip: {
            mode: "index",
            intersect: false
        }
    },
    interaction: {
        mode: "nearest",
        axis: "x",
        intersect: false
    },
};

new Chart(ctx, {
    type: "line",
    data: data,
    options: options
});

window.addEventListener("load", () => {
    const loader = document.getElementById("loader");
    loader.classList.add("hidden");
});

document.querySelectorAll("a, button").forEach(el => {
    el.addEventListener("click", () => {
        const loader = document.getElementById("loader");
        loader.classList.remove("hidden");
    });
});
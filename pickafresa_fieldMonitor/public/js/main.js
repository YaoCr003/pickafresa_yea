alerts.forEach(msg => {
    const div = document.createElement("div");
    div.className = "alert";
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 4000);
});
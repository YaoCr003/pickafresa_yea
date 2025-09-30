const rain = document.querySelector(".rain");

for (let i = 0; i < 60; i++) {
    const drop = document.createElement("div");
    drop.classList.add("drop");
    drop.style.left = Math.random() * 100 + "vw";
    drop.style.animationDuration = (Math.random() * 2 + 1) + "s";
    drop.style.animationDelay = Math.random() * 5 + "s";
    rain.appendChild(drop);
}
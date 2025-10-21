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
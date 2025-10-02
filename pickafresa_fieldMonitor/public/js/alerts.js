document.addEventListener("DOMContentLoaded", () => {

    document.querySelectorAll(".styled-select").forEach(el => {
        new Choise(el, {
            searchPlaceholderValue: "Buscar...",
            noResultsText: "No hay resultados",
            itemSelectText: "Seleccionar",
            shouldSort: false
        });
    });

    // Calendario
    flatpickr("#startDate", {
        dateFormat: "Y-m-d",
        altInput: true,
        altFormat: "d/m/Y",
        locale: "es"
    });

    flatpickr("#endDate", {
        dateFormat: "Y-m-d",
        altInput: true,
        altFormat: "d/m/Y",
        locale: "es"
    });

    document.getElementById('resetBtn').addEventListener('click', () => {
        window.location.href = '/alerts'
    });
});
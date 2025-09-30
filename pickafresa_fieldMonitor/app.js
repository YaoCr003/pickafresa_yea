const express = require("express")
const mysql = require("mysql2")
const multer = require("multer")
const fs = require("fs")
const path = require("path")
const app = express()
const PORT = 3000;

// Connection to the database
const db = mysql.createConnection({
    host: "10.25.12.61", //Raspberry IP
    user: "a01275893",
    password: "a01275893",
    database: "fieldMonitor"
});

// EJS Configuration 
app.set("view engine", "ejs")
app.set("views", __dirname + "/views")
app.use(express.static(path.join(__dirname, "public")));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

/* 
============================================================
📌 ROUTES
============================================================ 
*/

// Main route 
app.get("/", (req,res) => {
    const sqlMesuraments = "SELECT * FROM measurements ORDER BY date DESC LIMIT 1";
    const sqlPhotos = "SELECT * FROM images ORDER BY date DESC LIMIT 1";

    db.query(sqlMesuraments, (errMed, measure) => {

        if (errMed) {
            console.error("Error in measurements:", errMed);
            return res.status(500).send("Error consulting measurements");
        }

        db.query(sqlPhotos, (errPhoto, images) => {

            if (errPhoto) {
                console.error("Error in photos:", errFoto);
                return res.status(500).send("Error when consulting photos");
            }

            // Convert images to base64
            const processedPhotos = images.map(f => ({
                ...f,
                image: f.image ? Buffer.from(f.image).toString("base64") : null,
                mime: f.mime || "image/jpeg"
            }));

            const lastMeasureDate = measure.length > 0
                ? new Date(measure[0].date).toLocaleString("es-MX", {
                    dateStyle: "short",
                    timeStyle: "short"
                })
                : null;

            const lastPhotoDate = processedPhotos.length > 0
                ? new Date(processedPhotos[0].date).toLocaleString("es-MX", {
                    dateStyle: "short",
                    timeStyle: "short"
                })
                : null;

            res.render("main", { 
                measure, 
                photos: processedPhotos,
                lastMeasureDate,
                lastPhotoDate 
            });
        });
    });
});

// Graph measurements view
app.get("/graphmeasurements", (req, res) => {
    const sql = `
        SELECT temperature, ambient_humidity, substrate_moisture, percentage_light, date 
        FROM measurements 
        WHERE date >= NOW() - INTERVAL 1 DAY
        ORDER BY date ASC
    `;

    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching last 24h data: ", err);
            return res.status(500).send("Error fetching data");
        }

        const labels = results.map(r => 
            new Date(r.date).toLocaleTimeString("es-MX", { hour: "2-digit", minute: "2-digit" })
        );

        const temperature = results.map(r => r.temperature);
        const ambientHumidity = results.map(r => r.ambient_humidity);
        const substrateMoisture = results.map(r => r.substrate_moisture);
        const percentageLight = results.map(r => r.percentage_light);

        res.render("graphmeasurements", {
            labels,
            temperature,
            ambientHumidity,
            substrateMoisture,
            percentageLight
        });
    });
});

// Historical data menu
app.get("/historical", (req, res) => {
    res.render("historical")
})

app.get("/historical/temperature", (req, res) => {
    const { start, end } = req.query;

    if (!start || !end) {
        return res.render("temperature", { labels: [], data: [], start, end });
    }

    const startDate = `${start} 00:00:00`;
    const endDate = `${end} 23:59:59`;

    const sql = `
        SELECT temperature, date 
        FROM measurements 
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
    `;


    db.query(sql, [startDate, endDate], (err, results) => {
        if (err) {
            console.error("Error fetching temperature data: ", err)
            return res.status(500).send("Error feching data")
        }

         console.log("Resultados query:", results);

        const labels = results.map(r =>
            new Date (r.date).toLocaleString("es-MX", {
                dateStyle: "short",
                timeStyle: "short"
            })
        );

        const data = results.map(r => r.temperature)

        res.render("temperature", { labels, data, start, end })
    })
})

// Iniciar app en el puerto 3000
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`)
});
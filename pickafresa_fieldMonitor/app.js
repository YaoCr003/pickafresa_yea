const express = require("express")
const mysql = require("mysql2")
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
    const sqlAlerts = "SELECT type, level, message, date FROM alerts ORDER BY date DESC";
    const sqlNewAlerts = "SELECT COUNT(*) AS total FROM alerts WHERE date >= NOW() - INTERVAL 1 DAY";

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

            db.query(sqlAlerts, (errAlerts, alerts) => {

                if (errAlerts) {
                    console.log("Error in alerts:", errAlerts);
                    alerts = [];
                }

                db.query(sqlNewAlerts, (errNew, rows) => {
                    if (errNew) {
                        console.log("Error counting new alerts:", errNew);
                        rows = [{ total: 0 }];
                    }

                    const newAlertsCount = rows[0].total;

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
                        lastPhotoDate,
                        alerts, 
                        newAlertsCount
                    });
                }); 
            });
        });
    });
});

// Alerts view
app.get("/alerts", (req, res) => {
    const { level, type, q, start, end } = req.query;
    let sql = "SELECT * FROM alerts WHERE 1=1";
    const params = [];

    if (level) {
        sql += " AND level = ?";
        params.push(level);
    }

    if (type) {
        sql += " AND type = ?";
        params.push(type)
    }

    if (q) {
        sql += " AND (message LIKE ? OR type LIKE ?)";
        params.push(`%${q}%`, `%${q}%`);
    }   

    if (start && end) {
        sql += " AND date BETWEEN ? AND ?";
        params.push(`${start} 00:00:00`, `${end} 23:59:59`);
    }

    sql += " ORDER BY date DESC";

    db.query("SELECT DISTINCT type FROM alerts", (errTypes, typesRows) => {
        if (errTypes) {
            console.error("Error fetching alert types:", errTypes);
            return res.status(500).send("Error fetching alert types");
        }

        const types = typesRows.map(r => r.type);

        db.query(sql, params, (err, results) => {
            if (err) {
                console.error("Error fetching alerts:", err);
                return res.status(500).send("Error fetching alerts");
            }

            res.render("alerts", {
                alerts: results,
                filters: { level: level || "", type: type || "", q: q || "", start: start || "", end: end || "" },
                types
            })
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

// Temperature view
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

        const labels = results.map(r =>
            new Date (r.date).toLocaleString("es-MX", {
                dateStyle: "short",
                timeStyle: "short"
            })
        );

        const data = results.map(r => r.temperature)

        res.render("temperature", { labels, data, start, end })
    });
});

// Humidity view
app.get("/historical/humidity", (req, res) => {
    const { start, end } = req.query

    if (!start || !end) {
        return res.render("humidity", { labels: [], data: [], start, end, message: "You must select a date range" });
    }

    const startDate = `${start} 00:00:00`;
    const endDate = `${end} 23:59:59`;

    const sql = `
        SELECT ambient_humidity, substrate_moisture, date 
        FROM measurements 
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
    `;

    db.query(sql, [startDate, endDate], (err, results) => {
        if (err) {
            console.error("Error fetching humidity data: ", err);
            return res.status(500).send("Error fetching data");
        }

        if (results.length === 0) {
            return res.render("humidity", { data: [], start, end, message: "No information found in this range" });
        }

        const data = results.map(r => ({
            ambient: r.ambient_humidity,
            substrate: r.substrate_moisture,
            date: new Date(r.date).toLocaleString("es-MX", {
                dateStyle: "short",
                timeStyle: "short"
            })
        }));

        res.render("humidity", { data, start, end, message: null });
    });
});

//View percentage of light
app.get("/historical/light", (req,res) => {
    const { start, end } = req.query

    if (!start || !end) {
        return res.render("percentageLight", { labels: [], data: [], start, end });
    }

    const startDate = `${start} 00:00:00`;
    const endDate = `${end} 23:59:59`;

    const sql = `
        SELECT percentage_light, date 
        FROM measurements 
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
    `;

    db.query(sql, [startDate, endDate], (err, results) => {
        if (err) {
            console.error("Error fetching percentage ligth data: ", err)
            return res.status(500).send("Error feching data")
        }

        const labels = results.map(r =>
            new Date (r.date).toLocaleString("es-MX", {
                dateStyle: "short",
                timeStyle: "short"
            })
        );

        const data = results.map(r => r.percentage_light)

        res.render("percentageLight", { labels, data, start, end })
    });
});

// Image view
app.get("/historical/images", (req, res) => {
    const { start, end } = req.query;

    if (!start || !end) {
        return res.render("images", { photos: [], start, end, message: "You must select a date range" });
    }

    const startDate = `${start} 00:00:00`;
    const endDate = `${end} 23:59:59`;

    const sql = `
        SELECT image, mime, date 
        FROM images 
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC
        LIMIT 10
    `;

    db.query(sql, [startDate, endDate], (err, results) => {
        if (err) {
            console.error("Error fetching images: ", err);
            return res.status(500).send("Error fetching images");
        }

        if (results.length === 0) {
            return res.render("images", { photos: [], start, end, message: "No information found in this range" });
        }

        const processedPhotos = results.map(f => ({
            image: f.image ? Buffer.from(f.image).toString("base64") : null,
            mime: f.mime || "image/jpeg",
            date: new Date(f.date).toLocaleString("es-MX", {
                dateStyle: "short",
                timeStyle: "short"
            })
        }));

        res.render("images", { photos: processedPhotos, start, end, message: null });
    });
});

//Alerts view

// Iniciar app en el puerto 3000
app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://10.25.87.195:${PORT}`)
});
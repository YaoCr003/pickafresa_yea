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

            res.render("main", { measure, photos: processedPhotos });
        });
    });
});


// Iniciar app en el puerto 3000
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`)
});
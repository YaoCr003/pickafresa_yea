const express = require("express")

const app = express()

// Utilizar plantillas ejs 
app.set("view engine", "ejs")

app.get("", (req,res) => {
    res.render("main")
})

// Iniciar app en el puerto 3000
app.listen(3000, (req,res) => {
    console.log("Corriendo app en el puerto 3000")
})
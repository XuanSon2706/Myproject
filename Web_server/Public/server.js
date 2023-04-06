const express = require('express');
const port = 80;
const mysql = require('mysql');
const bodyParser = require("body-parser");
const path = require('path');
 

//------------------------------------------------------------------------------------------>

var db = [];
var app = express();

//------------------------------------------------------------------------------------------>

var con = mysql.createConnection({
        host: "localhost",
        user: "root",
        password: "Thanhtuyen1",
        database: "mydb"
        });         

//------------------------------------------------------------------------------------------>

app.get('/update', function (req, res) {
        var data = {
            temp: req.query.temp,
            humd: req.query.humd,
            time: new Date()
        };
        con.connect(function(err){
          var sql = "INSERT INTO dashboard SET ? "   
        con.query(sql, data,function (err, result) {
            if (err) throw err; 
                console.log("1 record inserted");
        });
        });    
        db.push (data);
        console.log(data);
        res.end();      
})
app.get('/get', function (req, res) {
        res.writeHead(200, {
            'Content-Type': 'application/json'
        });             
        res.end(JSON.stringify(db));
        db = [];        
})
//------------------------------------------------------------------------------------------>
app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.static('public'));
app.get('/', function(req, res) {
  res.sendFile(path.join(__dirname, '/public/index.html'));
});
//------------------------------------------------------------------------------------------
const server = app.listen(port,function(){
  console.log('connected');
})


var http = require('http');
var fs = require('fs');

function read_file_json(path) {
  if (fs.existsSync(path)) {
    const model_file = fs.readFileSync(path);
    model_params = JSON.parse(model_file);
    return model_params;
  } else {
    throw new Error(`[ERROR]: ${path} does not exists`);
  }
}

var express = require('express');
var app = express();
var cors = require('cors');

app.use(cors());

app.use(express.static(__dirname)); // Current directory is root
//app.use(express.static(path.join(__dirname, 'public'))); //  "public" off of current is root

app.listen(3000);
console.log('Listening on port 80');
					

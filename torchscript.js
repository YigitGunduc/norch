'use strict';
const fs = require('fs');

function load_weights(path) {
  let model_params = fs.readFileSync(path);
  model_params = JSON.parse(model_params);

  return model_params;
}

function matmul(a, b) {
  var aNumRows = a.length, aNumCols = a[0].length,
  bNumRows = b.length, bNumCols = b[0].length,
  m = new Array(aNumRows);  // initialize array of rows
  for (var r = 0; r < aNumRows; ++r) {
    m[r] = new Array(bNumCols); // initialize the current row
    for (var c = 0; c < bNumCols; ++c) {
      m[r][c] = 0;             // initialize the current cell
      for (var i = 0; i < aNumCols; ++i) {
        m[r][c] += a[r][i] * b[i][c];
      }
    }
  }
  return m;
}

function transpose(m) {
  return m[0].map((x, i) => m.map(x => x[i]));
} 

function addmat(m, v) {
  for (let i = 0; i < m[0].length; i++) {
    m[0][i] = m[0][i] + v[i];
  }
  return m;
}

function forward(x, model_params) {
  const weights = (model_params['weights']);
  const biases = (model_params['biases']);

  for (let i = 0; i < weights.length; i++) {
    x = matmul(x, transpose(weights[i]));
    x = addmat(x, biases[i])
  }

  return x[0];
}

const model_params = load_weights('weights.json');

let input = [[1, 1, 1]];

let out = forward(input, model_params)

console.log(out);

'use strict';

// ################################################################################ 
// tensor_ops
// ################################################################################ 

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

function add(m, v) {
  for (let i = 0; i < m[0].length; i++) {
    m[0][i] = m[0][i] + v[i];
  }
  return m;
}

function relu (x) {
  if (Array.isArray(x)) {
    for (let j = 0; j < x.length; j++) {
      x[j].map(e => Math.max(0, e))
    }
    return x;
  }
  return Math.max(0, x);
};

// ################################################################################ 
// norch forward function
// ################################################################################ 

function forward(x, model_params) {
  const weights = model_params['weights'];
  const biases = model_params['biases'];

  for (let i = 0; i < weights.length; i++) {
    x = matmul(x, transpose(weights[i]));
    x = add(x, biases[i])
    if (i != weights.length - 1) {
			x = relu(x)
    }
  }
  return x;
}


// ################################################################################ 
// utils
// ################################################################################ 

function argmax(arr) {
  let max = 0;
  let argm = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i]
      argm = i
    }
  }
  return argm
}

function refresh() {
  window.location.reload();
}

function getCtx() {
  var canvas = document.getElementById('canvas');
  if (!canvas.getContext) console.log("error");
  const context = canvas.getContext('2d');

  return context;
}

function createMatrix(rows, cols) {
  let mat = [];

  for (let i = 0; i < rows; i++) {
     mat[i] = new Array(cols).fill(0);
  }

  return mat;
}

// ################################################################################ 
// webapp
// ################################################################################ 

let loaded = false;
let weights = '';
let drag = false;
const noRows = 28;
const noCols = 28;
const squareSize = 16;
const context = getCtx();
let state = createMatrix(noRows, noCols)
const PARAMS_URL = 'https://raw.githubusercontent.com/yigitgunduc/norch/master/webapp/model/weights.json'


$.getJSON(PARAMS_URL, function( data ) {
  weights = data
  loaded = true;
});

function predict() {
  if (!loaded) return;

  state = [].concat.apply([], state); // flatten
  state = [state] // expand -> (bs, in_size)
  const out = forward(state, weights)
  const arg_max_out = argmax(out[0])

  document.getElementById('res').innerHTML = `${arg_max_out}`;
}

displayGridPaper();

function displayGridPaper() {
  for(var row = 0; row < noRows; row++) {
    for(var col = 0; col < noCols; col++) {
      var x = col * squareSize;
      var y = row * squareSize;

      context.strokeRect(x, y, squareSize, squareSize);
    }
  }
}

function displayCell(x, y) {

    var col = Math.floor( x / squareSize );
    var row = Math.floor( y / squareSize );

    if (col >= noCols || row >= noRows) return;

    state[row][col]     = 1;
    state[row + 1][col] = 1;
    state[row][col + 1] = 1;
    state[row - 1][col] = 1;
    state[row][col - 1] = 1;


    var cellX = col * squareSize;
    var cellY = row * squareSize;
    context.fillRect(cellX, cellY, squareSize, squareSize);

    var cellX = (col + 1) * squareSize;
    var cellY = row * squareSize;
    context.fillRect(cellX, cellY, squareSize, squareSize);

    var cellX = col * squareSize;
    var cellY = (row + 1) * squareSize;
    context.fillRect(cellX, cellY, squareSize, squareSize);

    var cellX = (col - 1) * squareSize;
    var cellY = (row) * squareSize;
    context.fillRect(cellX, cellY, squareSize, squareSize);

    var cellX = (col) * squareSize;
    var cellY = (row - 1) * squareSize;
    context.fillRect(cellX, cellY, squareSize, squareSize);
}

document.addEventListener('mousedown', () => drag = true);
document.addEventListener('mouseup', () => drag = false);
document.addEventListener('mousemove', () => {

  if(drag) {

    let x = window.event.clientX; 
    let y = window.event.clientY; 

    var rect = canvas.getBoundingClientRect();
    x = x - rect.left;
    y = y - rect.top;

    displayCell(x, y)

  }

});

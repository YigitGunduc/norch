'use strict';

// ################################################################################ 
// Activations
// ################################################################################ 

const Identity = function(x) {
    return x;
};

function Inverse (x) {
  if (Array.isArray(x)) {
    return x.map(x => (1 - x));
  }
  return (1 - x);
};

 function Tanh(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.tanh(x));
  }
  return (Math.tanh(x));
};

function ReLU(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.max(0, x));
  }
  return Math.max(0, x);
};

 function Sinusoid(x) {
  if (Array.isArray(x)) {
    x.map(x => Math.sin(x));
  }
  return Math.sin(x);
};

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

// ################################################################################ 
// norch
// ################################################################################ 

function forward(x, model_params) {
  const weights = model_params['weights'];
  const biases = model_params['biases'];

  for (let i = 0; i < weights.length; i++) {
    x = matmul(x, transpose(weights[i]));
    x = add(x, biases[i])
    if (i != weights.length - 1) {
      for (let j = 0; j < x.length; j++) {
        x[j]= x[j].map(e => Math.max(0, e))
      }
    }
  }
  return x;
}

/*
function forward(x, model_params) {
  const w = (model_params['weights']);
  const b = (model_params['biases']);

  for (let i = 0; i < w.length; i++) {
    x = matmul(x, transpose(w[i]));
    x = add(x, b[i])
    if (i < weights.length - 1) {
      x = ReLU(x);
    }
    console.log('pass')
  }
  return x;
}
*/

// ################################################################################ 
// webapp
// ################################################################################ 

const URL = 'http://localhost:3000/weights.json';
let weights = '';
fetch(URL, {
  mode: 'cors',
  headers: {
    'Access-Control-Allow-Origin':'*'
  }
}).then(response => response.text())
  .then((data) => {
    weights = data;
}) 

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

const squareSize = 16;
const noRows = 28;
const noCols = 28;
const context = getCtx();

let drag = false;
let state = createMatrix(noRows, noCols)

function predict() {
  console.log(state)
  state = [].concat.apply([], state);
  weights = JSON.parse(weights);
  console.log(weights)
  const out = forward([state.flat()], weights)
  console.log(out)
  const max = argmax(out[0])
  console.log(max)

  document.getElementById('res').innerHTML = `${max}`;
}

function refresh() {
  window.location.reload();
}

displayGridPaper();

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

function displayCell(x, y) {

    var col = Math.floor( x / squareSize );
    var row = Math.floor( y / squareSize );

    if (col >= noCols || row >= noRows) return;

    state[row][col] = 1;
    //state[row + 1][col] = 1;
    //state[row][col + 1] = 1;
    //state[row - 1][col] = 1;
    //state[row][col - 1] = 1;


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

function displayGridPaper() {
  for(var row = 0; row < noRows; row++) {
    for(var col = 0; col < noCols; col++) {
      var x = col * squareSize;
      var y = row * squareSize;

      context.strokeRect(x, y, squareSize, squareSize);
    }
  }
}

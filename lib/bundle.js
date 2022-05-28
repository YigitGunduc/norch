'use strict';

// ################################################################################ 
// norch
// ################################################################################ 

class Model {
  constructor(opt) {
    if (opt.activation) {
      this.activation = opt.activation;
    } else {
      this.activation = identity;
    }
    this.path = opt.path;
    this.load_weights();
  }

  load_weights() {
    this.model_params = this.path;
    //if (fs.existsSync(this.path)) {
      //const model_file = fs.readFileSync(this.path);
      //this.model_params = JSON.parse(model_file);

    //} else {
      //throw new Error(`[ERROR]: ${path} does not exists`);
    //}
  }

  forward(x) {
    const weights = (this.model_params['weights']);
    const biases = (this.model_params['biases']);

    for (let i = 0; i < weights.length; i++) {
      x = matmul(x, transpose(weights[i]));
      x = add(x, biases[i])
      if (i < weights.lenght - 1) {
        x = this.activation(x);
      }
    }
    return x;
  }
}

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
// Activations
// ################################################################################ 

Identity = function(x) {
    return x;
};

Inverse = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => (1 - x));
  }
  return (1 - x);
};

Tanh = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.tanh(x));
  }
  return (Math.tanh(x));
};

ReLU = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.max(0, x));
  }
  return Math.max(0, x);
};

BinaryStep = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => ((x < 0) ? 0 : 1));
  }
  return ((x < 0) ? 0 : 1);
};

Logistic = Sigmoid = SoftStep = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => 1 / (1 + Math.exp(-x)));
  }
  return 1 / (1 + Math.exp(-x));
};

SiLU = Swish1 = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => x * Sigmoid(x));
  }
  return x * Sigmoid(x);
};

ArcTan = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.atan(x));
  }
  return Math.atan(x);
};

PReLU = function(x, a) {
  if (Array.isArray(x)) {
    return x.map(x => ((x < 0) ? (a * x) : x));
  }
  return ((x < 0) ? (a * x) : x);
};

ELU = function(x, a) {
  if (Array.isArray(x)) {
    return x.map(x => ((x > 0) ? $x : (a*Math.expm1(x))));
  }
  return ((x > 0) ? x : ($a*Math.expm1(x)));
};

SELU = function(x) {
  if (Array.isArray(x)) {
    x.map(x => 1.0507 * ELU(x, 1.67326));
  }
  return 1.0507 * ELU(x, 1.67326);
};

Sinusoid = function(x) {
  if (Array.isArray(x)) {
    x.map(x => Math.sin(x));
  }
  return Math.sin(x);
};

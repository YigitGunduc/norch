'use strict';

const fs = require('fs');
const { transpose, matmul, add } = require('./tensor_ops');
const { identity } = require('./activations')

class Model {
  constructor(path, activation) {
    if (activation) {
      this.activation = activation;
    } else {
      this.activation = identity;
    }
    this.path = path;
    this.load_weights();
  }

  load_weights() {
    if (fs.existsSync(this.path)) {
      const model_file = fs.readFileSync(this.path);
      this.model_params = JSON.parse(model_file);

    } else {
      throw new Error(`[ERROR]: ${path} does not exists`);
    }
  }

  forward(x) {
    const weights = this.model_params['weights'];
    const biases = this.model_params['biases'];

    for (let i = 0; i < weights.length; i++) {
      x = matmul(x, transpose(weights[i]));
      x = add(x, biases[i])
      if (i != weights.length - 1) {
        x = this.activation(x);
      }
    }
    return x;
  }
}

module.exports = {
  Model
};
